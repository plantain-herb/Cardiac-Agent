"""
SA (Short-Axis) LGE Heart Segmentation Model Worker
A FastAPI service that executes the short-axis LGE heart segmentation model.

Usage:
    python -m serve.lge_sa_seg_worker --gpu 0 --port 21013 --no-register
"""
import sys
import os

# Get the directory paths
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_SRC_DIR = os.path.join(SRC_DIR, "LGE_SA_SEG")

# Add paths for imports
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODEL_SRC_DIR)

from app.config import (
    WEIGHTS_DIR,
    EXPERT_DIR_LGE_SA_FIRST,
    EXPERT_DIR_LGE_SA_SECOND,
    EXPERT_CKPT_LGE_SAX_SEG1,
    EXPERT_CKPT_LGE_SAX_SEG2,
    expert_weight_path,
)

import argparse
import asyncio
import time
import threading
import uuid
import traceback
import tempfile
import base64
from pathlib import Path

import h11  # noqa: F401

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import SimpleITK as sitk
import requests
import torch
import uvicorn

# Import from src/LGE_SA_SEG/infer/
from infer.predictor_LGE_seg import LGESegHeartModel, LGESegHeartPredictor

# Try to import from MMedAgent serve utilities
try:
    from serve.constants import WORKER_HEART_BEAT_INTERVAL, ErrorCode, SERVER_ERROR_MSG
    from serve.utils import build_logger, pretty_print_semaphore
except ImportError:
    WORKER_HEART_BEAT_INTERVAL = 45
    SERVER_ERROR_MSG = "**SERVER ERROR. PLEASE TRY AGAIN.**"
    
    class ErrorCode:
        INTERNAL_ERROR = 50001
        CUDA_OUT_OF_MEMORY = 50002
    
    import logging
    def build_logger(name, filename):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    
    def pretty_print_semaphore(sem):
        if sem is None:
            return "None"
        return f"Semaphore(value={sem._value})"


worker_id = str(uuid.uuid4())[:6]
logger = build_logger("lge_sa_seg_worker", os.path.join("workers", "lge_sa_seg.log"))
global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):
    """Send heartbeat to controller periodically."""
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class LGESegSAWorker:
    """
    Worker class for Short-Axis LGE heart segmentation.
    Handles model loading, inference, and communication with controller.
    """
    
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        no_register: bool,
        model_names: list,
        model_config: dict,
        device: str,
        gpu: int = 0,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device
        self.gpu = gpu
        self.model_config = model_config
        
        # Load the segmentation model
        logger.info("Loading SA LGE heart segmentation model...")
        self._load_model()
        logger.info("Model loaded successfully!")
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.daemon = True
            self.heart_beat_thread.start()
    
    def _load_model(self):
        """Load the LGE segmentation model and predictor."""
        model_seg = LGESegHeartModel(
            model_heart_f=self.model_config["model_file_heart_first"],
            model_heart_crop_f=self.model_config["model_file_heart_second"],
            network_heart_f=self.model_config["network_file_heart_first"],
            network_heart_crop_f=self.model_config["network_file_heart_second"],
            config_f=self.model_config["config_file"],
        )
        self.predictor = LGESegHeartPredictor(gpu=self.gpu, model=model_seg)
    
    def register_to_controller(self):
        """Register this worker to the controller."""
        logger.info("Register to controller")
        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        try:
            r = requests.post(url, json=data, timeout=10)
            assert r.status_code == 200
        except Exception as e:
            logger.error(f"Failed to register to controller: {e}")
    
    def send_heart_beat(self):
        """Send heartbeat to controller."""
        logger.info(
            f"Send heart beat. Models: {self.model_names}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}. "
            f"worker_id: {self.worker_id}. "
        )
        
        url = self.controller_addr + "/receive_heart_beat"
        
        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_addr,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)
        
        if not exist:
            self.register_to_controller()
    
    def get_queue_length(self):
        """Get current queue length."""
        if (
            model_semaphore is None
            or model_semaphore._value is None
            or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                args.limit_model_concurrency
                - model_semaphore._value
                + len(model_semaphore._waiters)
            )
    
    def get_status(self):
        """Get worker status."""
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def load_image(self, image_input: str) -> tuple:
        """
        Load medical image from file path or base64 encoded data.
        
        Args:
            image_input: File path (.nii.gz) or base64 encoded nii.gz data
            
        Returns:
            tuple: (sitk_image, hu_volume, spacing)
        """
        if os.path.exists(image_input):
            if image_input.endswith(".nii.gz") or image_input.endswith(".nii"):
                sitk_img = sitk.ReadImage(image_input)
            else:
                reader = sitk.ImageSeriesReader()
                names = reader.GetGDCMSeriesFileNames(image_input)
                reader.SetFileNames(names)
                sitk_img = reader.Execute()
        else:
            try:
                decoded_data = base64.b64decode(image_input)
                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                    tmp.write(decoded_data)
                    tmp_path = tmp.name
                sitk_img = sitk.ReadImage(tmp_path)
                os.unlink(tmp_path)
            except Exception as e:
                raise ValueError(f"Failed to decode image data: {e}")
        
        spacing = np.array(sitk_img.GetSpacing()[::-1])
        hu_volume = sitk.GetArrayFromImage(sitk_img)
        
        return sitk_img, hu_volume, spacing
    
    def save_segmentation_image(self, hu_volume: np.ndarray, seg_mask: np.ndarray, 
                                  output_path: str, slice_idx: int = None):
        """
        Save segmentation visualization as PNG image.
        
        Args:
            hu_volume: Original image volume
            seg_mask: Segmentation mask
            output_path: Path to save the PNG image
            slice_idx: Slice index to visualize (default: middle slice)
        """
        try:
            from PIL import Image
            
            # Select slice to visualize
            if slice_idx is None:
                slice_idx = hu_volume.shape[0] // 2
            
            # Get the slice
            img_slice = hu_volume[slice_idx].astype(np.float32)
            mask_slice = seg_mask[slice_idx].astype(np.uint8)
            
            # Normalize image to 0-255
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8) * 255
            img_slice = img_slice.astype(np.uint8)
            
            # Create RGB image
            img_rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
            
            # Define colors for LGE labels (RGBA)
            colors = {
                1: [255, 0, 0, 128],      # Red - LV cavity
                2: [0, 255, 0, 128],      # Green - LV myocardium
                3: [255, 255, 0, 128],    # Yellow - Scar
                4: [0, 0, 255, 128],      # Blue - Other
            }
            
            # Overlay segmentation mask
            overlay = img_rgb.copy().astype(np.float32)
            for label, color in colors.items():
                mask = (mask_slice == label)
                if np.any(mask):
                    alpha = color[3] / 255.0
                    for c in range(3):
                        overlay[:, :, c][mask] = (
                            overlay[:, :, c][mask] * (1 - alpha) + color[c] * alpha
                        )
            
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Save image
            img = Image.fromarray(overlay)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            logger.info(f"Saved segmentation visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save segmentation image: {e}")
    
    @torch.inference_mode()
    def segment(self, params: dict) -> dict:
        """
        Perform LGE heart segmentation.
        
        Args:
            params: Dictionary containing:
                - image: File path or base64 encoded nii.gz data
                - output_path: Optional path to save output nii.gz
                - image_output_path: Optional path to save visualization PNG
                
        Returns:
            dict: Segmentation results including mask or file path
        """
        try:
            image_input = params.get("image")
            output_path = params.get("output_path", None)
            image_output_path = params.get("image_output_path", None)
            
            if not image_input:
                return {
                    "error": "No image provided",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
            
            # Load image
            sitk_img, hu_volume, spacing = self.load_image(image_input)
            
            # Run segmentation (LGE uses predict without spacing)
            logger.info(f"Running LGE segmentation on image with shape {hu_volume.shape}")
            seg_mask = self.predictor.predict(hu_volume)
            logger.info(f"Segmentation complete. Unique labels: {np.unique(seg_mask)}")
            
            result = {
                "shape": list(seg_mask.shape),
                "unique_labels": [int(x) for x in np.unique(seg_mask)],
                "error_code": 0,
            }
            
            # Save or encode output
            if output_path:
                seg_img = sitk.GetImageFromArray(seg_mask.astype(np.uint8))
                seg_img.CopyInformation(sitk_img)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sitk.WriteImage(seg_img, output_path)
                result["output_path"] = output_path
                logger.info(f"Saved segmentation mask to {output_path}")
            else:
                seg_img = sitk.GetImageFromArray(seg_mask.astype(np.uint8))
                seg_img.CopyInformation(sitk_img)
                
                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                    sitk.WriteImage(seg_img, tmp.name)
                    with open(tmp.name, "rb") as f:
                        encoded_mask = base64.b64encode(f.read()).decode("utf-8")
                    os.unlink(tmp.name)
                
                result["mask_base64"] = encoded_mask
            
            # Save visualization image if requested
            if image_output_path:
                self.save_segmentation_image(hu_volume, seg_mask, image_output_path)
                result["image_output_path"] = image_output_path
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            return {
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except Exception as e:
            logger.error(f"Segmentation error: {traceback.format_exc()}")
            return {
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
    
    def generate_gate(self, params: dict) -> dict:
        """Entry point for segmentation requests."""
        return self.segment(params)


# FastAPI Application
app = FastAPI(title="SA LGE Heart Segmentation Service")


def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


@app.post("/worker_generate")
async def api_generate(request: Request):
    """
    Main API endpoint for LGE segmentation.
    
    Request body:
        - image: str, file path or base64 encoded nii.gz
        - output_path: str, optional path to save output
    
    Returns:
        - shape: list, shape of segmentation mask
        - unique_labels: list, unique label values in mask
        - mask_base64: str, base64 encoded result (if no output_path)
        - output_path: str, path to saved file (if output_path provided)
    """
    params = await request.json()
    await acquire_model_semaphore()
    try:
        output = worker.generate_gate(params)
    finally:
        release_model_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    """Get worker status."""
    return worker.get_status()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_names": worker.model_names}


@app.post("/model_details")
async def model_details(request: Request):
    """Get model details."""
    return {
        "model_names": worker.model_names,
        "task": "SA (Short-Axis) LGE Heart Segmentation",
        "input_format": "nii.gz (file path or base64)",
        "output_format": "nii.gz (file path or base64)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA LGE Heart Segmentation Worker")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21013)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21013")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    
    # Model configuration
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Model paths
    MODEL_BASE_DIR = Path(MODEL_SRC_DIR)
    
    parser.add_argument(
        "--model-file-heart-first",
        type=str,
        default=expert_weight_path(EXPERT_DIR_LGE_SA_FIRST, EXPERT_CKPT_LGE_SAX_SEG1),
    )
    parser.add_argument(
        "--model-file-heart-second",
        type=str,
        default=expert_weight_path(EXPERT_DIR_LGE_SA_SECOND, EXPERT_CKPT_LGE_SAX_SEG2),
    )
    parser.add_argument(
        "--network-file-heart-first",
        type=str,
        default=os.path.join(MODEL_BASE_DIR, "train/config/seg_LGE_first_config.py"),
    )
    parser.add_argument(
        "--network-file-heart-second",
        type=str,
        default=os.path.join(MODEL_BASE_DIR, "train/config/seg_LGE_second_config.py"),
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=os.path.join(MODEL_BASE_DIR, "example/heart_seg.yaml"),
    )
    
    # Worker configuration
    parser.add_argument(
        "--model-names",
        default="LgeSAXSegmentation",
        type=lambda s: s.split(","),
        help="Model names (comma separated)",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true", 
                        help="Don't register to controller")
    
    args = parser.parse_args()
    logger.info(f"args: {args}")
    
    # Build model config
    model_config = {
        "model_file_heart_first": args.model_file_heart_first,
        "model_file_heart_second": args.model_file_heart_second,
        "network_file_heart_first": args.network_file_heart_first,
        "network_file_heart_second": args.network_file_heart_second,
        "config_file": args.config_file,
    }
    
    # Create worker
    worker = LGESegSAWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_names=args.model_names,
        model_config=model_config,
        device=args.device,
        gpu=args.gpu,
    )
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

