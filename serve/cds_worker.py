"""
CC CINE Heart Classification Model Worker
A FastAPI service for Cardiomyopathy classification using CINE sequences.

CC分类使用的输入:
    - CINE 2CH (cine_2ch)
    - CINE 4CH (cine_4ch)
    - CINE SA (cine_sa)

Pipeline:
    1. Segmentation - call segmentation workers for each view
    2. Phase processing - keep middle N blocks (2ch/4ch: 3 blocks, sa: 9 blocks)
    3. Resample - resample to target spacing based on modality
    4. Crop - crop based on segmentation mask
    5. Classification - run classification model

Input: 3 volumes (2CH CINE, 4CH CINE, SA CINE) + slice_num parameters
Output: Classification result (0: Normal, 1: Ischemic, 2: Non-ischemic)

Usage:
    python -m serve.cds_worker --gpu 0 --port 21020 --no-register
"""
import sys
import os

# Get the directory paths
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_SRC_DIR = os.path.join(SRC_DIR, "CDS")

# Add paths for imports
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODEL_SRC_DIR)

from app.config import WEIGHTS_DIR

import argparse
import asyncio
import time
import threading
import uuid
import traceback
import tempfile
import base64
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import h11  # noqa: F401

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import SimpleITK as sitk
import requests
import torch
import uvicorn

# Import from src/CC/heart_diagnosis/infer/
from infer.predictor_cine_class import CineClassificationModel, CineClassificationPredictor

# Target spacing for different modalities
TARGET_SPACING = {
    "2ch": (0.9375, 0.9375, 0.1123),
    "4ch": (0.9375, 0.9375, 0.1123),
    "sa": (0.9375, 0.9375, 0.2451),
    "lge_sa": (0.8438, 0.8438, 2.5142),
}

# Segmentation worker endpoints
SEG_WORKERS = {
    "2ch": "http://localhost:21010",  # Cine2CHSegWorker
    "4ch": "http://localhost:21011",  # Cine4CHSegWorker
    "sa": "http://localhost:21012",   # CineSAXSegWorker
}

# Number of blocks to keep for each modality
BLOCKS_TO_KEEP = {
    "2ch": 3,
    "4ch": 3,
    "sa": 9,
    "lge_sa": 9,
}

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
logger = build_logger("cds_worker", os.path.join("workers", "cds.log"))
global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):
    """Send heartbeat to controller periodically."""
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class CCCineClassWorker:
    """
    Worker class for CC CINE heart classification.
    Handles model loading, inference, and communication with controller.
    
    Pipeline:
        1. Segmentation - call segmentation workers for each view
        2. Phase processing - keep middle N blocks based on modality
        3. Resample - resample to target spacing
        4. Crop - crop based on segmentation mask
        5. Classification - run classification model
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
        seg_workers: dict = None,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.device = device
        self.gpu = gpu
        self.model_config = model_config
        self.seg_workers = seg_workers or SEG_WORKERS
        
        # Load the classification model
        logger.info("Loading CC CINE classification model...")
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
        """Load the classification model and predictor."""
        model_cls = CineClassificationModel(
            model_f=self.model_config["model_cls_file"],
            network_f=self.model_config["network_cls_file"],
            config_f=self.model_config["config_file"],
        )
        self.predictor = CineClassificationPredictor(gpu=self.gpu, model=model_cls)
    
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
    
    def load_image(self, image_input: str) -> Tuple[sitk.Image, np.ndarray]:
        """
        Load medical image from file path or base64 encoded data.
        
        Args:
            image_input: File path (.nii.gz) or base64 encoded nii.gz data
            
        Returns:
            Tuple[sitk.Image, np.ndarray]: (sitk_image, volume_data)
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
        
        hu_volume = sitk.GetArrayFromImage(sitk_img)
        return sitk_img, hu_volume.astype(np.float32)
    
    def call_segmentation_worker(self, modality: str, image_path: str, output_path: str = None) -> Dict:
        """
        Call segmentation worker to get segmentation mask.
        
        Args:
            modality: "2ch", "4ch", or "sa"
            image_path: Path to input image
            output_path: Optional path to save segmentation result
            
        Returns:
            Dict: Segmentation result with mask_base64 or output_path
        """
        worker_url = self.seg_workers.get(modality)
        if not worker_url:
            raise ValueError(f"No segmentation worker for modality: {modality}")
        
        params = {"image": image_path}
        if output_path:
            params["output_path"] = output_path
        
        logger.info(f"Calling segmentation worker for {modality}: {worker_url}")
        resp = requests.post(f"{worker_url}/worker_generate", json=params, timeout=300)
        result = resp.json()
        
        if result.get("error_code", -1) != 0:
            raise RuntimeError(f"Segmentation failed for {modality}: {result.get('error')}")
        
        logger.info(f"Segmentation complete for {modality}. Labels: {result.get('unique_labels')}")
        return result
    
    def load_segmentation_mask(self, seg_result: Dict, modality: str) -> np.ndarray:
        """
        Load segmentation mask from result dict.
        
        Args:
            seg_result: Result from segmentation worker
            modality: Modality type
            
        Returns:
            np.ndarray: Segmentation mask
        """
        if "output_path" in seg_result and os.path.exists(seg_result["output_path"]):
            sitk_seg = sitk.ReadImage(seg_result["output_path"])
            return sitk.GetArrayFromImage(sitk_seg)
        elif "mask_base64" in seg_result:
            decoded_data = base64.b64decode(seg_result["mask_base64"])
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(decoded_data)
                tmp_path = tmp.name
            sitk_seg = sitk.ReadImage(tmp_path)
            os.unlink(tmp_path)
            return sitk.GetArrayFromImage(sitk_seg)
        else:
            raise ValueError(f"No segmentation mask found for {modality}")
    
    def phase_processing(self, data: np.ndarray, phase: int, modality: str) -> np.ndarray:
        """
        Perform phase processing - keep middle N blocks.
        
        Args:
            data: Input volume data (Z, H, W)
            phase: Number of frames per phase
            modality: "2ch", "4ch", "sa", or "lge_sa"
            
        Returns:
            np.ndarray: Processed volume with middle blocks kept
        """
        z_frames = data.shape[0]
        total_layers = z_frames // phase
        blocks_to_keep = BLOCKS_TO_KEEP.get(modality, 3)
        
        if total_layers <= blocks_to_keep:
            logger.info(f"Phase processing: layers({total_layers}) <= {blocks_to_keep}, skip")
            return data
        
        del_layers = total_layers - blocks_to_keep
        del_back = del_layers // 2 * phase
        del_front = (del_layers - del_layers // 2) * phase  # 心尖多去点
        
        cropped_data = data[del_front:z_frames - del_back, :, :]
        logger.info(f"Phase processing: {data.shape} -> {cropped_data.shape} (kept {blocks_to_keep} blocks)")
        return cropped_data
    
    def resample_volume(self, sitk_img: sitk.Image, modality: str) -> Tuple[sitk.Image, np.ndarray]:
        """
        Resample volume to target spacing.
        
        Args:
            sitk_img: Input SimpleITK image
            modality: "2ch", "4ch", "sa", or "lge_sa"
            
        Returns:
            Tuple[sitk.Image, np.ndarray]: (resampled_sitk_img, resampled_volume)
        """
        target_spacing = TARGET_SPACING.get(modality)
        if not target_spacing:
            raise ValueError(f"No target spacing for modality: {modality}")
        
        original_size = sitk_img.GetSize()
        original_spacing = sitk_img.GetSpacing()
        
        new_size = [
            int(original_size[0] * original_spacing[0] / target_spacing[0]),
            int(original_size[1] * original_spacing[1] / target_spacing[1]),
            int(original_size[2] * original_spacing[2] / target_spacing[2])
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_img.GetDirection())
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        resample.SetSize(new_size)
        resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetOutputSpacing(target_spacing)
        
        resampled_img = resample.Execute(sitk_img)
        resampled_volume = sitk.GetArrayFromImage(resampled_img).astype(np.float32)
        
        logger.info(f"Resample: {original_size} -> {new_size}, spacing: {original_spacing} -> {target_spacing}")
        return resampled_img, resampled_volume
    
    def resample_mask(self, sitk_mask: sitk.Image, modality: str) -> Tuple[sitk.Image, np.ndarray]:
        """
        Resample segmentation mask to target spacing using nearest neighbor.
        
        Args:
            sitk_mask: Input SimpleITK segmentation mask
            modality: "2ch", "4ch", "sa", or "lge_sa"
            
        Returns:
            Tuple[sitk.Image, np.ndarray]: (resampled_sitk_mask, resampled_mask)
        """
        target_spacing = TARGET_SPACING.get(modality)
        if not target_spacing:
            raise ValueError(f"No target spacing for modality: {modality}")
        
        original_size = sitk_mask.GetSize()
        original_spacing = sitk_mask.GetSpacing()
        
        new_size = [
            int(original_size[0] * original_spacing[0] / target_spacing[0]),
            int(original_size[1] * original_spacing[1] / target_spacing[1]),
            int(original_size[2] * original_spacing[2] / target_spacing[2])
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_mask.GetDirection())
        resample.SetOutputOrigin(sitk_mask.GetOrigin())
        resample.SetSize(new_size)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetOutputSpacing(target_spacing)
        
        resampled_mask = resample.Execute(sitk_mask)
        resampled_mask_arr = sitk.GetArrayFromImage(resampled_mask)
        
        return resampled_mask, resampled_mask_arr
    
    def crop_with_mask(self, image: np.ndarray, mask: np.ndarray, crop_size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """
        Crop image based on segmentation mask center.
        
        Args:
            image: Input volume (Z, H, W)
            mask: Segmentation mask (Z, H, W)
            crop_size: Target crop size (H, W)
            
        Returns:
            np.ndarray: Cropped volume
        """
        if np.sum(mask) == 0:
            logger.warning("Empty mask, returning zero crop")
            return np.zeros((image.shape[0], crop_size[0], crop_size[1]), dtype=image.dtype)
        
        non_zero_coords = np.argwhere(mask > 0)
        z_min, y_min, x_min = non_zero_coords.min(axis=0)
        z_max, y_max, x_max = non_zero_coords.max(axis=0)
        
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        
        crop_height, crop_width = crop_size
        y_start = max(center_y - crop_height // 2, 0)
        y_end = min(center_y + crop_height // 2, image.shape[1])
        x_start = max(center_x - crop_width // 2, 0)
        x_end = min(center_x + crop_width // 2, image.shape[2])
        
        cropped_image = image[:, y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        pad_y_before = max(0, crop_height // 2 - center_y)
        pad_y_after = max(0, crop_height // 2 - (image.shape[1] - center_y))
        pad_x_before = max(0, crop_width // 2 - center_x)
        pad_x_after = max(0, crop_width // 2 - (image.shape[2] - center_x))
        
        cropped_image = np.pad(
            cropped_image,
            ((0, 0), (pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
            mode='constant',
            constant_values=0
        )
        
        logger.info(f"Crop: {image.shape} -> {cropped_image.shape}")
        return cropped_image
    
    def preprocess_volume(
        self, 
        image_input: str, 
        modality: str, 
        phase: int,
        seg_output_path: str = None
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for a single volume.
        
        Pipeline:
            1. Load image
            2. Call segmentation worker
            3. Phase processing (keep middle blocks)
            4. Resample to target spacing
            5. Crop based on segmentation mask
        
        Args:
            image_input: Path to input image
            modality: "2ch", "4ch", or "sa"
            phase: Number of frames per phase
            seg_output_path: Optional path to save segmentation result
            
        Returns:
            np.ndarray: Preprocessed volume ready for classification
        """
        logger.info(f"Preprocessing {modality} volume with phase={phase}")
        
        # Step 1: Load original image
        sitk_img, volume = self.load_image(image_input)
        logger.info(f"Loaded {modality}: shape={volume.shape}")
        
        # Step 2: Call segmentation worker
        seg_result = self.call_segmentation_worker(modality, image_input, seg_output_path)
        seg_mask = self.load_segmentation_mask(seg_result, modality)
        
        # Step 3: Phase processing - keep middle blocks
        volume_phased = self.phase_processing(volume, phase, modality)
        seg_mask_phased = self.phase_processing(seg_mask.astype(np.float32), phase, modality)
        
        # Create new sitk images for phased data
        sitk_img_phased = sitk.GetImageFromArray(volume_phased)
        sitk_img_phased.SetSpacing(sitk_img.GetSpacing())
        sitk_img_phased.SetDirection(sitk_img.GetDirection())
        sitk_img_phased.SetOrigin(sitk_img.GetOrigin())
        
        sitk_mask_phased = sitk.GetImageFromArray(seg_mask_phased.astype(np.uint8))
        sitk_mask_phased.SetSpacing(sitk_img.GetSpacing())
        sitk_mask_phased.SetDirection(sitk_img.GetDirection())
        sitk_mask_phased.SetOrigin(sitk_img.GetOrigin())
        
        # Step 4: Resample to target spacing
        _, volume_resampled = self.resample_volume(sitk_img_phased, modality)
        _, mask_resampled = self.resample_mask(sitk_mask_phased, modality)
        
        # Step 5: Crop based on segmentation mask
        volume_cropped = self.crop_with_mask(volume_resampled, mask_resampled)
        
        logger.info(f"Preprocessing complete for {modality}: final shape={volume_cropped.shape}")
        return volume_cropped
    
    @torch.inference_mode()
    def classify(self, params: dict) -> dict:
        """
        Perform heart classification with full preprocessing pipeline.
        
        Pipeline:
            1. Segmentation - call segmentation workers for each view
            2. Phase processing - keep middle N blocks
            3. Resample - resample to target spacing
            4. Crop - crop based on segmentation mask
            5. Classification - run classification model
        
        Args:
            params: Dictionary containing:
                - image_2ch: File path or base64 for 2CH volume
                - image_4ch: File path or base64 for 4CH volume
                - image_sa: File path or base64 for SA volume
                - slice_num_2ch: Number of slices for 2CH
                - slice_num_4ch: Number of slices for 4CH
                - slice_num_sa: Number of slices for SA
                - num_crops: Number of crops for sliding window (default: 3)
                - seg_output_2ch: Optional path to save 2CH segmentation
                - seg_output_4ch: Optional path to save 4CH segmentation
                - seg_output_sa: Optional path to save SA segmentation
                - skip_preprocess: If True, skip preprocessing (use raw volumes)
                
        Returns:
            dict: Classification results
        """
        try:
            image_2ch = params.get("image_2ch")
            image_4ch = params.get("image_4ch")
            image_sa = params.get("image_sa")
            
            # Slice number parameters (default to 1 if not provided - no phase processing)
            slice_num_2ch = params.get("slice_num_2ch", 1)
            slice_num_4ch = params.get("slice_num_4ch", 1)
            slice_num_sa = params.get("slice_num_sa", 1)
            
            num_crops = params.get("num_crops", 3)
            skip_preprocess = params.get("skip_preprocess", False)
            
            # Optional segmentation output paths
            seg_output_2ch = params.get("seg_output_2ch")
            seg_output_4ch = params.get("seg_output_4ch")
            seg_output_sa = params.get("seg_output_sa")
            
            # 必须有 4ch 和 sa，2ch 可选（如果缺失则用空图像占位）
            if not all([image_4ch, image_sa]):
                return {
                    "error": "4CH and SA images are required (2CH is optional)",
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
            
            # 检查 2ch 是否缺失
            use_placeholder_2ch = image_2ch is None
            if use_placeholder_2ch:
                logger.info("2CH image not provided, will use placeholder (zeros)")
            
            if skip_preprocess:
                # Legacy mode: just load and classify without preprocessing
                logger.info("Skip preprocessing mode - loading raw volumes")
                _, vol_4ch = self.load_image(image_4ch)
                _, vol_sa = self.load_image(image_sa)
                
                if use_placeholder_2ch:
                    # 创建与 4ch 相同形状的空图像作为占位
                    vol_2ch = np.zeros_like(vol_4ch)
                    logger.info(f"Created placeholder 2CH with shape: {vol_2ch.shape}")
                else:
                    _, vol_2ch = self.load_image(image_2ch)
            else:
                # Full preprocessing pipeline
                logger.info("Running full preprocessing pipeline...")
                logger.info(f"Slice num values: 2CH={slice_num_2ch}, 4CH={slice_num_4ch}, SA={slice_num_sa}")
                
                vol_4ch = self.preprocess_volume(image_4ch, "4ch", slice_num_4ch, seg_output_4ch)
                vol_sa = self.preprocess_volume(image_sa, "sa", slice_num_sa, seg_output_sa)
                
                if use_placeholder_2ch:
                    # 创建标准形状的空图像作为 2ch 占位符
                    # 2ch 预处理后的标准形状：根据 crop_shapes 为 [80, 192, 192]
                    vol_2ch = np.zeros((80, 192, 192), dtype=np.float32)
                    logger.info(f"Created placeholder 2CH with shape: {vol_2ch.shape}")
                else:
                    vol_2ch = self.preprocess_volume(image_2ch, "2ch", slice_num_2ch, seg_output_2ch)
            
            vols = [vol_2ch, vol_4ch, vol_sa]
            
            # Run classification
            logger.info(f"Running classification with shapes: 2CH={vol_2ch.shape}, 4CH={vol_4ch.shape}, SA={vol_sa.shape}")
            preds, avg_pred = self.predictor.predict(vols, num_crops)
            
            pred_class = int(np.argmax(avg_pred))
            logger.info(f"Classification complete. Predicted class: {pred_class}")
            
            result = {
                "pred_class": pred_class,
                "avg_pred": avg_pred.tolist(),
                "preds": preds.tolist(),
                "preprocessing": {
                    "slice_num_2ch": slice_num_2ch,
                    "slice_num_4ch": slice_num_4ch,
                    "slice_num_sa": slice_num_sa,
                    "skip_preprocess": skip_preprocess,
                },
                "error_code": 0,
            }
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM: {e}")
            return {
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        except Exception as e:
            logger.error(f"Classification error: {traceback.format_exc()}")
            return {
                "error": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
    
    def generate_gate(self, params: dict) -> dict:
        """Entry point for classification requests."""
        return self.classify(params)


# FastAPI Application
app = FastAPI(title="CC CINE Heart Classification Service")


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
    Main API endpoint for classification with full preprocessing pipeline.
    
    Pipeline:
        1. Segmentation - call segmentation workers for each view
        2. Phase processing - keep middle N blocks (2ch/4ch: 3, sa: 9)
        3. Resample - resample to target spacing
        4. Crop - crop based on segmentation mask
        5. Classification - run classification model
    
    Request body:
        - image_2ch: str, file path or base64 encoded nii.gz for 2CH
        - image_4ch: str, file path or base64 encoded nii.gz for 4CH
        - image_sa: str, file path or base64 encoded nii.gz for SA
        - slice_num_2ch: int, number of slices for 2CH (default: 1)
        - slice_num_4ch: int, number of slices for 4CH (default: 1)
        - slice_num_sa: int, number of slices for SA (default: 1)
        - num_crops: int, number of crops (default: 3)
        - seg_output_2ch: str, optional path to save 2CH segmentation
        - seg_output_4ch: str, optional path to save 4CH segmentation
        - seg_output_sa: str, optional path to save SA segmentation
        - skip_preprocess: bool, skip preprocessing if True (default: False)
    
    Returns:
        - pred_class: int, predicted class
        - avg_pred: list, average prediction probabilities
        - preds: list, all crop predictions
        - preprocessing: dict, preprocessing parameters used
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
        "task": "CC CINE Heart Classification",
        "input_format": "3 nii.gz files (2CH, 4CH, SA) + phase parameters",
        "output_format": "Classification result",
        "pipeline": [
            "1. Segmentation (call segmentation workers)",
            "2. Phase processing (keep middle blocks)",
            "3. Resample to target spacing",
            "4. Crop based on segmentation mask",
            "5. Classification"
        ],
        "target_spacing": TARGET_SPACING,
        "blocks_to_keep": BLOCKS_TO_KEEP,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CC CINE Heart Classification Worker")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21020)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21020")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    
    # Model configuration
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Model paths
    MODEL_BASE_DIR = Path(MODEL_SRC_DIR)
    WEIGHTS_PATH = Path(WEIGHTS_DIR)
    
    parser.add_argument(
        "--model-cls-file",
        type=str,
        default=os.path.join(WEIGHTS_PATH, "diagnosis_first/epoch_53.pth"),
    )
    parser.add_argument(
        "--network-cls-file",
        type=str,
        default=os.path.join(MODEL_BASE_DIR, "train/config/cine_class_config_5fold.py"),
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=os.path.join(MODEL_BASE_DIR, "example/cls.yaml"),
    )
    
    # Segmentation worker configuration
    parser.add_argument(
        "--seg-worker-2ch",
        type=str,
        default="http://localhost:21010",
        help="Segmentation worker URL for 2CH",
    )
    parser.add_argument(
        "--seg-worker-4ch",
        type=str,
        default="http://localhost:21011",
        help="Segmentation worker URL for 4CH",
    )
    parser.add_argument(
        "--seg-worker-sa",
        type=str,
        default="http://localhost:21012",
        help="Segmentation worker URL for SA",
    )
    
    # Worker configuration
    parser.add_argument(
        "--model-names",
        default="CardiacDiseaseScreening",
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
        "model_cls_file": args.model_cls_file,
        "network_cls_file": args.network_cls_file,
        "config_file": args.config_file,
    }
    
    # Build segmentation worker config
    seg_workers = {
        "2ch": args.seg_worker_2ch,
        "4ch": args.seg_worker_4ch,
        "sa": args.seg_worker_sa,
    }
    
    # Create worker
    worker = CCCineClassWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_names=args.model_names,
        model_config=model_config,
        device=args.device,
        gpu=args.gpu,
        seg_workers=seg_workers,
    )
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

