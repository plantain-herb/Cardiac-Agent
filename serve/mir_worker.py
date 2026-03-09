"""
MIR (Medical Info Retrieval) Worker
A FastAPI service that executes RAG-based medical information retrieval.

Usage:
    python -m serve.mir_worker --port 21040 --no-register
"""
import sys
import os

# Get the directory paths
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RAG_SRC_DIR = os.path.join(SRC_DIR, "RAG", "ChatCAD")

# Add paths for imports
sys.path.insert(0, RAG_SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from util_minimal import *

import argparse
import asyncio
import time
import threading
import uuid
import traceback
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import uvicorn
import openai

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
logger = build_logger("mir_worker", os.path.join("workers", "mir.log"))
global_counter = 0
model_semaphore = None

# If building your own server, you can initialize chatbot in advance to accelerate inference speed on server
# chatbot = initialize_chatbot("")  # put your api key
chatbot = None


def heart_beat_worker(controller):
    """Send heartbeat to controller periodically."""
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    """
    Worker class for RAG Search.
    Handles RAG model execution and communication with controller.
    """
    
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        no_register: bool,
        model_names: list,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        
        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.daemon = True
            self.heart_beat_thread.start()
    
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
    
    def generate_stream_func(self, params):
        """
        Generate report using RAG.
        支持多种 LLM: OpenAI, DeepSeek, Qwen 等
        """
        prompt = params.get("prompt", "")
        api_key = params.get("api_key") or params.get("openai_key")  # 兼容旧参数名
        engine = params.get("engine", "gpt-4o")
        base_url = params.get("base_url")
        
        report = RAG(
            prompt, 
            api_key=api_key, 
            chatbot=chatbot,
            engine=engine,
            base_url=base_url
        )
        return report
    
    def generate_gate(self, params):
        """Entry point for RAG generation requests."""
        try:
            text = self.generate_stream_func(params)
            ret = {"text": text, "error_code": 0}
        except Exception as e:
            logger.error(f"RAG generation error: {traceback.format_exc()}")
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


# FastAPI Application
app = FastAPI(title="MIR (Medical Info Retrieval) Service")


def release_model_semaphore():
    """Release model semaphore."""
    model_semaphore.release()


def acquire_model_semaphore():
    """Acquire model semaphore."""
    global model_semaphore, global_counter
    global_counter += 1
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    return model_semaphore.acquire()


@app.post("/worker_generate")
async def api_generate(request: Request):
    """
    Main API endpoint for RAG generation.
    支持多种 LLM: OpenAI, DeepSeek, Qwen 等
    
    Request body:
        - prompt: str, the input prompt/question
        - api_key: str, API key (支持 OpenAI, DeepSeek, Qwen 等)
        - engine: str, 模型名称 (默认 "gpt-4o")
        - base_url: str, API base URL (可选)
        - openai_key: str, (已弃用，使用 api_key)
    
    Returns:
        - text: str, generated report/response
        - error_code: int, 0 for success
        
    示例:
        # OpenAI
        {"prompt": "...", "api_key": "sk-...", "engine": "gpt-4o"}
        
        # DeepSeek
        {"prompt": "...", "api_key": "sk-...", "engine": "deepseek-chat"}
        
        # Qwen
        {"prompt": "...", "api_key": "sk-...", "engine": "qwen-plus"}
    """
    params = await request.json()
    
    # 兼容旧参数名 openai_key
    api_key = params.get("api_key") or params.get("openai_key", "")
    if api_key:
        openai.api_key = api_key
    
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
        "task": "MIR (Medical Info Retrieval)",
        "description": "Medical information retrieval using RAG",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIR (Medical Info Retrieval) Worker")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21021)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21021")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-names",
        default="MedicalInformationRetrieval",
        type=lambda s: s.split(","),
        help="Model names (comma separated)",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true",
                        help="Don't register to controller")
    
    args = parser.parse_args()
    logger.info(f"args: {args}")
    
    # Create worker
    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_names=args.model_names,
    )
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
