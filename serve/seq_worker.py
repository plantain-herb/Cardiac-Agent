"""
Sequence Analysis Worker
A FastAPI service that uses Agent to analyze images, then uses LLM to normalize the sequence type.

Usage:
    python -m serve.seq_worker --port 21050 --no-register
"""
import sys
import os

# Get the directory paths
SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVE_DIR)

# Add paths for imports
sys.path.insert(0, PROJECT_ROOT)

import argparse
import asyncio
import time
import threading
import uuid
import traceback
import json
import re
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import uvicorn
import openai

# Import LLaVA Agent components
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_IMAGE_TOKEN

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
logger = build_logger("seq_worker", os.path.join("workers", "seq.log"))
global_counter = 0
model_semaphore = None

# Agent URL
AGENT_URL = "http://localhost:40000"

# Valid sequence types
VALID_SEQUENCES = ["cine 2ch", "cine 4ch", "cine sa", "lge sa"]


def heart_beat_worker(controller):
    """Send heartbeat to controller periodically."""
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def call_agent(prompt: str, images: List[str], agent_url: str = AGENT_URL) -> str:
    """
    Call LLaVA Agent to analyze images.
    
    Returns:
        Agent's response text (extracted from conversation)
    """
    # Build conversation with proper template
    conv = conv_templates["v1"].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    
    # 根据图片数量添加正确数量的 <image> token（与脚本推理保持一致）
    num_images = len(images) if images else 0
    logger.info(f"[DEBUG call_agent] num_images={num_images}, DEFAULT_IMAGE_TOKEN='{DEFAULT_IMAGE_TOKEN}'")
    if num_images > 0:
        # 移除 prompt 中已有的 <image> token
        prompt_clean = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        # 根据图片数量添加 <image> token
        image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
        prompt_with_images = image_tokens + prompt_clean
        logger.info(f"[DEBUG call_agent] Added {num_images} image tokens, prompt_with_images (first 100 chars): {prompt_with_images[:100]}")
    else:
        prompt_with_images = prompt
    
    # Add user message
    conv.append_message(conv.roles[0], prompt_with_images)
    conv.append_message(conv.roles[1], None)
    formatted_prompt = conv.get_prompt()
    
    # Build request（移除 top_p，与脚本推理保持一致）
    pload = {
        "model": "agent",
        "prompt": formatted_prompt,
        "temperature": 0.2,
        "max_new_tokens": 1024,
        "stop": stop_str,
        "images": images,
    }
    
    try:
        # Send request
        resp = requests.post(
            f"{agent_url}/worker_generate_stream",
            json=pload,
            stream=True,
            timeout=60
        )
        
        # Collect response
        full_response = ""
        for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data.get("error_code", 0) == 0:
                    full_response = data.get("text", "")
        
        # Extract only the ASSISTANT's response (remove the conversation template)
        # The full_response contains: "...USER: <prompt> ASSISTANT: <actual_response>"
        # We need to extract only the <actual_response> part
        if "ASSISTANT:" in full_response:
            # Extract text after "ASSISTANT:"
            assistant_response = full_response.split("ASSISTANT:")[-1].strip()
            # Remove any trailing stop sequences
            if stop_str and assistant_response.endswith(stop_str):
                assistant_response = assistant_response[:-len(stop_str)].strip()
            return assistant_response
        else:
            # Fallback: return full response
            return full_response
        
    except Exception as e:
        logger.error(f"Agent call failed: {e}")
        return str(e)


def normalize_sequence_with_llm(agent_response: str, api_key: str, 
                                engine: str = "gpt-4o", 
                                base_url: str = None) -> List[str]:
    """
    Use LLM to normalize Agent's response to one of the valid sequence types.
    
    Args:
        agent_response: Agent's analysis text
        api_key: API key for LLM
        engine: LLM model name
        base_url: API base URL
    
    Returns:
        List of detected sequences (normalized to valid types)
    """
    # Build prompt for LLM to classify the sequence
    classification_prompt = f"""Based on the following cardiac MRI image analysis, identify which sequence type it is.

Agent's Analysis:
{agent_response}

Valid Sequence Types:
1. cine 2ch - Two-chamber cine sequence
2. cine 4ch - Four-chamber cine sequence  
3. cine sa - Short-axis cine sequence
4. lge sa - Short-axis late gadolinium enhancement sequence

Instructions:
- Analyze the description and determine which sequence type matches best
- Return ONLY the sequence type from the valid options above
- Return in this exact format: ["sequence_type"]
- If multiple types are detected, return them as a list: ["type1", "type2"]
- If uncertain, return the most likely option

Your response (JSON format only):"""

    # Set up OpenAI client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = openai.OpenAI(**client_kwargs)
    
    # Call LLM
    response = client.chat.completions.create(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a medical imaging expert. Classify cardiac MRI sequences accurately. Always return valid JSON array format."},
            {"role": "user", "content": classification_prompt}
        ],
        temperature=0.1,  # Low temperature for consistent classification
        max_tokens=100,
    )
    
    llm_output = response.choices[0].message.content.strip()
    
    # Parse LLM output to extract sequence types
    try:
        # Try to parse as JSON
        if llm_output.startswith('[') and llm_output.endswith(']'):
            detected = json.loads(llm_output)
        else:
            # Extract JSON array from text
            match = re.search(r'\[.*?\]', llm_output)
            if match:
                detected = json.loads(match.group())
            else:
                # Fallback: check for valid sequence names in text
                detected = []
                for seq in VALID_SEQUENCES:
                    if seq in llm_output.lower():
                        detected.append(seq)
        
        # Validate and filter
        valid_detected = [seq for seq in detected if seq.lower() in [v.lower() for v in VALID_SEQUENCES]]
        
        # Normalize to exact format
        normalized = []
        for seq in valid_detected:
            for valid_seq in VALID_SEQUENCES:
                if seq.lower() == valid_seq.lower():
                    normalized.append(valid_seq)
                    break
        
        return normalized if normalized else ["cine sa"]  # Default fallback
        
    except Exception as e:
        logger.error(f"Failed to parse LLM output: {e}, output: {llm_output}")
        # Fallback: simple text matching
        for seq in VALID_SEQUENCES:
            if seq in llm_output.lower():
                return [seq]
        return ["cine sa"]  # Default fallback


class ModelWorker:
    """
    Worker class for Sequence Analysis.
    First calls Agent to analyze images, then uses LLM to normalize sequence type.
    """
    
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        no_register: bool,
        model_names: list,
        agent_url: str,
    ):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_names = model_names
        self.agent_url = agent_url

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")
        logger.info(f"Agent URL: {self.agent_url}")
        
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
        Generate sequence classification using Agent + LLM pipeline.
        
        Pipeline:
        1. Call Agent to analyze images
        2. Use LLM to normalize Agent's output to one of: cine 2ch, cine 4ch, cine sa, lge sa
        3. Return only the detected_sequences
        """
        prompt = params.get("prompt", "Which sequence does this cardiac MRI belong to?")
        images = params.get("images", [])  # Base64 encoded images
        api_key = params.get("api_key") or params.get("openai_key")
        engine = params.get("engine", "gpt-4o")
        base_url = params.get("base_url")
        
        logger.info(f"[Step 1] Calling Agent to analyze {len(images)} images...")
        
        # Step 1: Call Agent to analyze images
        agent_response = call_agent(prompt, images, self.agent_url)
        
        # Log the full response for debugging
        logger.info(f"[Step 1] Agent full response: {agent_response}")
        logger.info(f"[Step 1] Agent response (preview): {agent_response[:200]}...")
        
        # Step 2: Use LLM to normalize sequence type
        logger.info(f"[Step 2] Using {engine} to normalize sequence type...")
        detected_sequences = normalize_sequence_with_llm(
            agent_response, 
            api_key, 
            engine, 
            base_url
        )
        
        logger.info(f"[Step 2] Detected sequences: {detected_sequences}")
        
        # Return result
        result = {
            "agent_response": agent_response,
            "detected_sequences": detected_sequences,
        }
        
        return result
    
    def generate_gate(self, params):
        """Entry point for sequence analysis requests."""
        try:
            result = self.generate_stream_func(params)
            ret = {
                "agent_response": result["agent_response"],
                "detected_sequences": result["detected_sequences"],
                "error_code": 0
            }
        except Exception as e:
            logger.error(f"Sequence analysis error: {traceback.format_exc()}")
            ret = {
                "detected_sequences": [],
                "error": str(e),
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
        return ret


# FastAPI Application
app = FastAPI(title="Sequence Analysis Service")


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
    Main API endpoint for sequence analysis.
    
    Request body:
        - prompt: str, the input prompt/question (optional)
        - images: list, base64 encoded images
        - api_key: str, API key (支持 OpenAI, DeepSeek, Qwen 等)
        - engine: str, 模型名称 (默认 "gpt-4o")
        - base_url: str, API base URL (可选)
    
    Returns:
        - agent_response: str, agent's raw analysis
        - detected_sequences: list, normalized sequence types (e.g., ["cine sa"])
        - error_code: int, 0 for success
    """
    params = await request.json()
    
    # Set API key
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
        "task": "Sequence Analysis (Agent + LLM Normalization)",
        "description": "Cardiac MRI sequence classification using Agent and LLM normalization",
        "valid_sequences": VALID_SEQUENCES,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence Analysis Worker")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21050)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21050")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:20001"
    )
    parser.add_argument("--agent-url", type=str, default="http://localhost:40000",
                        help="URL of LLaVA Agent service")
    
    # Model configuration
    parser.add_argument(
        "--model-names",
        default="SequenceAnalysis",
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
        agent_url=args.agent_url,
    )
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
