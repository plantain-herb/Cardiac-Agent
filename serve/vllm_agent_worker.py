"""vLLM-backed worker compatible with the existing Cardiac-Agent client.

The frozen LLaVA checkpoint is split into a CLIP/projector/token-embedding
bridge and a standard Mistral decoder.  This keeps the existing multimodal
prompt contract while letting vLLM serve the decoder.
"""

import argparse
import asyncio
import base64
import io
import json
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from PIL import Image

from bridge.vision_bridge import VisionBridge
from serve.vllm_utils import normalize_image_tokens


class VLLMAgentWorker:
    def __init__(self, args):
        from vllm import LLM

        self.args = args
        self.model_names = [args.model_name]
        self.active = 0
        self.lock = threading.Lock()
        self.bridge = VisionBridge(args.bridge)
        self.llm = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype="float16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            enable_prompt_embeds=True,
            trust_remote_code=False,
        )
        self.register()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    def status(self):
        return {
            "model_names": self.model_names,
            "speed": 2,
            "queue_length": self.active,
            "backend": "vllm",
        }

    def register(self):
        response = requests.post(
            self.args.controller_address + "/register_worker",
            json={
                "worker_name": self.args.worker_address,
                "check_heart_beat": True,
                "worker_status": self.status(),
            },
            timeout=10,
        )
        response.raise_for_status()

    def _heartbeat_loop(self):
        while True:
            time.sleep(30)
            try:
                response = requests.post(
                    self.args.controller_address + "/receive_heart_beat",
                    json={
                        "worker_name": self.args.worker_address,
                        "queue_length": self.active,
                    },
                    timeout=10,
                )
                if not response.json().get("exist", False):
                    self.register()
            except Exception as exc:
                print(f"vLLM Agent heartbeat failed: {exc}", flush=True)

    @staticmethod
    def _decode_images(encoded_images):
        images = []
        for encoded in encoded_images or []:
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            images.append(
                Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            )
        return images

    @staticmethod
    def _normalize_image_tokens(prompt, image_count):
        return normalize_image_tokens(prompt, image_count)

    def generate(self, params):
        from vllm import SamplingParams

        original_prompt = params["prompt"]
        images = self._decode_images(params.get("images"))
        prompt = self._normalize_image_tokens(original_prompt, len(images))
        if images:
            prompt_embeds, _ = self.bridge.prepare(prompt, images)
        else:
            token_ids = self.bridge.tokenize(prompt)
            prompt_embeds = self.bridge.embed_tokens(token_ids)

        stop = params.get("stop")
        sampling = SamplingParams(
            temperature=float(params.get("temperature", 0.2)),
            top_p=float(params.get("top_p", 1.0)),
            max_tokens=min(int(params.get("max_new_tokens", 1024)), 1024),
            stop=[stop] if stop else None,
        )

        self.active += 1
        try:
            with self.lock:
                outputs = self.llm.generate(
                    {"prompt_embeds": prompt_embeds.detach().cpu()},
                    sampling,
                    use_tqdm=False,
                )
            text = outputs[0].outputs[0].text
            return (
                json.dumps(
                    {"text": original_prompt + text, "error_code": 0},
                    ensure_ascii=False,
                ).encode()
                + b"\0"
            )
        finally:
            self.active -= 1


app = FastAPI()
worker = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": worker.model_names[0],
        "backend": "vllm",
    }


@app.post("/worker_get_status")
async def worker_get_status(_request: Request):
    return worker.status()


@app.post("/worker_generate_stream")
async def worker_generate_stream(request: Request):
    params = await request.json()
    try:
        payload = await asyncio.to_thread(worker.generate, params)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        payload = (
            json.dumps(
                {"text": f"vLLM worker error: {exc}", "error_code": 1},
                ensure_ascii=False,
            ).encode()
            + b"\0"
        )
    return Response(payload, media_type="application/octet-stream")


def main():
    global worker
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=40000)
    parser.add_argument("--worker-address", default="http://localhost:40000")
    parser.add_argument("--controller-address", default="http://localhost:30000")
    parser.add_argument("--model-name", default="agent")
    parser.add_argument("--model", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    args = parser.parse_args()
    worker = VLLMAgentWorker(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
