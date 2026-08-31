"""CPU HTTP worker for the frozen ProMax cine wall-motion ConvNeXt head."""

from __future__ import annotations

import argparse
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from torch import nn
from torchvision.models import convnext_tiny


MODEL_VERSION = "promax-cine-wall-motion-convnext-seed20260821"
VALIDATION_AUROC = 0.8182749326145553
THRESHOLD = 0.8


class ConvNeXtBagHead(nn.Module):
    def __init__(self, imagenet_checkpoint: Path, embedding: int = 256):
        super().__init__()
        model = convnext_tiny(weights=None)
        model.load_state_dict(torch.load(
            imagenet_checkpoint, map_location="cpu", weights_only=True
        ))
        self.features = model.features
        self.avgpool = model.avgpool
        self.norm = model.classifier[0]
        self.freeze_backbone = True
        self.unfreeze_tail_modules = 0
        self.project = nn.Sequential(
            nn.Linear(768, embedding), nn.LayerNorm(embedding), nn.GELU()
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(embedding, embedding, 3, padding=1, groups=embedding, bias=False),
            nn.Conv1d(embedding, embedding, 1, bias=False),
            nn.BatchNorm1d(embedding), nn.GELU(),
        )
        self.attention = nn.Conv1d(embedding, 1, 1)
        self.head = nn.Sequential(
            nn.Linear(embedding * 3, embedding), nn.GELU(),
            nn.Dropout(0.35), nn.Linear(embedding, 1),
        )
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, depth, channels, height, width = tensor.shape
        frames = tensor.reshape(batch * depth, channels, height, width)
        frames = (frames.repeat(1, 3, 1, 1) - self.mean) / self.std
        value = self.features(frames)
        value = self.norm(self.avgpool(value)).flatten(1)
        features = self.project(value).reshape(batch, depth, -1).transpose(1, 2)
        features = self.temporal(features)
        weights = self.attention(features).softmax(-1)
        pooled = torch.cat([
            features.mean(-1), features.amax(-1), (features * weights).sum(-1)
        ], dim=1)
        return self.head(pooled).flatten()


class Worker:
    def __init__(self, checkpoint: Path, imagenet_checkpoint: Path, threads: int):
        torch.set_num_threads(max(1, threads))
        self.model = ConvNeXtBagHead(imagenet_checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @torch.inference_mode()
    def infer(self, tensor: np.ndarray) -> dict:
        if tensor.shape != (8, 1, 224, 224) or tensor.dtype != np.float32:
            raise ValueError(f"expected float32 [8,1,224,224], got {tensor.dtype} {tensor.shape}")
        probability = float(torch.sigmoid(self.model(torch.from_numpy(tensor)[None]))[0])
        return {
            "status": "available",
            "finding": "wall_motion_abnormality",
            "probability": probability,
            "positive": probability >= THRESHOLD,
            "threshold": THRESHOLD,
            "model_version": MODEL_VERSION,
            "validation_auroc": VALIDATION_AUROC,
            "reliability": "research_model_evidence_not_calibrated_burden",
        }


def handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "healthy", "model_version": MODEL_VERSION})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/infer":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32 * 1024 * 1024:
                    raise ValueError("invalid payload size")
                tensor = np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False)
                self._json(200, worker.infer(tensor))
            except Exception as exc:
                self._json(422, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print(f"[wall-motion-worker] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--imagenet-checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21141)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    worker = Worker(args.checkpoint, args.imagenet_checkpoint, args.threads)
    print(json.dumps({"status": "ready", "port": args.port, "model": MODEL_VERSION}), flush=True)
    ThreadingHTTPServer((args.host, args.port), handler(worker)).serve_forever()


if __name__ == "__main__":
    main()
