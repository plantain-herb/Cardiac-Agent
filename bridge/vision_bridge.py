import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
)


def expand_square(image, color):
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    result = Image.new(image.mode, (size, size), color)
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


class VisionBridge(nn.Module):
    """Rebuild LLaVA prompt embeddings for a standard vLLM Mistral decoder."""

    def __init__(self, path, device="cuda", dtype=torch.float16):
        super().__init__()
        self.path = Path(path)
        self.bridge_config = json.loads(
            (self.path / "bridge_config.json").read_text()
        )
        clip_raw = json.loads((self.path / "clip_config.json").read_text())
        clip_config = CLIPVisionConfig(**clip_raw["vision_config"])
        self.vision = CLIPVisionModel(clip_config)
        self.projector = nn.Sequential(
            nn.Linear(
                self.bridge_config["mm_hidden_size"],
                self.bridge_config["hidden_size"],
            ),
            nn.GELU(),
            nn.Linear(
                self.bridge_config["hidden_size"],
                self.bridge_config["hidden_size"],
            ),
        )
        self.embed_tokens = nn.Embedding(
            self.bridge_config["vocab_size"],
            self.bridge_config["hidden_size"],
        )

        state = load_file(self.path / "bridge.safetensors")
        vision_state = {
            key.removeprefix("vision."): value
            for key, value in state.items()
            if key.startswith("vision.")
        }
        target_keys = self.vision.state_dict().keys()
        if target_keys and not next(iter(target_keys)).startswith("vision_model."):
            vision_state = {
                key.removeprefix("vision_model."): value
                for key, value in vision_state.items()
            }
        self.vision.load_state_dict(vision_state)
        self.projector.load_state_dict(
            {
                key.removeprefix("projector."): value
                for key, value in state.items()
                if key.startswith("projector.")
            }
        )
        self.embed_tokens.load_state_dict({"weight": state["embed_tokens.weight"]})
        self.processor = CLIPImageProcessor.from_pretrained(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, use_fast=False)
        self.to(device=device, dtype=dtype).eval()

    def tokenize(self, prompt):
        image_token = self.bridge_config["image_token_index"]
        chunks = [
            self.tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
        ]
        offset, token_ids = 0, []
        if chunks and chunks[0] and chunks[0][0] == self.tokenizer.bos_token_id:
            offset, token_ids = 1, [chunks[0][0]]
        for position, chunk in enumerate(chunks):
            token_ids.extend(chunk[offset:])
            if position < len(chunks) - 1:
                token_ids.append(image_token)
        return torch.tensor(
            token_ids,
            dtype=torch.long,
            device=self.embed_tokens.weight.device,
        )

    @torch.inference_mode()
    def encode_images(self, images):
        color = tuple(int(value * 255) for value in self.processor.image_mean)
        pixels = [
            self.processor.preprocess(
                expand_square(image.convert("RGB"), color), return_tensors="pt"
            )["pixel_values"][0]
            for image in images
        ]
        pixels = torch.stack(pixels).to(
            self.vision.device, dtype=self.vision.dtype
        )
        output = self.vision(pixels, output_hidden_states=True)
        features = output.hidden_states[
            self.bridge_config["mm_vision_select_layer"]
        ]
        if self.bridge_config["mm_vision_select_feature"] == "patch":
            features = features[:, 1:]
        return self.projector(features)

    @torch.inference_mode()
    def prepare(self, prompt, images):
        token_ids = self.tokenize(prompt)
        features = self.encode_images(images)
        positions = torch.where(
            token_ids == self.bridge_config["image_token_index"]
        )[0].tolist()
        if len(positions) != len(images):
            raise ValueError(
                f"prompt has {len(positions)} image tokens but received "
                f"{len(images)} images"
            )
        pieces, start = [], 0
        for image_index, position in enumerate(positions):
            pieces.append(self.embed_tokens(token_ids[start:position]))
            pieces.append(features[image_index])
            start = position + 1
        pieces.append(self.embed_tokens(token_ids[start:]))
        return torch.cat(pieces, dim=0), token_ids
