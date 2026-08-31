"""Optional single-exam wall-motion evidence client.

The remote worker hosts the frozen ProMax ConvNeXt head.  This adapter creates
the segmentation-derived SAX ROI used by that model and fails open so the main
segmentation/measurement workflow is never blocked by an unavailable research
head.
"""

from __future__ import annotations

import io
import os
from typing import Dict, Optional

import numpy as np
import requests
import SimpleITK as sitk
from PIL import Image

from app.utils.dicom import load_scans


WALL_MOTION_URL = os.getenv("CARDIAC_WALL_MOTION_URL", "").rstrip("/")


def _frames(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 2:
        return array[None]
    if array.ndim < 2:
        raise ValueError(f"invalid image dimensionality: {array.shape}")
    return array.reshape(-1, array.shape[-2], array.shape[-1])


def _resize(array: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.stack([
        np.asarray(Image.fromarray(frame.astype(np.float32)).resize((size, size), mode))
        for frame in array
    ])


def prepare_wall_motion_tensor(image_path: str, mask_path: str) -> np.ndarray:
    image, _ = load_scans(image_path)
    image_array = _frames(sitk.GetArrayFromImage(image).astype(np.float32))
    mask_array = _frames(
        sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)
    )
    if image_array.shape != mask_array.shape:
        raise ValueError(
            f"SAX image/mask shape mismatch: {image_array.shape} vs {mask_array.shape}"
        )
    indices = np.linspace(0, len(image_array) - 1, min(8, len(image_array))).round().astype(int)
    selected = image_array[indices]
    selected_mask = mask_array[indices]
    if len(selected) < 8:
        repeat = 8 - len(selected)
        selected = np.concatenate([selected, np.repeat(selected[-1:], repeat, axis=0)])
        selected_mask = np.concatenate([
            selected_mask, np.repeat(selected_mask[-1:], repeat, axis=0)
        ])

    finite = selected[np.isfinite(selected)]
    nonzero = finite[np.abs(finite) > 0]
    reference = nonzero if nonzero.size >= 64 else finite
    if reference.size < 64:
        raise ValueError("insufficient finite SAX pixels")
    low, high = np.percentile(reference, (1, 99))
    if not np.isfinite(high) or high <= low:
        raise ValueError("degenerate SAX intensity range")
    selected = np.nan_to_num(
        np.clip(selected, low, high), nan=low, posinf=high, neginf=low
    )
    selected = (selected - low) / (high - low)
    image256 = _resize(selected, 256)
    mask256 = _resize(selected_mask, 256, nearest=True)

    points = np.argwhere(np.any(mask256 != 0, axis=0))
    if len(points) < 32:
        crop = round(256 * 0.82)
        y0 = x0 = (256 - crop) // 2
        y1 = x1 = y0 + crop
        roi_fallback = True
    else:
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        height, width = y1 - y0, x1 - x0
        ypad, xpad = round(height * 0.30), round(width * 0.30)
        y0, x0 = max(0, y0 - ypad), max(0, x0 - xpad)
        y1, x1 = min(256, y1 + ypad), min(256, x1 + xpad)
        roi_fallback = False
    cropped = image256[:, y0:y1, x0:x1]
    roi128 = _resize(cropped, 128)
    tensor = _resize(roi128, 224).astype(np.float32)[:, None]
    # Attach preprocessing QC without changing the model tensor contract.
    prepare_wall_motion_tensor.last_roi_fallback = roi_fallback
    return tensor


prepare_wall_motion_tensor.last_roi_fallback = None


def score_wall_motion(image_path: str, mask_path: str,
                      timeout: float = 90.0) -> Optional[Dict]:
    if not WALL_MOTION_URL or not image_path or not mask_path:
        return None
    try:
        tensor = prepare_wall_motion_tensor(image_path, mask_path)
        payload = io.BytesIO()
        np.save(payload, tensor, allow_pickle=False)
        # This is a private-LAN model endpoint.  The portal process may inherit
        # HTTP(S)_PROXY for public API traffic; routing the binary tensor through
        # that proxy produces a 502 before it reaches the worker.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"{WALL_MOTION_URL}/infer",
                data=payload.getvalue(),
                headers={"Content-Type": "application/octet-stream"},
                timeout=timeout,
            )
        response.raise_for_status()
        result = response.json()
        result["roi_fallback"] = bool(prepare_wall_motion_tensor.last_roi_fallback)
        return result
    except Exception as exc:
        print(f"[wall_motion] optional research head unavailable: {exc}")
        return None
