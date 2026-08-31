"""人工修正分割的 NIfTI 契约与校验。"""

from typing import Dict, Iterable

import nibabel as nib
import numpy as np


ALLOWED_LABELS = {
    "4ch": frozenset(range(7)),
    "sa": frozenset(range(5)),
}

REQUIRED_LABELS = {
    "4ch": frozenset({1, 6}),  # LV blood pool and LA blood pool
    "sa": frozenset({1, 2}),   # LV myocardium and LV blood pool
}


class SegmentationCorrectionError(ValueError):
    """修正版 mask 不满足自动 mask 的几何或标签契约。"""


def _format_labels(labels: Iterable[int]) -> str:
    return ", ".join(str(label) for label in sorted(labels))


def validate_corrected_segmentation(
    corrected_path: str,
    baseline_path: str,
    modality: str,
) -> Dict:
    """校验修正版 mask 与本轮自动 mask 的几何、数值和标签一致性。"""
    if modality not in ALLOWED_LABELS:
        raise SegmentationCorrectionError(f"Unsupported modality: {modality}")

    try:
        corrected_img = nib.load(corrected_path)
        baseline_img = nib.load(baseline_path)
    except Exception as exc:
        raise SegmentationCorrectionError(f"Cannot read NIfTI file: {exc}") from exc

    if len(corrected_img.shape) != 3:
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} mask must be 3D, got shape {corrected_img.shape}."
        )
    if corrected_img.shape != baseline_img.shape:
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} shape {corrected_img.shape} does not match "
            f"the automatic mask shape {baseline_img.shape}."
        )

    corrected_zooms = np.asarray(corrected_img.header.get_zooms()[:3], dtype=float)
    baseline_zooms = np.asarray(baseline_img.header.get_zooms()[:3], dtype=float)
    if not np.allclose(corrected_zooms, baseline_zooms, rtol=1e-5, atol=1e-5):
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} voxel spacing {tuple(corrected_zooms)} does not "
            f"match {tuple(baseline_zooms)}. Preserve the original NIfTI geometry."
        )
    if not np.allclose(corrected_img.affine, baseline_img.affine, rtol=1e-5, atol=1e-4):
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} affine does not match the automatic mask. "
            "Preserve the original NIfTI orientation and origin."
        )

    try:
        data = np.asarray(corrected_img.dataobj)
    except Exception as exc:
        raise SegmentationCorrectionError(f"Cannot read NIfTI voxel data: {exc}") from exc
    if not np.all(np.isfinite(data)):
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} mask contains NaN or infinite values."
        )

    rounded = np.rint(data)
    if not np.allclose(data, rounded, rtol=0, atol=1e-4):
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} mask must contain integer label values."
        )

    labels = {int(value) for value in np.unique(rounded)}
    invalid_labels = labels - ALLOWED_LABELS[modality]
    if invalid_labels:
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} mask contains unsupported labels "
            f"[{_format_labels(invalid_labels)}]. Allowed labels: "
            f"[{_format_labels(ALLOWED_LABELS[modality])}]."
        )
    missing_labels = REQUIRED_LABELS[modality] - labels
    if missing_labels:
        raise SegmentationCorrectionError(
            f"Corrected {modality.upper()} mask is missing labels required for metric "
            f"calculation: [{_format_labels(missing_labels)}]."
        )

    return {
        "shape": list(corrected_img.shape),
        "voxel_spacing": corrected_zooms.tolist(),
        "unique_labels": sorted(labels),
    }
