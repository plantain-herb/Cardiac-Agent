"""Geometry validation helpers for cardiac segmentation inputs."""

import numpy as np


def prepare_cine_4ch_for_model(volume: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous 4CH volume without reorientation."""
    value = np.asarray(volume)
    if value.ndim < 3:
        raise ValueError(f"cine 4CH volume must be at least 3D, got {value.shape}")
    return np.ascontiguousarray(value)


def restore_cine_4ch_mask_to_source(mask: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous mask already on the source grid."""
    value = np.asarray(mask)
    if value.ndim < 3:
        raise ValueError(f"cine 4CH mask must be at least 3D, got {value.shape}")
    return np.ascontiguousarray(value)
