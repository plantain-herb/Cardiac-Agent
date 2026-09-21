"""Geometry transforms required by legacy cardiac segmentation models."""

import numpy as np


def prepare_cine_4ch_for_model(volume: np.ndarray) -> np.ndarray:
    """Return the orientation used to train the legacy cine-4CH model."""
    value = np.asarray(volume)
    if value.ndim < 3:
        raise ValueError(f"cine 4CH volume must be at least 3D, got {value.shape}")
    return np.flip(value, axis=-2).copy()


def restore_cine_4ch_mask_to_source(mask: np.ndarray) -> np.ndarray:
    """Map a legacy cine-4CH prediction back onto the source image grid."""
    value = np.asarray(mask)
    if value.ndim < 3:
        raise ValueError(f"cine 4CH mask must be at least 3D, got {value.shape}")
    return np.flip(value, axis=-2).copy()
