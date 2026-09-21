import numpy as np

from app.utils.segmentation_geometry import (
    prepare_cine_4ch_for_model,
    restore_cine_4ch_mask_to_source,
)


def test_cine_4ch_orientation_round_trip():
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    prepared = prepare_cine_4ch_for_model(volume)

    assert np.array_equal(prepared, volume[:, ::-1, :])
    assert np.array_equal(restore_cine_4ch_mask_to_source(prepared), volume)


def test_cine_4ch_transform_returns_contiguous_arrays():
    volume = np.zeros((2, 3, 4), dtype=np.float32)
    assert prepare_cine_4ch_for_model(volume).flags.c_contiguous
