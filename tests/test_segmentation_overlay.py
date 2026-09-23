import os
import tempfile

import numpy as np
import SimpleITK as sitk

from app.utils.dicom import save_segmentation_images


def _write_image(path, shape, *, origin=(0.0, 0.0, 0.0)):
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.uint8))
    image.SetOrigin(origin)
    sitk.WriteImage(image, path)


def test_overlay_rejects_different_image_and_mask_size():
    with tempfile.TemporaryDirectory() as directory:
        image_path = os.path.join(directory, "image.nii.gz")
        mask_path = os.path.join(directory, "mask.nii.gz")
        _write_image(image_path, (4, 8, 8))
        _write_image(mask_path, (4, 7, 8))

        result = save_segmentation_images(image_path, mask_path, "test-session")

    assert "different size" in result["error"]


def test_overlay_rejects_different_affine_origin():
    with tempfile.TemporaryDirectory() as directory:
        image_path = os.path.join(directory, "image.nii.gz")
        mask_path = os.path.join(directory, "mask.nii.gz")
        _write_image(image_path, (4, 8, 8))
        _write_image(mask_path, (4, 8, 8), origin=(5.0, 0.0, 0.0))

        result = save_segmentation_images(image_path, mask_path, "test-session")

    assert "different origin" in result["error"]
