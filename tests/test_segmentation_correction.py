import os
import tempfile
import unittest

import nibabel as nib
import numpy as np

from app.services.segmentation_correction import (
    SegmentationCorrectionError,
    validate_corrected_segmentation,
)


class SegmentationCorrectionValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.affine = np.diag([1.25, 1.25, 2.5, 1.0])
        self.baseline = self._save("baseline.nii.gz", np.zeros((8, 9, 10)))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save(self, name, data, affine=None):
        path = os.path.join(self.temp_dir.name, name)
        nib.save(
            nib.Nifti1Image(data.astype(np.float32), affine if affine is not None else self.affine),
            path,
        )
        return path

    def test_accepts_matching_integer_labels(self):
        data = np.zeros((8, 9, 10))
        data[1:3] = 1
        data[3:5] = 2
        data[5:7] = 4
        corrected = self._save("corrected_sa.nii.gz", data)

        metadata = validate_corrected_segmentation(corrected, self.baseline, "sa")

        self.assertEqual(metadata["shape"], [8, 9, 10])
        self.assertEqual(metadata["unique_labels"], [0, 1, 2, 4])

    def test_rejects_labels_outside_modality_contract(self):
        data = np.zeros((8, 9, 10))
        data[0, 0, 0] = 5
        corrected = self._save("invalid_sa.nii.gz", data)

        with self.assertRaisesRegex(SegmentationCorrectionError, "unsupported labels"):
            validate_corrected_segmentation(corrected, self.baseline, "sa")

    def test_rejects_shape_change(self):
        corrected = self._save("wrong_shape.nii.gz", np.zeros((8, 9, 11)))

        with self.assertRaisesRegex(SegmentationCorrectionError, "does not match"):
            validate_corrected_segmentation(corrected, self.baseline, "4ch")

    def test_rejects_affine_change(self):
        shifted_affine = self.affine.copy()
        shifted_affine[0, 3] = 10
        corrected = self._save(
            "wrong_affine.nii.gz", np.zeros((8, 9, 10)), affine=shifted_affine
        )

        with self.assertRaisesRegex(SegmentationCorrectionError, "affine"):
            validate_corrected_segmentation(corrected, self.baseline, "4ch")

    def test_rejects_fractional_labels(self):
        data = np.zeros((8, 9, 10))
        data[0, 0, 0] = 1.5
        corrected = self._save("fractional.nii.gz", data)

        with self.assertRaisesRegex(SegmentationCorrectionError, "integer label"):
            validate_corrected_segmentation(corrected, self.baseline, "4ch")

    def test_rejects_missing_labels_required_for_metrics(self):
        data = np.zeros((8, 9, 10))
        data[1:3] = 1
        corrected = self._save("missing_required.nii.gz", data)

        with self.assertRaisesRegex(SegmentationCorrectionError, "missing labels required"):
            validate_corrected_segmentation(corrected, self.baseline, "sa")


if __name__ == "__main__":
    unittest.main()
