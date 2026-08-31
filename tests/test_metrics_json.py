import unittest

import numpy as np

from serve.metrics_worker import _json_safe


class MetricsJsonTest(unittest.TestCase):
    def test_converts_numpy_values_and_arrays(self):
        result = _json_safe(
            {
                "scalar": np.float32(12.5),
                "count": np.int64(3),
                "array": np.asarray([1.0, 2.0], dtype=np.float32),
            }
        )

        self.assertEqual(result, {"scalar": 12.5, "count": 3, "array": [1.0, 2.0]})

    def test_converts_non_finite_values_to_null(self):
        result = _json_safe({"nan": np.float32(np.nan), "inf": float("inf")})

        self.assertEqual(result, {"nan": None, "inf": None})


if __name__ == "__main__":
    unittest.main()
