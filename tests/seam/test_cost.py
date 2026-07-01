from __future__ import annotations

import unittest

import numpy as np

from deep_shark_studio.seam.cost import SeamCostBuilder, SeamCostParams


class SeamCostBuilderTests(unittest.TestCase):
    def test_cost_map_uses_overlap_roi_size(self) -> None:
        image_a = np.full((12, 20, 3), (20, 60, 120), dtype=np.uint8)
        image_b = np.full((12, 20, 3), (22, 62, 122), dtype=np.uint8)

        result = SeamCostBuilder().build(
            image_a,
            image_b,
            (4, 9),
            SeamCostParams(safe_erode_px=0),
        )

        self.assertEqual((12, 6), result.cost.shape)
        self.assertEqual((12, 6), result.valid_mask.shape)
        self.assertEqual((4, 9), result.x_range)

    def test_invalid_mask_produces_high_cost(self) -> None:
        image_a = np.full((8, 12, 3), 80, dtype=np.uint8)
        image_b = np.full((8, 12, 3), 90, dtype=np.uint8)
        image_b[:, 5] = 0
        params = SeamCostParams(safe_erode_px=0, invalid_cost=12345.0)

        result = SeamCostBuilder().build(image_a, image_b, (3, 7), params)

        invalid_column = 5 - result.x_range[0]
        self.assertFalse(np.any(result.valid_mask[:, invalid_column]))
        self.assertTrue(np.all(result.cost[:, invalid_column] == params.invalid_cost))


if __name__ == "__main__":
    unittest.main()
