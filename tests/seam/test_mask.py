from __future__ import annotations

import unittest

import numpy as np

from deep_shark_studio.seam.dp import SeamPath
from deep_shark_studio.seam.mask import SeamMaskBuilder


def vertical_path(x: int, height: int) -> SeamPath:
    points = [(x, y) for y in range(height)]
    return SeamPath(
        points_roi=points,
        points_canvas=points,
        mean_cost=0.0,
        p95_cost=0.0,
        jaggedness=0.0,
    )


class SeamMaskBuilderTests(unittest.TestCase):
    def test_hard_seam_masks_left_and_right(self) -> None:
        result = SeamMaskBuilder().build(
            vertical_path(5, 10),
            (10, 12),
            (2, 9),
            feather_width_px=0,
        )

        self.assertTrue(np.all(result.hard_mask_a[:, :5]))
        self.assertTrue(np.all(result.hard_mask_b[:, 5:]))
        self.assertTrue(np.all(result.alpha_a[:, :5] == 1.0))
        self.assertTrue(np.all(result.alpha_b[:, 5:] == 1.0))

    def test_feather_alpha_changes_smoothly_near_seam(self) -> None:
        result = SeamMaskBuilder().build(
            vertical_path(5, 10),
            (10, 12),
            (2, 9),
            feather_width_px=4,
        )

        row = result.alpha_b[0]
        self.assertAlmostEqual(0.0, float(row[3]), places=6)
        self.assertAlmostEqual(0.5, float(row[5]), places=6)
        self.assertAlmostEqual(1.0, float(row[7]), places=6)
        self.assertTrue(np.all(np.diff(row[2:9]) >= -1e-6))


if __name__ == "__main__":
    unittest.main()
