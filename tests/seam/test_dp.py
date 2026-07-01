from __future__ import annotations

import unittest

import numpy as np

from deep_shark_studio.seam.dp import DPSeamParams, VerticalDPSeamFinder


class VerticalDPSeamFinderTests(unittest.TestCase):
    def test_finds_low_cost_channel(self) -> None:
        cost = np.full((24, 30), 10.0, dtype=np.float32)
        valid = np.ones_like(cost, dtype=bool)
        cost[:, 11] = 0.05

        path = VerticalDPSeamFinder().find(
            cost,
            valid,
            DPSeamParams(max_step_px=2, smoothness_weight=0.5, downsample=1),
            x_offset=100,
        )

        xs = [x for x, _ in path.points_roi]
        self.assertTrue(all(abs(x - 11) <= 1 for x in xs))
        self.assertEqual((111, 0), path.points_canvas[0])
        self.assertLess(path.mean_cost, 1.0)

    def test_path_respects_max_step(self) -> None:
        cost = np.full((20, 40), 5.0, dtype=np.float32)
        valid = np.ones_like(cost, dtype=bool)
        for y in range(cost.shape[0]):
            cost[y, min(30, 5 + y)] = 0.0

        path = VerticalDPSeamFinder().find(
            cost,
            valid,
            DPSeamParams(max_step_px=3, smoothness_weight=0.1, downsample=1),
        )

        xs = np.asarray([x for x, _ in path.points_roi])
        self.assertTrue(np.all(np.abs(np.diff(xs)) <= 3))

    def test_empty_valid_rows_fallback_without_crashing(self) -> None:
        cost = np.full((8, 10), 1.0, dtype=np.float32)
        valid = np.ones_like(cost, dtype=bool)
        valid[3] = False

        path = VerticalDPSeamFinder().find(
            cost,
            valid,
            DPSeamParams(max_step_px=2, downsample=1),
        )

        self.assertEqual(8, len(path.points_roi))


if __name__ == "__main__":
    unittest.main()
