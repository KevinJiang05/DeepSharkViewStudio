from __future__ import annotations

from copy import deepcopy
import time
import unittest

import numpy as np

from deep_shark_studio.config import load_config
from deep_shark_studio.stitch_processing import StitchProcessingManager
from deep_shark_studio.stitcher import SurroundStitcher
from deep_shark_studio.topology import resolve_stitch_config, seam_x


def solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((540, 960, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


def triple_reference_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((1080, 1920, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


class TopologyConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("calibration.yaml")

    def test_dual_profile_geometry_is_horizontal_and_isolated(self) -> None:
        profile = self.config["topologies"]["dual_horizontal"]
        self.assertEqual(["front_left", "front_right"], profile["active_cameras"])
        self.assertNotEqual(
            profile["cameras"]["front_left"]["target_points"],
            profile["cameras"]["front_right"]["target_points"],
        )
        self.assertNotIn("mask_rules", profile)
        overlap = profile["overlaps"][0]
        position = seam_x(profile["stitch_points"], overlap["seam"])
        self.assertLess(overlap["x_range"][0], position)
        self.assertLess(position, overlap["x_range"][1])

    def test_triple_profile_declares_two_valid_overlaps(self) -> None:
        profile = self.config["topologies"]["triple_front_panorama"]
        self.assertEqual(
            ["front_left", "front", "front_right"],
            profile["active_cameras"],
        )
        targets = [
            profile["cameras"][key]["target_points"]
            for key in profile["active_cameras"]
        ]
        self.assertEqual(3, len({str(points) for points in targets}))
        self.assertEqual(2, len(profile["overlaps"]))
        for overlap in profile["overlaps"]:
            position = seam_x(profile["stitch_points"], overlap["seam"])
            self.assertLess(overlap["x_range"][0], position)
            self.assertLess(position, overlap["x_range"][1])
        self.assertNotIn("mask_rules", profile)

    def test_legacy_root_config_still_loads(self) -> None:
        legacy = deepcopy(self.config)
        legacy.pop("stitch_topology", None)
        legacy.pop("topologies", None)
        stitcher = SurroundStitcher(legacy, max_input_width=None, use_intrinsics=False)
        self.assertEqual("legacy_surround_5", stitcher.topology_name)
        self.assertEqual(5, len(stitcher.cameras))
        self.assertTrue(stitcher.mask_rules)


class TopologyImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("calibration.yaml")

    def test_dual_output_is_left_right_not_top_bottom(self) -> None:
        config = deepcopy(self.config)
        config["stitch_topology"] = "dual_horizontal"
        stitcher = SurroundStitcher(
            config,
            max_input_width=None,
            use_intrinsics=False,
        )
        _, canvas = stitcher.process(
            {
                "front_left": solid_frame((0, 0, 255)),
                "front_right": solid_frame((0, 255, 0)),
            }
        )
        self.assertEqual((600, 1600, 3), canvas.shape)
        for y in (60, 300, 540):
            self.assertGreater(int(canvas[y, 250, 2]), 220)
            self.assertGreater(int(canvas[y, 1350, 1]), 220)
        seam_pixel = canvas[300, 800]
        self.assertGreater(int(seam_pixel[1]), 80)
        self.assertGreater(int(seam_pixel[2]), 80)
        self.assertGreater(float(np.any(canvas != 0, axis=2).mean()), 0.95)

    def test_triple_output_is_continuous_left_front_right(self) -> None:
        config = deepcopy(self.config)
        config["stitch_topology"] = "triple_front_panorama"
        stitcher = SurroundStitcher(config, max_input_width=None, use_intrinsics=False)
        _, canvas = stitcher.process(
            {
                "front_left": triple_reference_frame((0, 0, 255)),
                "front": triple_reference_frame((0, 255, 0)),
                "front_right": triple_reference_frame((255, 0, 0)),
            }
        )
        self.assertEqual((700, 2200, 3), canvas.shape)
        self.assertGreater(int(canvas[350, 250, 2]), 220)
        self.assertGreater(int(canvas[350, 1100, 1]), 220)
        self.assertGreater(int(canvas[350, 1950, 0]), 220)
        self.assertGreater(float(np.any(canvas != 0, axis=2).mean()), 0.95)

    def test_live_worker_uses_selected_topology(self) -> None:
        manager = StitchProcessingManager()
        try:
            manager.configure(7, self.config, None, False)
            manager.submit_latest(
                7,
                {
                    "front_left": triple_reference_frame((0, 0, 255)),
                    "front": triple_reference_frame((0, 255, 0)),
                    "front_right": triple_reference_frame((255, 0, 0)),
                },
            )
            deadline = time.monotonic() + 3.0
            result = None
            while time.monotonic() < deadline:
                result = manager.take_latest_result()
                if result is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertEqual("", result.error)
            self.assertEqual((700, 2200, 3), result.canvas.shape)
        finally:
            self.assertTrue(manager.shutdown())


if __name__ == "__main__":
    unittest.main()
