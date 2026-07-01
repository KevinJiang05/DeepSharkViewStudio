from __future__ import annotations

from copy import deepcopy
import unittest

import cv2
import numpy as np

from deep_shark_studio.config import load_config
from deep_shark_studio.geometry_diagnostics import (
    ALPHA_50,
    NO_FEATHER,
    compose_pair_only,
    run_geometry_diagnostics,
)
from deep_shark_studio.stitcher import SurroundStitcher
from deep_shark_studio.topology import (
    active_stitch_profile,
    seam_x,
    source_coordinate_diagnostics,
    validate_overlap_seams,
)


def solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((1080, 1920, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


class GeometryDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = deepcopy(load_config("calibration.yaml"))
        profile = active_stitch_profile(self.config)
        height = float(profile["canvas"]["height"])
        for overlap in profile["overlaps"]:
            center_x = sum(overlap["x_range"]) / 2.0
            profile["stitch_points"][overlap["seam"]] = [
                [center_x, 0.0],
                [center_x, height],
            ]
        self.frames = {
            "front_left": solid_frame((0, 0, 255)),
            "front": solid_frame((0, 255, 0)),
            "front_right": solid_frame((255, 0, 0)),
        }

    def test_diagnostics_do_not_modify_profile(self) -> None:
        before = deepcopy(self.config)

        run_geometry_diagnostics(
            self.frames,
            self.config,
            max_input_width=960,
            use_intrinsics=False,
        )

        self.assertEqual(before, self.config)

    def test_pair_only_results_use_only_declared_pair(self) -> None:
        result = run_geometry_diagnostics(
            self.frames,
            self.config,
            max_input_width=960,
            use_intrinsics=False,
            pair_mode=ALPHA_50,
        )

        self.assertEqual(
            ["front_left", "front"],
            result.metadata["pairs"][0]["cameras"],
        )
        self.assertEqual(
            ["front", "front_right"],
            result.metadata["pairs"][1]["cameras"],
        )
        left_front = result.pair_only["left_front"]
        front_right = result.pair_only["front_right"]
        self.assertFalse(
            np.any(np.all(left_front == np.array([255, 0, 0]), axis=2))
        )
        self.assertFalse(
            np.any(np.all(front_right == np.array([0, 0, 255]), axis=2))
        )

    def test_overlap_and_seam_metadata_match_profile(self) -> None:
        result = run_geometry_diagnostics(
            self.frames,
            self.config,
            max_input_width=960,
            use_intrinsics=False,
        )
        profile = active_stitch_profile(self.config)

        self.assertEqual(profile["overlaps"], result.metadata["overlaps"])
        self.assertEqual(
            profile["stitch_points"],
            result.metadata["stitch_points"],
        )
        for pair, overlap in zip(
            result.metadata["pairs"],
            profile["overlaps"],
        ):
            self.assertEqual(
                seam_x(profile["stitch_points"], overlap["seam"]),
                pair["seam_x"],
            )

    def test_coordinate_scale_matches_runtime_resize(self) -> None:
        result = run_geometry_diagnostics(
            self.frames,
            self.config,
            max_input_width=960,
            use_intrinsics=False,
        )

        camera = result.metadata["cameras"]["front"]
        self.assertEqual([1920, 1080], camera["raw_size"])
        self.assertEqual([960, 540], camera["normalized_size"])
        self.assertEqual(0.5, camera["raw_to_normalized_scale"])
        self.assertEqual(2.0, camera["normalized_to_source_factor"])
        self.assertEqual("raw_frame_pixels", camera["source_point_space"])
        self.assertEqual([1920, 1080], camera["source_reference_size"])
        self.assertEqual([0.0, 0.0, 1919.0, 1079.0], camera["source_bounds"])
        self.assertEqual([1.0, 1.0], camera["source_coverage_fraction"])
        self.assertEqual("loaded_image_pixels", camera["editor_point_space"])
        self.assertEqual([], camera["coordinate_warnings"])
        self.assertFalse(camera["intrinsics_effective"])

    def test_source_contract_warns_for_normalized_points(self) -> None:
        profile = deepcopy(active_stitch_profile(self.config))
        profile["cameras"]["front"]["source_points"] = [
            [0.0, 0.0],
            [0.0, 539.0],
            [959.0, 0.0],
            [959.0, 539.0],
        ]

        diagnostics = source_coordinate_diagnostics(
            profile,
            "front",
            (1920, 1080),
        )

        self.assertAlmostEqual(959.0 / 1919.0, diagnostics["coverage_fraction"][0])
        self.assertAlmostEqual(539.0 / 1079.0, diagnostics["coverage_fraction"][1])
        self.assertTrue(
            any(
                "suspected normalized/raw coordinate mix" in warning
                for warning in diagnostics["warnings"]
            )
        )

    def test_source_contract_warns_for_actual_size_mismatch(self) -> None:
        profile = active_stitch_profile(self.config)

        diagnostics = source_coordinate_diagnostics(
            profile,
            "front",
            (3840, 2160),
        )

        self.assertTrue(
            any(
                "actual raw frame is 3840x2160" in warning
                for warning in diagnostics["warnings"]
            )
        )

    def test_intrinsics_status_reports_scaling_without_enabling_it(self) -> None:
        result = run_geometry_diagnostics(
            self.frames,
            self.config,
            max_input_width=960,
            use_intrinsics=False,
        )

        left = result.metadata["cameras"]["front_left"]
        front = result.metadata["cameras"]["front"]
        right = result.metadata["cameras"]["front_right"]
        for camera in (left, right):
            self.assertTrue(camera["intrinsics_available"])
            self.assertEqual([3840, 2160], camera["intrinsics_reference_size"])
            self.assertEqual([0.5, 0.5], camera["intrinsics_scale_to_raw"])
            self.assertEqual(
                [0.25, 0.25],
                camera["intrinsics_scale_to_normalized"],
            )
            self.assertTrue(camera["intrinsics_uniform_scale_compatible"])
            self.assertFalse(camera["intrinsics_runtime_matrix_scaled"])
            self.assertFalse(camera["intrinsics_effective"])
        self.assertFalse(front["intrinsics_available"])

    def test_normalized_full_frame_maps_to_target_quadrilateral(self) -> None:
        stitcher = SurroundStitcher(
            self.config,
            max_input_width=960,
            use_intrinsics=False,
        )
        calibration = stitcher.cameras["front"]
        runtime_matrix = calibration.matrix @ np.asarray(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        normalized_corners = np.asarray(
            [[[0.0, 0.0], [0.0, 539.0], [959.0, 0.0], [959.0, 539.0]]],
            dtype=np.float32,
        )

        mapped = cv2.perspectiveTransform(
            normalized_corners,
            runtime_matrix,
        )[0]

        np.testing.assert_allclose(
            mapped,
            calibration.target_points,
            atol=1.5,
        )

    def test_warp_uses_all_four_raw_frame_quadrants(self) -> None:
        frame = np.empty((1080, 1920, 3), dtype=np.uint8)
        frame[:540, :960] = (0, 0, 255)
        frame[540:, :960] = (0, 255, 0)
        frame[:540, 960:] = (255, 0, 0)
        frame[540:, 960:] = (0, 255, 255)
        stitcher = SurroundStitcher(
            self.config,
            max_input_width=960,
            use_intrinsics=False,
        )

        warped = stitcher.warp(frame, "front")

        expected_samples = {
            (175, 875): np.asarray((0, 0, 255)),
            (525, 875): np.asarray((0, 255, 0)),
            (175, 1325): np.asarray((255, 0, 0)),
            (525, 1325): np.asarray((0, 255, 255)),
        }
        for (y, x), expected in expected_samples.items():
            np.testing.assert_allclose(
                warped[y, x],
                expected,
                atol=3,
            )

    def test_current_triple_seams_are_inside_declared_overlaps(self) -> None:
        current_config = load_config("calibration.yaml")
        profile = active_stitch_profile(current_config)

        self.assertEqual("triple_front_panorama", current_config["stitch_topology"])
        self.assertEqual(750.0, seam_x(profile["stitch_points"], "left_front"))
        self.assertEqual(1450.0, seam_x(profile["stitch_points"], "front_right"))
        self.assertEqual([], validate_overlap_seams(profile))

    def test_triple_source_migration_is_traceable(self) -> None:
        profile = active_stitch_profile(load_config("calibration.yaml"))
        migration = profile["source_coordinate_migration"]

        self.assertTrue(migration["applied"])
        self.assertEqual(
            "normalized_input_pixels",
            migration["from_coordinate_space"],
        )
        self.assertEqual([960, 540], migration["from_reference_size"])
        self.assertEqual(
            "raw_frame_pixels",
            migration["to_coordinate_space"],
        )
        self.assertEqual([1920, 1080], migration["to_reference_size"])
        self.assertIn("not a final", migration["note"].lower())

    def test_no_feather_mode_uses_hard_seam(self) -> None:
        left = np.full((2, 4, 3), (0, 0, 255), dtype=np.uint8)
        right = np.full((2, 4, 3), (0, 255, 0), dtype=np.uint8)

        result = compose_pair_only(
            left,
            right,
            mode=NO_FEATHER,
            seam_position=2,
        )

        self.assertTrue(np.all(result[:, :2] == left[:, :2]))
        self.assertTrue(np.all(result[:, 2:] == right[:, 2:]))


if __name__ == "__main__":
    unittest.main()
