from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from deep_shark_studio.config import load_config, save_yaml
from deep_shark_studio.pairwise_diagnostics import (
    PAIRWISE_FORMAT,
    estimate_pairwise_candidate,
    pair_definitions,
    save_pairwise_results,
)


def chessboard_frame() -> np.ndarray:
    frame = np.full((1080, 1920, 3), 127, dtype=np.uint8)
    square = 70
    start_x, start_y = 300, 220
    for row in range(9):
        for column in range(12):
            value = 255 if (row + column) % 2 else 0
            cv2.rectangle(
                frame,
                (
                    start_x + column * square,
                    start_y + row * square,
                ),
                (
                    start_x + (column + 1) * square,
                    start_y + (row + 1) * square,
                ),
                (value, value, value),
                -1,
            )
    return frame


def write_snapshot(
    directory: Path,
    frames: dict[str, np.ndarray],
) -> Path:
    snapshot = directory / "20260701_120000_000"
    snapshot.mkdir()
    raw_files = {}
    for key, frame in frames.items():
        filename = f"raw_{key}.png"
        cv2.imwrite(str(snapshot / filename), frame)
        raw_files[key] = filename
    save_yaml(
        snapshot / "metadata.yaml",
        {
            "format": "DeepSharkCalibrationSnapshot",
            "schema_version": 2,
            "topology": "triple_front_panorama",
            "camera_order": ["front_left", "front", "front_right"],
            "files": {"raw": raw_files},
        },
    )
    return snapshot


class PairwiseDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = deepcopy(load_config("calibration.yaml"))
        self.left_front = pair_definitions(self.config)[0]

    def test_chessboard_candidate_detects_common_points(self) -> None:
        left = chessboard_frame()
        transform = np.asarray(
            [
                [1.0, 0.03, 60.0],
                [-0.01, 1.0, 30.0],
                [0.00002, 0.00001, 1.0],
            ],
            dtype=np.float64,
        )
        right = cv2.warpPerspective(
            left,
            transform,
            (1920, 1080),
            borderValue=(127, 127, 127),
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = write_snapshot(
                Path(directory),
                {
                    "front_left": left,
                    "front": right,
                },
            )

            result = estimate_pairwise_candidate(
                snapshot,
                self.config,
                self.left_front,
            )

        self.assertTrue(result.method.startswith("chessboard:"))
        self.assertEqual(88, result.point_count)
        self.assertIsNotNone(result.homography)
        self.assertIsNotNone(result.alpha_overlay)
        self.assertIsNotNone(result.hard_cut_overlay)
        self.assertLess(result.reprojection_error, 2.0)

    def test_manual_candidate_is_temporary_and_does_not_mutate_profile(self) -> None:
        frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)
        before = deepcopy(self.config)
        left_points = [
            [300.0, 250.0],
            [300.0, 800.0],
            [1500.0, 250.0],
            [1500.0, 800.0],
        ]
        right_points = [
            [340.0, 270.0],
            [340.0, 820.0],
            [1540.0, 270.0],
            [1540.0, 820.0],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = write_snapshot(
                root,
                {
                    "front_left": frame,
                    "front": frame,
                },
            )
            result = estimate_pairwise_candidate(
                snapshot,
                self.config,
                self.left_front,
                manual_left_points=left_points,
                manual_right_points=right_points,
            )
            output = save_pairwise_results(
                root / "outputs",
                [result],
                captured_at=datetime(
                    2026,
                    7,
                    1,
                    12,
                    30,
                    tzinfo=timezone.utc,
                ),
            )
            metadata = json.loads(
                (output / "diagnostics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(before, self.config)
        self.assertEqual("manual_correspondences", result.method)
        self.assertEqual(4, result.point_count)
        self.assertEqual(PAIRWISE_FORMAT, metadata["format"])
        self.assertFalse(metadata["writes_formal_profile"])
        self.assertFalse(metadata["results"][0]["writes_formal_profile"])
        self.assertNotIn("calibration.yaml", json.dumps(metadata))

    def test_automatic_failure_can_fall_back_to_manual_points(self) -> None:
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        points = [
            [200.0, 200.0],
            [200.0, 800.0],
            [1600.0, 200.0],
            [1600.0, 800.0],
        ]
        with tempfile.TemporaryDirectory() as directory:
            snapshot = write_snapshot(
                Path(directory),
                {
                    "front_left": blank,
                    "front": blank,
                },
            )
            automatic = estimate_pairwise_candidate(
                snapshot,
                self.config,
                self.left_front,
            )
            manual = estimate_pairwise_candidate(
                snapshot,
                self.config,
                self.left_front,
                manual_left_points=points,
                manual_right_points=points,
            )

        self.assertIsNone(automatic.homography)
        self.assertIn("manual correspondences", automatic.error)
        self.assertIsNotNone(manual.homography)
        self.assertEqual("", manual.error)


if __name__ == "__main__":
    unittest.main()
