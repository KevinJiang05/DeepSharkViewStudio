from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from deep_shark_studio.calibration_snapshot import (
    SNAPSHOT_FORMAT,
    SNAPSHOT_SCHEMA_VERSION,
    save_calibration_snapshot,
    topology_diagnostics,
)
from deep_shark_studio.config import load_config, load_yaml, save_yaml
from deep_shark_studio.stitcher import SurroundStitcher


def solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((540, 960, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


class CalibrationSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("calibration.yaml")

    def test_triple_diagnostics_expose_two_readable_overlaps(self) -> None:
        diagnostics = topology_diagnostics(self.config)

        self.assertEqual("triple_front_panorama", diagnostics["topology"])
        self.assertEqual(
            ["front_left", "front", "front_right"],
            diagnostics["camera_order"],
        )
        self.assertEqual(2, len(diagnostics["overlaps"]))
        self.assertEqual(
            [
                ["front_left", "front"],
                ["front", "front_right"],
            ],
            [item["cameras"] for item in diagnostics["overlaps"]],
        )
        for overlap in diagnostics["overlaps"]:
            self.assertIsNotNone(overlap["seam_x"])
            self.assertLessEqual(overlap["x_range"][0], overlap["seam_x"])
            self.assertLessEqual(overlap["seam_x"], overlap["x_range"][1])

    def test_snapshot_saves_available_frames_canvas_and_metadata_only(self) -> None:
        frames = {
            "front_left": solid_frame((0, 0, 255)),
            "front": solid_frame((0, 255, 0)),
        }
        stitcher = SurroundStitcher(
            self.config,
            max_input_width=None,
            use_intrinsics=False,
        )
        _, canvas = stitcher.process(frames)

        with tempfile.TemporaryDirectory() as directory:
            snapshot_dir = save_calibration_snapshot(
                snapshots_root=directory,
                frames=frames,
                canvas=canvas,
                calibration_config=self.config,
                frame_ages_seconds={
                    "front_left": 0.03,
                    "front": 0.05,
                    "front_right": None,
                },
                captured_at=datetime(
                    2026,
                    7,
                    1,
                    12,
                    30,
                    45,
                    123000,
                    tzinfo=timezone.utc,
                ),
            )

            self.assertEqual(
                {
                    "raw_front_left.png",
                    "raw_front.png",
                    "canvas.png",
                    "metadata.yaml",
                },
                {path.name for path in snapshot_dir.iterdir()},
            )
            self.assertIsNotNone(cv2.imread(str(snapshot_dir / "canvas.png")))
            metadata = load_yaml(snapshot_dir / "metadata.yaml")
            self.assertEqual(SNAPSHOT_FORMAT, metadata["format"])
            self.assertEqual(
                SNAPSHOT_SCHEMA_VERSION,
                metadata["schema_version"],
            )
            self.assertEqual(
                ["front_right"],
                metadata["missing_cameras"],
            )
            self.assertEqual(0.03, metadata["frame_age_seconds"]["front_left"])
            self.assertEqual(
                "initial_template",
                metadata["calibration_origin"]["target_points"],
            )
            serialized = (snapshot_dir / "metadata.yaml").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("rtsp://", serialized.lower())
            self.assertNotIn("password", serialized.lower())

            second_snapshot_dir = save_calibration_snapshot(
                snapshots_root=directory,
                frames=frames,
                canvas=canvas,
                calibration_config=self.config,
                frame_ages_seconds={},
                captured_at=datetime(
                    2026,
                    7,
                    1,
                    12,
                    30,
                    45,
                    123000,
                    tzinfo=timezone.utc,
                ),
            )
            self.assertNotEqual(snapshot_dir, second_snapshot_dir)
            self.assertEqual(
                "20260701_123045_123_01",
                second_snapshot_dir.name,
            )

    def test_geometry_round_trip_preserves_selected_profile(self) -> None:
        before = topology_diagnostics(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            save_yaml(path, self.config)
            after = topology_diagnostics(load_yaml(path))

        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
