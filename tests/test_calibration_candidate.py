from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from deep_shark_studio import calibration_candidate
from deep_shark_studio.calibration_candidate import (
    CandidatePanoramaProcessor,
    generate_calibration_candidate,
    latest_calibration_candidate,
    load_calibration_candidate,
)
from deep_shark_studio.calibration_session import CalibrationSession
from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.stitcher import save_image
from deep_shark_studio.stitch_processing import StitchProcessingManager


def rotation_y(angle: float) -> np.ndarray:
    return cv2.Rodrigues(
        np.asarray([0.0, angle, 0.0], dtype=np.float64)
    )[0]


def project(
    object_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[list[float]]:
    image_points, _ = cv2.fisheye.projectPoints(
        object_points,
        cv2.Rodrigues(rotation)[0],
        translation,
        camera_matrix,
        distortion,
    )
    return image_points.reshape(-1, 2).tolist()


def create_synthetic_session(root: Path) -> CalibrationSession:
    session = CalibrationSession.create(
        sessions_root=root / "sessions",
        topology="triple_front_panorama",
        resolution=(1920, 1080),
        source_coordinate_space="raw_frame_pixels",
        source_reference_size=[1920, 1080],
        source_contract_version=1,
        board_definition={
            "type": "chessboard",
            "inner_columns": 6,
            "inner_rows": 5,
            "square_size_mm": 30.0,
        },
        board_confirmed=True,
        captured_at=datetime(
            2026,
            7,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )
    columns, rows = 6, 5
    object_points = np.zeros((columns * rows, 1, 3), dtype=np.float64)
    object_points[:, 0, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * 30.0
    )
    point_ids = [str(index) for index in range(columns * rows)]
    camera_matrix = np.asarray(
        [
            [790.0, 0.0, 960.0],
            [0.0, 795.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [-0.025, 0.008, -0.002, 0.0005],
        dtype=np.float64,
    ).reshape(4, 1)
    left_to_front_rotation = rotation_y(-0.12)
    left_to_front_translation = np.asarray(
        [[-180.0], [2.0], [-5.0]],
        dtype=np.float64,
    )
    front_to_right_rotation = rotation_y(-0.12)
    front_to_right_translation = np.asarray(
        [[-180.0], [-2.0], [5.0]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(42)
    records_by_camera = {key: [] for key in ("front_left", "front", "front_right")}
    pair_records = {
        "front_left__front": [],
        "front__front_right": [],
    }
    coverage_zones = [
        f"{row}_{column}"
        for row in ("top", "middle", "bottom")
        for column in ("left", "center", "right")
    ]
    for index in range(20):
        board_rotation_front = cv2.Rodrigues(
            np.asarray(
                [
                    -0.20 + 0.02 * (index % 5),
                    -0.16 + 0.025 * (index % 7),
                    -0.10 + 0.02 * (index % 6),
                ],
                dtype=np.float64,
            )
        )[0]
        board_translation_front = np.asarray(
            [
                [-150.0 + 28.0 * (index % 8)],
                [-110.0 + 38.0 * (index % 6)],
                [820.0 + 35.0 * (index % 7)],
            ],
            dtype=np.float64,
        )
        board_rotation_left = (
            left_to_front_rotation.T @ board_rotation_front
        )
        board_translation_left = left_to_front_rotation.T @ (
            board_translation_front - left_to_front_translation
        )
        board_rotation_right = (
            front_to_right_rotation @ board_rotation_front
        )
        board_translation_right = (
            front_to_right_rotation @ board_translation_front
            + front_to_right_translation
        )
        geometry = {
            "front_left": (
                board_rotation_left,
                board_translation_left,
            ),
            "front": (
                board_rotation_front,
                board_translation_front,
            ),
            "front_right": (
                board_rotation_right,
                board_translation_right,
            ),
        }
        points_by_camera = {}
        for camera, (rotation, translation) in geometry.items():
            points = np.asarray(
                project(
                    object_points,
                    rotation,
                    translation,
                    camera_matrix,
                    distortion,
                ),
                dtype=np.float64,
            )
            points += rng.normal(0.0, 0.12, points.shape)
            points_by_camera[camera] = points.tolist()
            records_by_camera[camera].append(
                {
                    "sample_id": f"{camera}-{index:02d}",
                    "accepted": True,
                    "rejection_reasons": [],
                    "warnings": [],
                    "point_ids": point_ids,
                    "image_points": points.tolist(),
                    "metrics": {
                        "coverage_zone": coverage_zones[
                            index % len(coverage_zones)
                        ],
                        "pose_signature": [float(index)] * 4,
                    },
                    "files": {},
                }
            )
        if index < 15:
            for pair_key, left, right in (
                ("front_left__front", "front_left", "front"),
                ("front__front_right", "front", "front_right"),
            ):
                pair_records[pair_key].append(
                    {
                        "sample_id": f"{pair_key}-{index:02d}",
                        "accepted": True,
                        "rejection_reasons": [],
                        "warnings": [
                            "Software frame timestamps are not hardware "
                            "synchronization."
                        ],
                        "common_point_ids": point_ids,
                        "left_common_points": points_by_camera[left],
                        "right_common_points": points_by_camera[right],
                        "pair_coverage_zone": "center_far",
                        "pair_pose_signature": [float(index)] * 8,
                        "frame_time_delta_seconds": 0.015,
                    }
                )
    for camera, records in records_by_camera.items():
        image_path = (
            session.directory
            / "intrinsics"
            / camera
            / "accepted"
            / "preview.png"
        )
        image = np.full(
            (1080, 1920, 3),
            {
                "front_left": (180, 30, 30),
                "front": (30, 180, 30),
                "front_right": (30, 30, 180),
            }[camera],
            dtype=np.uint8,
        )
        save_image(image_path, image)
        for record in records:
            record["files"]["raw"] = image_path.relative_to(
                session.directory
            ).as_posix()
        session.data["intrinsics"][camera]["samples"] = records
    for pair_key, records in pair_records.items():
        session.data["stereo_pairs"][pair_key]["samples"] = records
    session._save()
    return session


class CalibrationCandidateTests(unittest.TestCase):
    def test_experimental_candidate_is_reproducible_and_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = create_synthetic_session(root)
            formal_path = CONFIG_DIR / "calibration.yaml"
            formal_before = file_revision(formal_path)
            with patch.object(
                calibration_candidate,
                "PROJECT_ROOT",
                root,
            ):
                output = generate_calibration_candidate(
                    session.directory,
                    output_root=root / "candidates",
                    generated_at=datetime(
                        2026,
                        7,
                        1,
                        13,
                        0,
                        tzinfo=timezone.utc,
                    ),
                )
            candidate, report = load_calibration_candidate(output)

            self.assertEqual(formal_before, file_revision(formal_path))
            self.assertEqual("experimental", candidate["quality_level"])
            self.assertTrue(candidate["experimental"])
            self.assertTrue(candidate["report_only"])
            self.assertFalse(candidate["recommended"])
            self.assertFalse(candidate["apply_allowed"])
            self.assertEqual([1920, 1080], candidate["resolution"])
            self.assertTrue(candidate["rig"]["complete"])
            self.assertTrue(report["summary"]["panorama_generated"])
            for result in candidate["intrinsics"].values():
                self.assertEqual("success", result["status"])
                self.assertEqual([1920, 1080], result["resolution"])
                self.assertIsNotNone(result["rms_px"])
            for result in candidate["stereo_pairs"].values():
                self.assertEqual("success", result["status"])
                self.assertEqual([1920, 1080], result["resolution"])
                self.assertIsNotNone(result["rms_px"])
                self.assertEqual(
                    "software_timestamp_only",
                    result["time_delta_statistics"]["timestamp_model"],
                )
            panorama = candidate["virtual_panorama"]
            self.assertTrue((output / panorama["files"]["canvas"]).exists())
            for relative_path in panorama["files"]["remaps"].values():
                self.assertTrue((output / relative_path).exists())
            for pair_files in panorama["files"]["pair_previews"].values():
                self.assertTrue(
                    (output / pair_files["alpha_50"]).exists()
                )
                self.assertTrue(
                    (output / pair_files["hard_seam"]).exists()
                )
            self.assertEqual(
                output,
                latest_calibration_candidate(
                    root / "candidates",
                    topology="triple_front_panorama",
                ),
            )

            processor = CandidatePanoramaProcessor(output)
            frames = {
                camera: np.full(
                    (1080, 1920, 3),
                    value,
                    dtype=np.uint8,
                )
                for camera, value in (
                    ("front_left", (220, 30, 30)),
                    ("front", (30, 220, 30)),
                    ("front_right", (30, 30, 220)),
                )
            }
            warped, canvas = processor.process(frames)
            self.assertEqual(
                {"front_left", "front", "front_right"},
                set(warped),
            )
            self.assertEqual((600, 1800, 3), canvas.shape)
            self.assertGreater(np.count_nonzero(canvas), 0)
            self.assertIn(
                tuple(processor.camera_order),
                processor._selection_masks_by_available,
            )
            self.assertTrue(processor.last_timings["precomputed_selection_masks"])
            self.assertGreaterEqual(processor.last_timings["candidate_remap_ms"], 0.0)
            self.assertGreaterEqual(processor.last_timings["candidate_compose_ms"], 0.0)
            if cv2.ocl.haveOpenCL():
                opencl_processor = CandidatePanoramaProcessor(output, use_opencl=True)
                _opencl_warped, opencl_canvas = opencl_processor.process(frames)
                self.assertEqual(canvas.shape, opencl_canvas.shape)
                mean_abs_diff = float(
                    np.mean(
                        np.abs(
                            canvas.astype(np.int16)
                            - opencl_canvas.astype(np.int16)
                        )
                    )
                )
                self.assertLess(mean_abs_diff, 1.0)
                self.assertTrue(opencl_processor.last_timings["opencl_enabled"])
                self.assertGreaterEqual(
                    opencl_processor.last_timings["candidate_download_ms"],
                    0.0,
                )

            manager = StitchProcessingManager()
            try:
                manager.configure(
                    9,
                    {},
                    None,
                    False,
                    processor_mode="candidate",
                    candidate_directory=str(output),
                )
                manager.submit_latest(9, frames)
                deadline = time.monotonic() + 3.0
                result = None
                while time.monotonic() < deadline:
                    result = manager.take_latest_result()
                    if result is not None:
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(result)
                self.assertEqual("", result.error)
                self.assertEqual((600, 1800, 3), result.canvas.shape)
                self.assertEqual("b2_candidate_view", result.runtime_status)
                self.assertIn(
                    "candidate_remap_ms",
                    result.runtime_metrics["timing"],
                )
            finally:
                self.assertTrue(manager.shutdown())

    def test_runtime_candidate_rejects_wrong_frame_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = create_synthetic_session(root)
            with patch.object(
                calibration_candidate,
                "PROJECT_ROOT",
                root,
            ):
                output = generate_calibration_candidate(
                    session.directory,
                    output_root=root / "candidates",
                )
            processor = CandidatePanoramaProcessor(output)
            with self.assertRaisesRegex(ValueError, "does not match"):
                processor.process(
                    {
                        "front": np.zeros(
                            (540, 960, 3),
                            dtype=np.uint8,
                        )
                    }
                )


if __name__ == "__main__":
    unittest.main()
