from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from deep_shark_studio.calibration_session import (
    BoardDefinitionNotConfirmedError,
    CalibrationSession,
    detect_board,
    validate_board_definition,
)
from deep_shark_studio.config import load_config


def chessboard_definition() -> dict:
    return {
        "type": "chessboard",
        "inner_columns": 11,
        "inner_rows": 8,
        "square_size_mm": 25.0,
    }


def chessboard_frame(
    shift_x: int = 0,
    shift_y: int = 0,
    square: int = 70,
) -> np.ndarray:
    frame = np.full((1080, 1920, 3), 127, dtype=np.uint8)
    start_x, start_y = 300 + shift_x, 220 + shift_y
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


def create_session(
    root: Path,
    confirmed: bool = True,
    quality_gate: dict | None = None,
) -> CalibrationSession:
    return CalibrationSession.create(
        sessions_root=root,
        topology="triple_front_panorama",
        resolution=(1920, 1080),
        source_coordinate_space="raw_frame_pixels",
        source_reference_size=[1920, 1080],
        source_contract_version=1,
        board_definition=chessboard_definition(),
        board_confirmed=confirmed,
        quality_gate={
            "minimum_board_area_ratio": 0.005,
            "minimum_blur_variance": 10.0,
            **(quality_gate or {}),
        },
        captured_at=datetime(
            2026,
            7,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )


def populate_ready_counts(
    session: CalibrationSession,
    mono_count: int = 20,
    pair_count: int = 15,
    diverse_coverage: bool = True,
) -> None:
    mono_zones = [
        f"{row}_{column}"
        for row in ("top", "middle", "bottom")
        for column in ("left", "center", "right")
    ]
    pair_zones = [
        f"{position}_{distance}"
        for position in ("left", "center", "right")
        for distance in ("far", "middle", "near")
    ]
    for camera in ("front_left", "front", "front_right"):
        session.data["intrinsics"][camera]["samples"] = [
            {
                "accepted": True,
                "rejection_reasons": [],
                "warnings": [],
                "metrics": {
                    "coverage_zone": (
                        mono_zones[index % len(mono_zones)]
                        if diverse_coverage
                        else "middle_center"
                    ),
                    "pose_signature": [0.1, 0.1, 0.1, 0.1],
                },
            }
            for index in range(mono_count)
        ]
    for pair in ("front_left__front", "front__front_right"):
        session.data["stereo_pairs"][pair]["samples"] = [
            {
                "accepted": True,
                "rejection_reasons": [],
                "warnings": [],
                "pair_coverage_zone": (
                    pair_zones[index % len(pair_zones)]
                    if diverse_coverage
                    else "center_middle"
                ),
                "pair_pose_signature": [0.1] * 8,
            }
            for index in range(pair_count)
        ]


class CalibrationSessionTests(unittest.TestCase):
    def test_aruco_and_charuco_definitions_are_detectable(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        grid = cv2.aruco.GridBoard((4, 3), 0.04, 0.01, dictionary)
        grid_image = cv2.cvtColor(
            grid.generateImage((800, 600), marginSize=20),
            cv2.COLOR_GRAY2BGR,
        )
        charuco = cv2.aruco.CharucoBoard(
            (7, 5),
            0.04,
            0.028,
            dictionary,
        )
        charuco_image = cv2.cvtColor(
            charuco.generateImage((900, 650), marginSize=30),
            cv2.COLOR_GRAY2BGR,
        )

        grid_detection = detect_board(
            grid_image,
            {
                "type": "aruco_grid",
                "markers_x": 4,
                "markers_y": 3,
                "marker_length_mm": 40,
                "marker_separation_mm": 10,
                "dictionary": "DICT_4X4_50",
            },
        )
        charuco_detection = detect_board(
            charuco_image,
            {
                "type": "charuco",
                "squares_x": 7,
                "squares_y": 5,
                "square_length_mm": 40,
                "marker_length_mm": 28,
                "dictionary": "DICT_4X4_50",
            },
        )

        self.assertTrue(grid_detection.detected)
        self.assertGreaterEqual(len(grid_detection.points), 8)
        self.assertTrue(charuco_detection.detected)
        self.assertGreaterEqual(len(charuco_detection.points), 6)

    def test_unconfirmed_or_incomplete_board_cannot_capture(self) -> None:
        self.assertTrue(
            validate_board_definition(
                {
                    "type": "chessboard",
                    "inner_columns": 0,
                    "inner_rows": 8,
                    "square_size_mm": 0,
                }
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory), confirmed=False)

            with self.assertRaises(BoardDefinitionNotConfirmedError):
                session.capture_intrinsic(
                    "front",
                    chessboard_frame(),
                )

            self.assertEqual(
                [],
                session.data["intrinsics"]["front"]["samples"],
            )

    def test_intrinsic_samples_are_saved_under_camera_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory))

            record = session.capture_intrinsic(
                "front",
                chessboard_frame(),
                frame_timestamp=123.4,
            )

            self.assertTrue(record["accepted"])
            self.assertEqual(88, record["metrics"]["point_count"])
            raw_path = session.directory / record["files"]["raw"]
            metadata_path = session.directory / record["files"]["metadata"]
            self.assertTrue(raw_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertIn(
                "intrinsics/front/accepted",
                record["files"]["raw"],
            )

    def test_pair_requires_common_detection_and_rejects_large_time_delta(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory))
            valid = chessboard_frame()
            blank = np.zeros_like(valid)
            before_count = len(
                session.data["stereo_pairs"]["front_left__front"]["samples"]
            )
            inspection = session.inspect_pair(
                "front_left",
                "front",
                valid,
                valid,
                0.0,
                0.02,
            )
            self.assertEqual(88, inspection["common_point_count"])
            self.assertTrue(inspection["common_points_sufficient"])
            self.assertEqual(
                before_count,
                len(
                    session.data["stereo_pairs"]["front_left__front"][
                        "samples"
                    ]
                ),
            )

            missing = session.capture_pair(
                "front_left",
                "front",
                valid,
                blank,
                1.0,
                1.01,
                "left_board",
                True,
            )
            late = session.capture_pair(
                "front_left",
                "front",
                valid,
                valid,
                2.0,
                2.25,
                "left_board",
                True,
            )
            accepted = session.capture_pair(
                "front_left",
                "front",
                valid,
                valid,
                3.0,
                3.02,
                "left_board",
                True,
            )

        self.assertFalse(missing["accepted"])
        self.assertTrue(
            any("common confirmed board points" in reason for reason in missing["rejection_reasons"])
        )
        self.assertFalse(late["accepted"])
        self.assertTrue(
            any("time delta" in reason for reason in late["rejection_reasons"])
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(88, len(accepted["common_point_ids"]))
        self.assertIn(
            "stereo_pairs/front_left__front/accepted",
            accepted["files"]["left_raw"],
        )
        self.assertTrue(
            any("not hardware" in warning for warning in accepted["warnings"])
        )

    def test_duplicate_and_blur_gates_keep_rejection_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = create_session(root)
            captured = datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc)
            first = session.capture_intrinsic(
                "front_left",
                chessboard_frame(),
                captured_at=captured,
            )
            duplicate = session.capture_intrinsic(
                "front_left",
                chessboard_frame(),
                captured_at=captured + timedelta(seconds=1),
            )

            blur_session = CalibrationSession.create(
                sessions_root=root,
                topology="triple_front_panorama",
                resolution=(1920, 1080),
                source_coordinate_space="raw_frame_pixels",
                source_reference_size=[1920, 1080],
                source_contract_version=1,
                board_definition=chessboard_definition(),
                board_confirmed=True,
                quality_gate={"minimum_blur_variance": 1e9},
                captured_at=captured + timedelta(seconds=2),
            )
            blurred = blur_session.capture_intrinsic(
                "front_right",
                chessboard_frame(shift_x=80),
            )

        self.assertTrue(first["accepted"])
        self.assertFalse(duplicate["accepted"])
        self.assertTrue(
            any("too similar" in reason for reason in duplicate["rejection_reasons"])
        )
        self.assertFalse(blurred["accepted"])
        self.assertTrue(
            any("blurred" in reason for reason in blurred["rejection_reasons"])
        )

    def test_small_board_is_rejected_but_edge_coverage_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small_gate_session = create_session(
                root,
                quality_gate={"minimum_board_area_ratio": 0.5},
            )
            too_small = small_gate_session.capture_intrinsic(
                "front",
                chessboard_frame(),
            )
            edge_session = CalibrationSession.create(
                sessions_root=root,
                topology="triple_front_panorama",
                resolution=(1920, 1080),
                source_coordinate_space="raw_frame_pixels",
                source_reference_size=[1920, 1080],
                source_contract_version=1,
                board_definition=chessboard_definition(),
                board_confirmed=True,
                quality_gate={
                    "minimum_board_area_ratio": 0.005,
                    "minimum_blur_variance": 10.0,
                },
                captured_at=datetime(
                    2026,
                    7,
                    1,
                    14,
                    0,
                    tzinfo=timezone.utc,
                ),
            )
            edge = edge_session.capture_intrinsic(
                "front",
                chessboard_frame(shift_x=-250, shift_y=-180),
            )

        self.assertFalse(too_small["accepted"])
        self.assertIn(
            "Board area is too small.",
            too_small["rejection_reasons"],
        )
        self.assertTrue(edge["accepted"])
        self.assertTrue(
            edge["metrics"]["coverage_zone"].startswith("top_")
        )

    def test_session_round_trip_preserves_definition_samples_and_reasons(
        self,
    ) -> None:
        formal_before = deepcopy(load_config("calibration.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory))
            session.capture_intrinsic(
                "front",
                np.zeros((1080, 1920, 3), dtype=np.uint8),
            )

            loaded = CalibrationSession.load(session.directory)
            serialized = (session.directory / "session.yaml").read_text(
                encoding="utf-8"
            )

        self.assertEqual(
            chessboard_definition(),
            loaded.data["board_definition"],
        )
        self.assertEqual(
            [1920, 1080],
            loaded.data["resolution"],
        )
        self.assertEqual(
            "raw_frame_pixels",
            loaded.data["source_coordinate_space"],
        )
        self.assertEqual(
            1,
            len(loaded.data["intrinsics"]["front"]["samples"]),
        )
        self.assertTrue(
            loaded.data["intrinsics"]["front"]["samples"][0][
                "rejection_reasons"
            ]
        )
        self.assertFalse(loaded.data["formal_profile_modified"])
        self.assertNotIn("rtsp://", serialized.lower())
        self.assertNotIn("password", serialized.lower())
        self.assertEqual(formal_before, load_config("calibration.yaml"))

    def test_readiness_requires_minimum_counts_and_quality_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory))
            empty = session.readiness()
            populate_ready_counts(
                session,
                diverse_coverage=False,
            )
            weak = session.readiness()
            populate_ready_counts(
                session,
                diverse_coverage=True,
            )
            ready = session.readiness()

        self.assertEqual("not_recommended", empty["status"])
        self.assertFalse(empty["minimum_counts_met"])
        self.assertEqual("try_with_risk", weak["status"])
        self.assertTrue(weak["quality_issues"])
        self.assertEqual("ready", ready["status"])
        self.assertEqual("可进入 B-2 求解", ready["label"])

    def test_b2_access_keeps_experimental_results_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory))
            blocked = session.b2_candidate_access()
            populate_ready_counts(session, diverse_coverage=False)
            experimental = session.b2_candidate_access()
            populate_ready_counts(session, diverse_coverage=True)
            official = session.b2_candidate_access()

        self.assertFalse(blocked["can_enter"])
        self.assertEqual("blocked", blocked["mode"])
        self.assertTrue(experimental["can_enter"])
        self.assertEqual("experimental", experimental["mode"])
        self.assertTrue(experimental["experimental"])
        self.assertTrue(experimental["report_only"])
        self.assertFalse(experimental["apply_allowed"])
        self.assertIn("rms_metrics", experimental["expected_outputs"])
        self.assertIn(
            "candidate_panorama_preview",
            experimental["expected_outputs"],
        )
        self.assertEqual("official", official["mode"])
        self.assertFalse(official["experimental"])
        self.assertFalse(official["report_only"])
        self.assertTrue(official["apply_allowed"])

    def test_readiness_flags_excessive_pair_time_risks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_session(Path(directory))
            populate_ready_counts(session)
            samples = session.data["stereo_pairs"]["front_left__front"][
                "samples"
            ]
            for sample in samples[:4]:
                sample["warnings"] = [
                    "Frame timestamps unavailable; not synchronized."
                ]

            readiness = session.readiness()

        self.assertEqual("try_with_risk", readiness["status"])
        self.assertTrue(
            any("时间差风险" in issue for issue in readiness["quality_issues"])
        )


if __name__ == "__main__":
    unittest.main()
