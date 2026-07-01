"""Calibration sample sessions with detection and quality gates only."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import ChessboardSpec, find_chessboard
from .config import load_yaml, save_yaml
from .stitcher import save_image


SESSION_FORMAT = "DeepSharkFisheyeCalibrationSession"
SESSION_SCHEMA_VERSION = 1
CAMERA_KEYS = ("front_left", "front", "front_right")
PAIR_KEYS = (
    ("front_left", "front"),
    ("front", "front_right"),
)
DEFAULT_INTRINSIC_TARGET = 25
DEFAULT_PAIR_TARGET = 20
MINIMUM_INTRINSIC_SAMPLES = 20
MINIMUM_PAIR_SAMPLES = 15
B2_CANDIDATE_OUTPUTS = (
    "intrinsics_candidates",
    "extrinsics_candidates",
    "rms_metrics",
    "outlier_report",
    "pair_only_previews",
    "candidate_panorama_preview",
)


class BoardDefinitionNotConfirmedError(RuntimeError):
    """Raised before capture when physical board geometry is unconfirmed."""


@dataclass(frozen=True)
class BoardDetection:
    detected: bool
    detector: str
    points: np.ndarray
    point_ids: tuple[str, ...]
    annotated: np.ndarray
    error: str = ""


def supported_aruco_dictionaries() -> list[str]:
    names = (
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_6X6_50",
        "DICT_6X6_250",
        "DICT_ARUCO_ORIGINAL",
    )
    return [name for name in names if hasattr(cv2.aruco, name)]


def validate_board_definition(definition: dict[str, Any]) -> list[str]:
    board_type = str(definition.get("type", ""))
    errors: list[str] = []
    if board_type == "chessboard":
        for field in ("inner_columns", "inner_rows"):
            if int(definition.get(field, 0)) < 2:
                errors.append(f"{field} must be at least 2.")
        if float(definition.get("square_size_mm", 0)) <= 0:
            errors.append("square_size_mm must be positive.")
    elif board_type == "aruco_grid":
        for field in ("markers_x", "markers_y"):
            if int(definition.get(field, 0)) < 1:
                errors.append(f"{field} must be at least 1.")
        if float(definition.get("marker_length_mm", 0)) <= 0:
            errors.append("marker_length_mm must be positive.")
        if float(definition.get("marker_separation_mm", -1)) < 0:
            errors.append("marker_separation_mm cannot be negative.")
    elif board_type == "charuco":
        for field in ("squares_x", "squares_y"):
            if int(definition.get(field, 0)) < 2:
                errors.append(f"{field} must be at least 2.")
        square_length = float(definition.get("square_length_mm", 0))
        marker_length = float(definition.get("marker_length_mm", 0))
        if square_length <= 0:
            errors.append("square_length_mm must be positive.")
        if marker_length <= 0 or marker_length >= square_length:
            errors.append(
                "marker_length_mm must be positive and smaller than "
                "square_length_mm."
            )
    else:
        errors.append(
            "type must be chessboard, aruco_grid, or charuco."
        )
    if board_type in {"aruco_grid", "charuco"}:
        dictionary = str(definition.get("dictionary", ""))
        if dictionary not in supported_aruco_dictionaries():
            errors.append(f"Unsupported ArUco dictionary: {dictionary}")
    return errors


def _aruco_dictionary(name: str):
    dictionary_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _detect_markers(
    image: np.ndarray,
    dictionary_name: str,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    dictionary = _aruco_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    return corners, ids


def detect_board(
    image: np.ndarray,
    definition: dict[str, Any],
) -> BoardDetection:
    errors = validate_board_definition(definition)
    if errors:
        return BoardDetection(
            False,
            str(definition.get("type", "invalid")),
            np.empty((0, 2), dtype=np.float32),
            (),
            image.copy(),
            "; ".join(errors),
        )
    board_type = str(definition["type"])
    annotated = image.copy()
    if board_type == "chessboard":
        spec = ChessboardSpec(
            total_columns=int(definition["inner_columns"]) + 1,
            total_rows=int(definition["inner_rows"]) + 1,
            square_size_mm=float(definition["square_size_mm"]),
        )
        ok, corners = find_chessboard(image, spec)
        if not ok or corners is None:
            return BoardDetection(
                False,
                board_type,
                np.empty((0, 2), dtype=np.float32),
                (),
                annotated,
                "Chessboard inner corners were not detected.",
            )
        cv2.drawChessboardCorners(
            annotated,
            spec.pattern_size,
            corners,
            True,
        )
        points = corners.reshape(-1, 2).astype(np.float32)
        return BoardDetection(
            True,
            board_type,
            points,
            tuple(str(index) for index in range(len(points))),
            annotated,
        )

    dictionary_name = str(definition["dictionary"])
    marker_corners, marker_ids = _detect_markers(
        image,
        dictionary_name,
    )
    if marker_ids is None or not len(marker_ids):
        return BoardDetection(
            False,
            board_type,
            np.empty((0, 2), dtype=np.float32),
            (),
            annotated,
            "No markers from the confirmed dictionary were detected.",
        )
    cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
    if board_type == "aruco_grid":
        marker_limit = (
            int(definition["markers_x"]) * int(definition["markers_y"])
        )
        points: list[list[float]] = []
        point_ids: list[str] = []
        for marker_id, corners in zip(
            marker_ids.reshape(-1),
            marker_corners,
        ):
            if int(marker_id) >= marker_limit:
                continue
            for corner_index, point in enumerate(corners.reshape(4, 2)):
                points.append([float(point[0]), float(point[1])])
                point_ids.append(f"{int(marker_id)}:{corner_index}")
        array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return BoardDetection(
            len(array) >= 4,
            board_type,
            array,
            tuple(point_ids),
            annotated,
            "" if len(array) >= 4 else "Too few configured grid markers.",
        )

    dictionary = _aruco_dictionary(dictionary_name)
    board = cv2.aruco.CharucoBoard(
        (
            int(definition["squares_x"]),
            int(definition["squares_y"]),
        ),
        float(definition["square_length_mm"]),
        float(definition["marker_length_mm"]),
        dictionary,
    )
    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    else:
        _, charuco_corners, charuco_ids = (
            cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                image,
                board,
            )
        )
    if charuco_ids is None or charuco_corners is None:
        return BoardDetection(
            False,
            board_type,
            np.empty((0, 2), dtype=np.float32),
            (),
            annotated,
            "Charuco corners were not detected.",
        )
    cv2.aruco.drawDetectedCornersCharuco(
        annotated,
        charuco_corners,
        charuco_ids,
    )
    points = charuco_corners.reshape(-1, 2).astype(np.float32)
    return BoardDetection(
        len(points) >= 4,
        board_type,
        points,
        tuple(str(int(value)) for value in charuco_ids.reshape(-1)),
        annotated,
        "" if len(points) >= 4 else "Too few Charuco corners.",
    )


def _blur_variance(image: np.ndarray) -> float:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _detection_metrics(
    image: np.ndarray,
    detection: BoardDetection,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    points = detection.points
    if len(points) >= 3:
        hull = cv2.convexHull(points.astype(np.float32))
        area_ratio = float(cv2.contourArea(hull) / (width * height))
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        center = np.mean(points, axis=0)
        rectangle = cv2.minAreaRect(points.astype(np.float32))
        angle = float(rectangle[2])
    else:
        area_ratio = 0.0
        minimum = np.asarray([0.0, 0.0])
        maximum = np.asarray([0.0, 0.0])
        center = np.asarray([0.0, 0.0])
        angle = 0.0
    normalized_center = [
        float(center[0] / max(1, width)),
        float(center[1] / max(1, height)),
    ]
    return {
        "raw_size": [width, height],
        "point_count": int(len(points)),
        "board_area_ratio": area_ratio,
        "board_bounds": [
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        ],
        "normalized_center": normalized_center,
        "coverage_zone": _coverage_zone(normalized_center),
        "tilt_proxy_degrees": angle,
        "blur_variance": _blur_variance(image),
        "pose_signature": [
            normalized_center[0],
            normalized_center[1],
            math.sqrt(max(0.0, area_ratio)),
            angle / 180.0,
        ],
    }


def _coverage_zone(center: list[float]) -> str:
    columns = ("left", "center", "right")
    rows = ("top", "middle", "bottom")
    column = min(2, max(0, int(center[0] * 3)))
    row = min(2, max(0, int(center[1] * 3)))
    return f"{rows[row]}_{columns[column]}"


def _minimum_point_count(definition: dict[str, Any]) -> int:
    board_type = str(definition.get("type", ""))
    if board_type == "chessboard":
        return int(definition["inner_columns"]) * int(
            definition["inner_rows"]
        )
    if board_type == "charuco":
        return 6
    return 8


def _is_duplicate(
    signature: list[float],
    existing_signatures: list[list[float]],
    gate: dict[str, Any],
) -> bool:
    for existing in existing_signatures:
        center_distance = math.dist(signature[:2], existing[:2])
        area_distance = abs(signature[2] - existing[2])
        angle_distance = abs(signature[3] - existing[3])
        if (
            center_distance
            < float(gate.get("duplicate_center_distance", 0.08))
            and area_distance
            < float(gate.get("duplicate_area_distance", 0.08))
            and angle_distance
            < float(gate.get("duplicate_angle_distance", 0.06))
        ):
            return True
    return False


def _common_detection_points(
    left: BoardDetection,
    right: BoardDetection,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    left_by_id = {
        point_id: left.points[index]
        for index, point_id in enumerate(left.point_ids)
    }
    right_by_id = {
        point_id: right.points[index]
        for index, point_id in enumerate(right.point_ids)
    }
    common_ids = sorted(set(left_by_id) & set(right_by_id))
    return (
        np.asarray(
            [left_by_id[point_id] for point_id in common_ids],
            dtype=np.float32,
        ).reshape(-1, 2),
        np.asarray(
            [right_by_id[point_id] for point_id in common_ids],
            dtype=np.float32,
        ).reshape(-1, 2),
        common_ids,
    )


def default_quality_gate() -> dict[str, Any]:
    return {
        "minimum_board_area_ratio": 0.015,
        "minimum_blur_variance": 60.0,
        "duplicate_center_distance": 0.08,
        "duplicate_area_distance": 0.08,
        "duplicate_angle_distance": 0.06,
        "maximum_pair_time_delta_seconds": 0.1,
    }


class CalibrationSession:
    """Persistent sample index; it never reads or writes formal calibration."""

    def __init__(self, directory: str | Path, data: dict[str, Any]):
        self.directory = Path(directory)
        self.data = data

    @classmethod
    def create(
        cls,
        sessions_root: str | Path,
        topology: str,
        resolution: tuple[int, int],
        source_coordinate_space: str,
        source_reference_size: list[int],
        source_contract_version: int | None,
        board_definition: dict[str, Any],
        board_confirmed: bool,
        quality_gate: dict[str, Any] | None = None,
        captured_at: datetime | None = None,
    ) -> "CalibrationSession":
        errors = validate_board_definition(board_definition)
        if errors:
            raise ValueError("; ".join(errors))
        if source_coordinate_space != "raw_frame_pixels":
            raise ValueError(
                "Calibration sessions require raw_frame_pixels source "
                "coordinates."
            )
        if list(source_reference_size) != [
            int(resolution[0]),
            int(resolution[1]),
        ]:
            raise ValueError(
                "Session resolution must match source_reference_size."
            )
        captured_at = captured_at or datetime.now().astimezone()
        name = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        root = Path(sessions_root)
        directory = root / name
        suffix = 1
        while directory.exists():
            directory = root / f"{name}_{suffix:02d}"
            suffix += 1
        directory.mkdir(parents=True, exist_ok=False)
        for camera in CAMERA_KEYS:
            (directory / "intrinsics" / camera).mkdir(parents=True)
        for left, right in PAIR_KEYS:
            (directory / "stereo_pairs" / f"{left}__{right}").mkdir(
                parents=True
            )
        data = {
            "format": SESSION_FORMAT,
            "schema_version": SESSION_SCHEMA_VERSION,
            "created_at": captured_at.isoformat(timespec="milliseconds"),
            "topology": topology,
            "resolution": [int(resolution[0]), int(resolution[1])],
            "source_coordinate_space": source_coordinate_space,
            "source_reference_size": list(source_reference_size),
            "source_contract_version": source_contract_version,
            "board_definition": deepcopy(board_definition),
            "board_confirmed": bool(board_confirmed),
            "targets": {
                "intrinsic_samples_per_camera": DEFAULT_INTRINSIC_TARGET,
                "pair_samples_per_pair": DEFAULT_PAIR_TARGET,
            },
            "quality_gate": {
                **default_quality_gate(),
                **(quality_gate or {}),
            },
            "intrinsics": {
                camera: {"samples": []}
                for camera in CAMERA_KEYS
            },
            "stereo_pairs": {
                f"{left}__{right}": {"samples": []}
                for left, right in PAIR_KEYS
            },
            "formal_profile_modified": False,
        }
        session = cls(directory, data)
        session._save()
        return session

    @classmethod
    def load(cls, directory: str | Path) -> "CalibrationSession":
        path = Path(directory)
        data = load_yaml(path / "session.yaml")
        if data.get("format") != SESSION_FORMAT:
            raise ValueError("Not a DeepShark calibration session.")
        return cls(path, data)

    def _save(self) -> None:
        save_yaml(self.directory / "session.yaml", self.data)

    def _require_confirmed_board(self) -> None:
        if not self.data.get("board_confirmed"):
            raise BoardDefinitionNotConfirmedError(
                "Confirm the physical board definition before capture."
            )

    def _accepted_signatures(
        self,
        group: dict[str, Any],
        metric_key: str = "metrics",
    ) -> list[list[float]]:
        return [
            sample[metric_key]["pose_signature"]
            for sample in group.get("samples", [])
            if sample.get("accepted") and metric_key in sample
        ]

    @staticmethod
    def _unique_sample_id(
        group: dict[str, Any],
        captured_at: datetime,
    ) -> str:
        base = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        existing = {
            str(sample.get("sample_id", ""))
            for sample in group.get("samples", [])
        }
        candidate = base
        suffix = 1
        while candidate in existing:
            candidate = f"{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def capture_intrinsic(
        self,
        camera_key: str,
        frame: np.ndarray,
        frame_timestamp: float | None = None,
        captured_at: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_confirmed_board()
        if camera_key not in CAMERA_KEYS:
            raise ValueError(f"Unsupported session camera: {camera_key}")
        expected_size = self.data["resolution"]
        actual_size = [int(frame.shape[1]), int(frame.shape[0])]
        detection = detect_board(frame, self.data["board_definition"])
        metrics = _detection_metrics(frame, detection)
        reasons: list[str] = []
        if actual_size != expected_size:
            reasons.append(
                f"Raw size {actual_size} does not match session "
                f"resolution {expected_size}."
            )
        if (
            not detection.detected
            or len(detection.points)
            < _minimum_point_count(self.data["board_definition"])
        ):
            reasons.append(
                detection.error
                or "Insufficient confirmed board corners."
            )
        gate = self.data["quality_gate"]
        if (
            metrics["board_area_ratio"]
            < float(gate["minimum_board_area_ratio"])
        ):
            reasons.append("Board area is too small.")
        if metrics["blur_variance"] < float(gate["minimum_blur_variance"]):
            reasons.append("Image is severely blurred.")
        group = self.data["intrinsics"][camera_key]
        if detection.detected and _is_duplicate(
            metrics["pose_signature"],
            self._accepted_signatures(group),
            gate,
        ):
            reasons.append("Pose and position are too similar to an accepted sample.")
        captured_at = captured_at or datetime.now().astimezone()
        record = {
            "sample_id": self._unique_sample_id(group, captured_at),
            "captured_at": captured_at.isoformat(timespec="milliseconds"),
            "frame_timestamp": frame_timestamp,
            "camera": camera_key,
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "warnings": [],
            "detector": detection.detector,
            "point_ids": list(detection.point_ids),
            "image_points": detection.points.tolist(),
            "metrics": metrics,
        }
        self._persist_intrinsic(camera_key, frame, detection.annotated, record)
        group["samples"].append(record)
        self._save()
        return deepcopy(record)

    def _persist_intrinsic(
        self,
        camera_key: str,
        frame: np.ndarray,
        annotated: np.ndarray,
        record: dict[str, Any],
    ) -> None:
        state = "accepted" if record["accepted"] else "rejected"
        directory = self.directory / "intrinsics" / camera_key / state
        directory.mkdir(parents=True, exist_ok=True)
        sample_id = record["sample_id"]
        raw_path = directory / f"{sample_id}_raw.png"
        detection_path = directory / f"{sample_id}_detection.png"
        metadata_path = directory / f"{sample_id}_metadata.yaml"
        save_image(raw_path, frame)
        save_image(detection_path, annotated)
        record["files"] = {
            "raw": raw_path.relative_to(self.directory).as_posix(),
            "detection": detection_path.relative_to(self.directory).as_posix(),
            "metadata": metadata_path.relative_to(self.directory).as_posix(),
        }
        save_yaml(metadata_path, record)

    def capture_pair(
        self,
        left_camera: str,
        right_camera: str,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_timestamp: float | None,
        right_timestamp: float | None,
        physical_board_id: str,
        physical_board_confirmed: bool,
        captured_at: datetime | None = None,
    ) -> dict[str, Any]:
        self._require_confirmed_board()
        pair_key = f"{left_camera}__{right_camera}"
        if pair_key not in self.data["stereo_pairs"]:
            raise ValueError(f"Unsupported session pair: {pair_key}")
        if not physical_board_confirmed or not physical_board_id.strip():
            raise BoardDefinitionNotConfirmedError(
                "Confirm that both images show the same physical board."
            )
        left_detection = detect_board(
            left_frame,
            self.data["board_definition"],
        )
        right_detection = detect_board(
            right_frame,
            self.data["board_definition"],
        )
        left_common, right_common, common_ids = _common_detection_points(
            left_detection,
            right_detection,
        )
        left_metrics = _detection_metrics(left_frame, left_detection)
        right_metrics = _detection_metrics(right_frame, right_detection)
        reasons: list[str] = []
        warnings: list[str] = []
        expected_size = self.data["resolution"]
        for camera, frame in (
            (left_camera, left_frame),
            (right_camera, right_frame),
        ):
            actual_size = [int(frame.shape[1]), int(frame.shape[0])]
            if actual_size != expected_size:
                reasons.append(
                    f"{camera} raw size {actual_size} does not match "
                    f"session resolution {expected_size}."
                )
        minimum_points = _minimum_point_count(self.data["board_definition"])
        if not left_detection.detected:
            reasons.append(f"{left_camera}: {left_detection.error}")
        if not right_detection.detected:
            reasons.append(f"{right_camera}: {right_detection.error}")
        if len(common_ids) < minimum_points:
            reasons.append(
                f"Only {len(common_ids)} common confirmed board points; "
                f"need at least {minimum_points}."
            )
        gate = self.data["quality_gate"]
        for camera, metrics in (
            (left_camera, left_metrics),
            (right_camera, right_metrics),
        ):
            if (
                metrics["board_area_ratio"]
                < float(gate["minimum_board_area_ratio"])
            ):
                reasons.append(f"{camera}: board area is too small.")
            if (
                metrics["blur_variance"]
                < float(gate["minimum_blur_variance"])
            ):
                reasons.append(f"{camera}: image is severely blurred.")
        time_delta = None
        synchronization = "unknown"
        if left_timestamp is not None and right_timestamp is not None:
            time_delta = abs(float(left_timestamp) - float(right_timestamp))
            synchronization = "software_timestamp_only"
            warnings.append(
                "Software frame timestamps are not hardware synchronization."
            )
            if time_delta > float(
                gate["maximum_pair_time_delta_seconds"]
            ):
                reasons.append(
                    f"Frame time delta {time_delta:.3f}s exceeds "
                    f"{gate['maximum_pair_time_delta_seconds']:.3f}s."
                )
        else:
            warnings.append(
                "Frame timestamps unavailable; this sample is not verified "
                "as synchronized."
            )
        pair_signature = (
            left_metrics["pose_signature"]
            + right_metrics["pose_signature"]
        )
        group = self.data["stereo_pairs"][pair_key]
        existing = [
            sample["pair_pose_signature"]
            for sample in group["samples"]
            if sample.get("accepted")
        ]
        if existing and any(
            _is_duplicate(
                pair_signature[:4],
                [signature[:4]],
                gate,
            )
            and _is_duplicate(
                pair_signature[4:],
                [signature[4:]],
                gate,
            )
            for signature in existing
        ):
            reasons.append("Pair pose and position duplicate an accepted sample.")
        captured_at = captured_at or datetime.now().astimezone()
        position_value = (
            left_metrics["normalized_center"][0]
            + right_metrics["normalized_center"][0]
        ) / 2.0
        position_bucket = ("left", "center", "right")[
            min(2, max(0, int(position_value * 3)))
        ]
        mean_area = (
            left_metrics["board_area_ratio"]
            + right_metrics["board_area_ratio"]
        ) / 2.0
        distance_bucket = (
            "far" if mean_area < 0.04 else "middle" if mean_area < 0.12 else "near"
        )
        record = {
            "sample_id": self._unique_sample_id(group, captured_at),
            "captured_at": captured_at.isoformat(timespec="milliseconds"),
            "pair": [left_camera, right_camera],
            "physical_board_id": physical_board_id.strip(),
            "physical_board_confirmed": True,
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "warnings": warnings,
            "synchronization": synchronization,
            "left_frame_timestamp": left_timestamp,
            "right_frame_timestamp": right_timestamp,
            "frame_time_delta_seconds": time_delta,
            "detector": self.data["board_definition"]["type"],
            "common_point_ids": common_ids,
            "left_common_points": left_common.tolist(),
            "right_common_points": right_common.tolist(),
            "left_metrics": left_metrics,
            "right_metrics": right_metrics,
            "pair_pose_signature": pair_signature,
            "pair_coverage_zone": f"{position_bucket}_{distance_bucket}",
        }
        self._persist_pair(
            pair_key,
            left_frame,
            right_frame,
            left_detection.annotated,
            right_detection.annotated,
            record,
        )
        group["samples"].append(record)
        self._save()
        return deepcopy(record)

    def inspect_pair(
        self,
        left_camera: str,
        right_camera: str,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_timestamp: float | None,
        right_timestamp: float | None,
    ) -> dict[str, Any]:
        """Detect a pair without saving images or mutating the session."""
        self._require_confirmed_board()
        pair_key = f"{left_camera}__{right_camera}"
        if pair_key not in self.data["stereo_pairs"]:
            raise ValueError(f"Unsupported session pair: {pair_key}")
        left_detection = detect_board(
            left_frame,
            self.data["board_definition"],
        )
        right_detection = detect_board(
            right_frame,
            self.data["board_definition"],
        )
        _, _, common_ids = _common_detection_points(
            left_detection,
            right_detection,
        )
        minimum_points = _minimum_point_count(self.data["board_definition"])
        time_delta = None
        time_risk = "时间戳不可用，不能确认同步。"
        if left_timestamp is not None and right_timestamp is not None:
            time_delta = abs(float(left_timestamp) - float(right_timestamp))
            threshold = float(
                self.data["quality_gate"][
                    "maximum_pair_time_delta_seconds"
                ]
            )
            time_risk = (
                f"软件时间差 {time_delta * 1000.0:.1f} ms"
                + (
                    "，超过门限。"
                    if time_delta > threshold
                    else "，在门限内；仍不是硬件同步。"
                )
            )
        return {
            "pair": [left_camera, right_camera],
            "left_detected": left_detection.detected,
            "right_detected": right_detection.detected,
            "left_point_count": int(len(left_detection.points)),
            "right_point_count": int(len(right_detection.points)),
            "common_point_count": len(common_ids),
            "minimum_common_points": minimum_points,
            "common_points_sufficient": len(common_ids) >= minimum_points,
            "frame_time_delta_seconds": time_delta,
            "time_status": time_risk,
            "left_error": left_detection.error,
            "right_error": right_detection.error,
        }

    def _persist_pair(
        self,
        pair_key: str,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        left_annotated: np.ndarray,
        right_annotated: np.ndarray,
        record: dict[str, Any],
    ) -> None:
        state = "accepted" if record["accepted"] else "rejected"
        directory = (
            self.directory / "stereo_pairs" / pair_key / state
        )
        directory.mkdir(parents=True, exist_ok=True)
        sample_id = record["sample_id"]
        paths = {
            "left_raw": directory / f"{sample_id}_left_raw.png",
            "right_raw": directory / f"{sample_id}_right_raw.png",
            "left_detection": directory / f"{sample_id}_left_detection.png",
            "right_detection": directory / f"{sample_id}_right_detection.png",
            "metadata": directory / f"{sample_id}_metadata.yaml",
        }
        save_image(paths["left_raw"], left_frame)
        save_image(paths["right_raw"], right_frame)
        save_image(paths["left_detection"], left_annotated)
        save_image(paths["right_detection"], right_annotated)
        record["files"] = {
            key: path.relative_to(self.directory).as_posix()
            for key, path in paths.items()
        }
        save_yaml(paths["metadata"], record)

    def summary(self) -> dict[str, Any]:
        all_mono_zones = {
            f"{row}_{column}"
            for row in ("top", "middle", "bottom")
            for column in ("left", "center", "right")
        }
        all_pair_zones = {
            f"{position}_{distance}"
            for position in ("left", "center", "right")
            for distance in ("far", "middle", "near")
        }
        intrinsics = {}
        for camera, group in self.data["intrinsics"].items():
            accepted = [
                sample for sample in group["samples"] if sample["accepted"]
            ]
            observed = {
                sample["metrics"]["coverage_zone"]
                for sample in accepted
            }
            intrinsics[camera] = {
                "accepted": len(accepted),
                "rejected": len(group["samples"]) - len(accepted),
                "target": self.data["targets"][
                    "intrinsic_samples_per_camera"
                ],
                "missing_coverage_zones": sorted(
                    all_mono_zones - observed
                ),
                "last_rejection_reason": next(
                    (
                        "; ".join(sample["rejection_reasons"])
                        for sample in reversed(group["samples"])
                        if not sample["accepted"]
                    ),
                    "",
                ),
            }
        pairs = {}
        for pair_key, group in self.data["stereo_pairs"].items():
            accepted = [
                sample for sample in group["samples"] if sample["accepted"]
            ]
            observed = {
                sample["pair_coverage_zone"]
                for sample in accepted
            }
            pairs[pair_key] = {
                "accepted": len(accepted),
                "rejected": len(group["samples"]) - len(accepted),
                "target": self.data["targets"]["pair_samples_per_pair"],
                "missing_coverage_zones": sorted(
                    all_pair_zones - observed
                ),
                "last_rejection_reason": next(
                    (
                        "; ".join(sample["rejection_reasons"])
                        for sample in reversed(group["samples"])
                        if not sample["accepted"]
                    ),
                    "",
                ),
            }
        return {"intrinsics": intrinsics, "stereo_pairs": pairs}

    def readiness(self) -> dict[str, Any]:
        """Return a read-only recommendation for a future B-2 solve."""
        summary = self.summary()
        blocking: list[str] = []
        quality_issues: list[str] = []
        if not self.data.get("board_confirmed"):
            blocking.append("标定板参数尚未确认。")
        if self.data.get("resolution") != [1920, 1080]:
            blocking.append("Session 分辨率不是 1920×1080。")
        for camera in CAMERA_KEYS:
            item = summary["intrinsics"][camera]
            if item["accepted"] < MINIMUM_INTRINSIC_SAMPLES:
                blocking.append(
                    f"{camera} 仅有 {item['accepted']} 张合格样本，"
                    f"最低需要 {MINIMUM_INTRINSIC_SAMPLES} 张。"
                )
            if len(item["missing_coverage_zones"]) > 4:
                quality_issues.append(
                    f"{camera} 的中心、边缘或四角覆盖仍明显不足。"
                )
            group = self.data["intrinsics"][camera]
            duplicate_count = _reason_count(
                group.get("samples", []),
                ("too similar", "duplicate"),
            )
            if duplicate_count > max(3, item["accepted"] // 4):
                quality_issues.append(
                    f"{camera} 的重复姿态拒绝过多（{duplicate_count} 次）。"
                )
        for left, right in PAIR_KEYS:
            pair_key = f"{left}__{right}"
            item = summary["stereo_pairs"][pair_key]
            if item["accepted"] < MINIMUM_PAIR_SAMPLES:
                blocking.append(
                    f"{left} ↔ {right} 仅有 {item['accepted']} 组合格样本，"
                    f"最低需要 {MINIMUM_PAIR_SAMPLES} 组。"
                )
            if len(item["missing_coverage_zones"]) > 5:
                quality_issues.append(
                    f"{left} ↔ {right} 的位置或距离覆盖仍明显不足。"
                )
            samples = self.data["stereo_pairs"][pair_key].get("samples", [])
            time_risk_count = _reason_count(
                samples,
                ("time delta", "timestamps unavailable"),
                include_warnings=True,
            )
            if time_risk_count > max(2, item["accepted"] // 5):
                quality_issues.append(
                    f"{left} ↔ {right} 的时间差风险样本过多"
                    f"（{time_risk_count} 组）。"
                )
            duplicate_count = _reason_count(
                samples,
                ("duplicate",),
            )
            if duplicate_count > max(3, item["accepted"] // 4):
                quality_issues.append(
                    f"{left} ↔ {right} 的重复姿态拒绝过多"
                    f"（{duplicate_count} 次）。"
                )

        minimum_counts_met = not any(
            "最低需要" in reason for reason in blocking
        )
        prerequisites_met = (
            bool(self.data.get("board_confirmed"))
            and self.data.get("resolution") == [1920, 1080]
        )
        near_minimum = prerequisites_met and all(
            summary["intrinsics"][camera]["accepted"] >= 15
            for camera in CAMERA_KEYS
        ) and all(
            summary["stereo_pairs"][f"{left}__{right}"]["accepted"] >= 10
            for left, right in PAIR_KEYS
        )
        if prerequisites_met and minimum_counts_met and not quality_issues:
            status = "ready"
            label = "可进入 B-2 求解"
        elif prerequisites_met and (minimum_counts_met or near_minimum):
            status = "try_with_risk"
            label = "可尝试，但样本质量不足"
        else:
            status = "not_recommended"
            label = "暂不建议求解"
        return {
            "status": status,
            "label": label,
            "blocking_reasons": blocking,
            "quality_issues": quality_issues,
            "minimum_counts_met": minimum_counts_met,
            "prerequisites_met": prerequisites_met,
        }

    def b2_candidate_access(self) -> dict[str, Any]:
        """Describe B-2 entry permissions without running or applying a solve."""
        readiness = self.readiness()
        status = readiness["status"]
        if status == "ready":
            mode = "official"
            can_enter = True
            experimental = False
            report_only = False
            apply_allowed = True
        elif status == "try_with_risk":
            mode = "experimental"
            can_enter = True
            experimental = True
            report_only = True
            apply_allowed = False
        else:
            mode = "blocked"
            can_enter = False
            experimental = False
            report_only = True
            apply_allowed = False
        return {
            "mode": mode,
            "can_enter": can_enter,
            "experimental": experimental,
            "report_only": report_only,
            "apply_allowed": apply_allowed,
            "expected_outputs": list(B2_CANDIDATE_OUTPUTS),
            "readiness": readiness,
        }


def _reason_count(
    samples: list[dict[str, Any]],
    fragments: tuple[str, ...],
    include_warnings: bool = False,
) -> int:
    count = 0
    for sample in samples:
        messages = list(sample.get("rejection_reasons", []))
        if include_warnings:
            messages.extend(sample.get("warnings", []))
        text = " ".join(str(message).lower() for message in messages)
        if any(fragment.lower() in text for fragment in fragments):
            count += 1
    return count
