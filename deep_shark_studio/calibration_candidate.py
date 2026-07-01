"""Report-only fisheye calibration candidates built from sample sessions."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration_session import (
    CAMERA_KEYS,
    PAIR_KEYS,
    CalibrationSession,
)
from .config import CONFIG_DIR, PROJECT_ROOT, file_revision, load_yaml, save_yaml
from .stitcher import save_image


CANDIDATE_FORMAT = "DeepSharkFisheyeCalibrationCandidate"
CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_RESOLUTION = (1920, 1080)
PANORAMA_SIZE = (1800, 600)
PANORAMA_YAW_RANGE_DEG = (-135.0, 135.0)
PANORAMA_PITCH_RANGE_DEG = (-45.0, 45.0)


class CandidateSolveError(RuntimeError):
    """Raised when a session is not eligible for candidate solving."""


def _object_point_lookup(definition: dict[str, Any]) -> dict[str, np.ndarray]:
    board_type = str(definition["type"])
    if board_type == "chessboard":
        columns = int(definition["inner_columns"])
        rows = int(definition["inner_rows"])
        square = float(definition["square_size_mm"])
        return {
            str(row * columns + column): np.asarray(
                [column * square, row * square, 0.0],
                dtype=np.float64,
            )
            for row in range(rows)
            for column in range(columns)
        }

    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, str(definition["dictionary"]))
    )
    if board_type == "aruco_grid":
        board = cv2.aruco.GridBoard(
            (
                int(definition["markers_x"]),
                int(definition["markers_y"]),
            ),
            float(definition["marker_length_mm"]),
            float(definition["marker_separation_mm"]),
            dictionary,
        )
        lookup: dict[str, np.ndarray] = {}
        for marker_id, corners in zip(
            board.getIds().reshape(-1),
            board.getObjPoints(),
        ):
            for corner_index, point in enumerate(
                np.asarray(corners).reshape(4, 3)
            ):
                lookup[f"{int(marker_id)}:{corner_index}"] = np.asarray(
                    point,
                    dtype=np.float64,
                )
        return lookup

    if board_type == "charuco":
        board = cv2.aruco.CharucoBoard(
            (
                int(definition["squares_x"]),
                int(definition["squares_y"]),
            ),
            float(definition["square_length_mm"]),
            float(definition["marker_length_mm"]),
            dictionary,
        )
        return {
            str(index): np.asarray(point, dtype=np.float64)
            for index, point in enumerate(board.getChessboardCorners())
        }
    raise ValueError(f"Unsupported board type: {board_type}")


def _object_points(
    point_ids: list[str],
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    missing = [point_id for point_id in point_ids if point_id not in lookup]
    if missing:
        raise ValueError(
            f"Unknown board point IDs: {', '.join(missing[:5])}"
        )
    return np.asarray(
        [lookup[point_id] for point_id in point_ids],
        dtype=np.float64,
    ).reshape(-1, 1, 3)


def _image_points(values: list[list[float]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1, 1, 2)


def _robust_outlier_indices(
    errors: list[float],
    minimum_error: float,
    maximum_fraction: float = 0.25,
) -> list[int]:
    if len(errors) < 8:
        return []
    values = np.asarray(errors, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(
        minimum_error,
        median * 1.8,
        median + 3.0 * 1.4826 * mad,
    )
    candidates = [
        index for index, value in enumerate(values) if value > threshold
    ]
    limit = max(1, int(math.floor(len(values) * maximum_fraction)))
    return sorted(
        candidates,
        key=lambda index: errors[index],
        reverse=True,
    )[:limit]


def _rejection_reason_counts(
    samples: list[dict[str, Any]],
) -> dict[str, int]:
    messages = (
        reason
        for sample in samples
        if not sample.get("accepted")
        for reason in sample.get("rejection_reasons", [])
    )
    return dict(Counter(str(message) for message in messages))


def _calibrate_intrinsics_once(
    records: list[dict[str, Any]],
    indices: list[int],
    lookup: dict[str, np.ndarray],
    resolution: tuple[int, int],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[float],
]:
    object_points = [
        _object_points(records[index]["point_ids"], lookup)
        for index in indices
    ]
    image_points = [
        _image_points(records[index]["image_points"])
        for index in indices
    ]
    width, height = resolution
    camera_matrix = np.asarray(
        [
            [width * 0.5, 0.0, width * 0.5],
            [0.0, width * 0.5, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    rms, camera_matrix, distortion, rvecs, tvecs = (
        cv2.fisheye.calibrate(
            object_points,
            image_points,
            resolution,
            camera_matrix,
            distortion,
            None,
            None,
            flags=flags,
            criteria=(
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_MAX_ITER,
                200,
                1e-7,
            ),
        )
    )
    per_view_errors: list[float] = []
    for object_array, image_array, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.fisheye.projectPoints(
            object_array,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        per_view_errors.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected - image_array) ** 2,
                            axis=2,
                        )
                    )
                )
            )
        )
    return (
        float(rms),
        camera_matrix,
        distortion,
        list(rvecs),
        list(tvecs),
        per_view_errors,
    )


def _solve_intrinsics(
    session: CalibrationSession,
    camera_key: str,
    lookup: dict[str, np.ndarray],
    resolution: tuple[int, int],
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    all_samples = session.data["intrinsics"][camera_key]["samples"]
    accepted = [sample for sample in all_samples if sample.get("accepted")]
    result: dict[str, Any] = {
        "camera": camera_key,
        "status": "failed",
        "model": "opencv_fisheye",
        "resolution": list(resolution),
        "accepted_input_count": len(accepted),
        "used_sample_count": 0,
        "outlier_count": 0,
        "outliers": [],
        "per_view_errors_px": [],
        "capture_rejection_count": len(all_samples) - len(accepted),
        "capture_rejection_reasons": _rejection_reason_counts(all_samples),
        "rms_px": None,
        "camera_matrix": None,
        "distortion_coefficients": None,
        "error": "",
    }
    if len(accepted) < 8:
        result["error"] = "At least 8 accepted intrinsic samples are required."
        return result, None
    indices = list(range(len(accepted)))
    try:
        initial = _calibrate_intrinsics_once(
            accepted,
            indices,
            lookup,
            resolution,
        )
        outlier_positions = _robust_outlier_indices(
            initial[-1],
            minimum_error=1.25,
        )
        if outlier_positions and len(indices) - len(outlier_positions) >= 8:
            outlier_set = set(outlier_positions)
            filtered_indices = [
                index for index in indices if index not in outlier_set
            ]
            solved = _calibrate_intrinsics_once(
                accepted,
                filtered_indices,
                lookup,
                resolution,
            )
        else:
            filtered_indices = indices
            outlier_positions = []
            solved = initial
        rms, camera_matrix, distortion, _, _, errors = solved
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result, None

    error_by_index = {
        index: error for index, error in zip(filtered_indices, errors)
    }
    initial_errors = initial[-1]
    result.update(
        {
            "status": "success",
            "used_sample_count": len(filtered_indices),
            "outlier_count": len(outlier_positions),
            "outliers": [
                {
                    "sample_id": accepted[index].get("sample_id", ""),
                    "reprojection_error_px": initial_errors[index],
                    "reason": "robust reprojection-error outlier",
                }
                for index in outlier_positions
            ],
            "per_view_errors_px": [
                {
                    "sample_id": accepted[index].get("sample_id", ""),
                    "error_px": error_by_index[index],
                }
                for index in filtered_indices
            ],
            "rms_px": rms,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": distortion.reshape(-1).tolist(),
        }
    )
    return result, (camera_matrix, distortion)


def _solve_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    undistorted = cv2.fisheye.undistortPoints(
        image_points,
        camera_matrix,
        distortion,
        P=camera_matrix,
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        undistorted,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for accepted pair sample.")
    return cv2.Rodrigues(rvec)[0], tvec


def _pair_pose_outliers(
    records: list[dict[str, Any]],
    lookup: dict[str, np.ndarray],
    left_calibration: tuple[np.ndarray, np.ndarray],
    right_calibration: tuple[np.ndarray, np.ndarray],
) -> tuple[list[int], list[dict[str, Any]]]:
    observations: list[np.ndarray] = []
    valid_indices: list[int] = []
    failures: list[dict[str, Any]] = []
    left_matrix, left_distortion = left_calibration
    right_matrix, right_distortion = right_calibration
    for index, record in enumerate(records):
        try:
            object_points = _object_points(
                record["common_point_ids"],
                lookup,
            )
            left_rotation, left_translation = _solve_board_pose(
                object_points,
                _image_points(record["left_common_points"]),
                left_matrix,
                left_distortion,
            )
            right_rotation, right_translation = _solve_board_pose(
                object_points,
                _image_points(record["right_common_points"]),
                right_matrix,
                right_distortion,
            )
            relative_rotation = right_rotation @ left_rotation.T
            relative_translation = (
                right_translation
                - relative_rotation @ left_translation
            )
            rotation_vector = cv2.Rodrigues(relative_rotation)[0].reshape(-1)
            observations.append(
                np.concatenate(
                    (rotation_vector, relative_translation.reshape(-1))
                )
            )
            valid_indices.append(index)
        except Exception as exc:
            failures.append(
                {
                    "sample_id": record.get("sample_id", ""),
                    "reason": f"pair pose check failed: {exc}",
                }
            )
    if len(observations) < 8:
        return valid_indices, failures
    values = np.asarray(observations, dtype=np.float64)
    median = np.median(values, axis=0)
    scale = np.median(np.abs(values - median), axis=0) * 1.4826
    scale = np.maximum(
        scale,
        np.asarray([0.01, 0.01, 0.01, 10.0, 10.0, 10.0]),
    )
    scores = np.sqrt(np.mean(((values - median) / scale) ** 2, axis=1))
    outlier_positions = [
        position for position, score in enumerate(scores) if score > 4.0
    ]
    maximum_outliers = max(1, int(math.floor(len(values) * 0.25)))
    outlier_positions = sorted(
        outlier_positions,
        key=lambda position: scores[position],
        reverse=True,
    )[:maximum_outliers]
    outlier_indices = {
        valid_indices[position] for position in outlier_positions
    }
    kept = [index for index in valid_indices if index not in outlier_indices]
    failures.extend(
        {
            "sample_id": records[index].get("sample_id", ""),
            "reason": "robust relative-pose outlier",
            "pose_consistency_score": float(
                scores[valid_indices.index(index)]
            ),
        }
        for index in sorted(outlier_indices)
    )
    return kept, failures


def _stereo_calibrate_once(
    records: list[dict[str, Any]],
    indices: list[int],
    lookup: dict[str, np.ndarray],
    left_calibration: tuple[np.ndarray, np.ndarray],
    right_calibration: tuple[np.ndarray, np.ndarray],
    resolution: tuple[int, int],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    list[float],
]:
    object_points = [
        _object_points(records[index]["common_point_ids"], lookup)
        for index in indices
    ]
    left_points = [
        _image_points(records[index]["left_common_points"])
        for index in indices
    ]
    right_points = [
        _image_points(records[index]["right_common_points"])
        for index in indices
    ]
    left_matrix, left_distortion = left_calibration
    right_matrix, right_distortion = right_calibration
    output = cv2.fisheye.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_matrix.copy(),
        left_distortion.copy(),
        right_matrix.copy(),
        right_distortion.copy(),
        resolution,
        flags=cv2.fisheye.CALIB_FIX_INTRINSIC,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            200,
            1e-7,
        ),
    )
    rms = float(output[0])
    rotation = np.asarray(output[5], dtype=np.float64)
    translation = np.asarray(output[6], dtype=np.float64).reshape(3, 1)
    per_view_errors: list[float] = []
    if len(output) >= 9:
        for (
            object_array,
            left_array,
            right_array,
            left_rvec,
            left_tvec,
        ) in zip(
            object_points,
            left_points,
            right_points,
            output[7],
            output[8],
        ):
            projected_left, _ = cv2.fisheye.projectPoints(
                object_array,
                left_rvec,
                left_tvec,
                left_matrix,
                left_distortion,
            )
            left_rotation = cv2.Rodrigues(left_rvec)[0]
            right_board_rotation = rotation @ left_rotation
            right_board_translation = (
                rotation @ np.asarray(left_tvec).reshape(3, 1)
                + translation
            )
            right_rvec = cv2.Rodrigues(right_board_rotation)[0]
            projected_right, _ = cv2.fisheye.projectPoints(
                object_array,
                right_rvec,
                right_board_translation,
                right_matrix,
                right_distortion,
            )
            squared = np.concatenate(
                (
                    np.sum((projected_left - left_array) ** 2, axis=2),
                    np.sum((projected_right - right_array) ** 2, axis=2),
                )
            )
            per_view_errors.append(float(np.sqrt(np.mean(squared))))
    return rms, rotation, translation, per_view_errors


def _time_delta_statistics(
    records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    values = [
        float(record["frame_time_delta_seconds"])
        for record in records
        if record.get("frame_time_delta_seconds") is not None
    ]
    return {
        "timestamp_model": "software_timestamp_only",
        "known_count": len(values),
        "missing_count": len(records) - len(values),
        "risk_count": sum(value > threshold for value in values),
        "threshold_seconds": threshold,
        "minimum_seconds": min(values) if values else None,
        "mean_seconds": float(np.mean(values)) if values else None,
        "p95_seconds": (
            float(np.percentile(values, 95)) if values else None
        ),
        "maximum_seconds": max(values) if values else None,
        "warning": (
            "Software timestamps are not hardware synchronization; "
            "motion can bias stereo extrinsics."
        ),
    }


def _solve_stereo_pair(
    session: CalibrationSession,
    left_camera: str,
    right_camera: str,
    lookup: dict[str, np.ndarray],
    calibrations: dict[str, tuple[np.ndarray, np.ndarray] | None],
    resolution: tuple[int, int],
) -> tuple[
    dict[str, Any],
    tuple[np.ndarray, np.ndarray] | None,
]:
    pair_key = f"{left_camera}__{right_camera}"
    all_samples = session.data["stereo_pairs"][pair_key]["samples"]
    accepted = [sample for sample in all_samples if sample.get("accepted")]
    time_statistics = _time_delta_statistics(
        accepted,
        float(
            session.data["quality_gate"][
                "maximum_pair_time_delta_seconds"
            ]
        ),
    )
    result: dict[str, Any] = {
        "pair": [left_camera, right_camera],
        "status": "failed",
        "model": "opencv_fisheye_stereo",
        "resolution": list(resolution),
        "accepted_input_count": len(accepted),
        "used_sample_count": 0,
        "outlier_count": 0,
        "outliers": [],
        "per_view_errors_px": [],
        "capture_rejection_count": len(all_samples) - len(accepted),
        "capture_rejection_reasons": _rejection_reason_counts(all_samples),
        "time_delta_statistics": time_statistics,
        "rms_px": None,
        "rotation_left_to_right": None,
        "translation_left_to_right_mm": None,
        "error": "",
    }
    left_calibration = calibrations.get(left_camera)
    right_calibration = calibrations.get(right_camera)
    if left_calibration is None or right_calibration is None:
        result["error"] = "Candidate intrinsics are unavailable for this pair."
        return result, None
    if len(accepted) < 8:
        result["error"] = "At least 8 accepted pair samples are required."
        return result, None
    indices, outliers = _pair_pose_outliers(
        accepted,
        lookup,
        left_calibration,
        right_calibration,
    )
    if len(indices) < 8:
        result["error"] = "Too few pair samples remain after pose validation."
        result["outliers"] = outliers
        result["outlier_count"] = len(outliers)
        return result, None
    try:
        initial = _stereo_calibrate_once(
            accepted,
            indices,
            lookup,
            left_calibration,
            right_calibration,
            resolution,
        )
        error_positions = _robust_outlier_indices(
            initial[-1],
            minimum_error=2.0,
        )
        if error_positions and len(indices) - len(error_positions) >= 8:
            rejected_indices = {indices[position] for position in error_positions}
            outliers.extend(
                {
                    "sample_id": accepted[index].get("sample_id", ""),
                    "reason": "robust stereo reprojection-error outlier",
                    "reprojection_error_px": initial[-1][
                        indices.index(index)
                    ],
                }
                for index in sorted(rejected_indices)
            )
            indices = [
                index for index in indices if index not in rejected_indices
            ]
            solved = _stereo_calibrate_once(
                accepted,
                indices,
                lookup,
                left_calibration,
                right_calibration,
                resolution,
            )
        else:
            solved = initial
        rms, rotation, translation, errors = solved
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["outliers"] = outliers
        result["outlier_count"] = len(outliers)
        return result, None

    result.update(
        {
            "status": "success",
            "used_sample_count": len(indices),
            "outlier_count": len(outliers),
            "outliers": outliers,
            "per_view_errors_px": [
                {
                    "sample_id": accepted[index].get("sample_id", ""),
                    "error_px": error,
                }
                for index, error in zip(indices, errors)
            ],
            "rms_px": rms,
            "rotation_left_to_right": rotation.tolist(),
            "translation_left_to_right_mm": translation.reshape(-1).tolist(),
        }
    )
    return result, (rotation, translation)


def _rig_transforms(
    pair_solutions: dict[
        str,
        tuple[np.ndarray, np.ndarray] | None,
    ],
) -> tuple[dict[str, dict[str, Any]], bool]:
    left_pair = pair_solutions.get("front_left__front")
    right_pair = pair_solutions.get("front__front_right")
    identity = np.eye(3, dtype=np.float64)
    zero = np.zeros((3, 1), dtype=np.float64)
    transforms = {
        "front": {
            "rotation_camera_to_front": identity.tolist(),
            "translation_camera_to_front_mm": zero.reshape(-1).tolist(),
        }
    }
    if left_pair is not None:
        rotation, translation = left_pair
        transforms["front_left"] = {
            "rotation_camera_to_front": rotation.tolist(),
            "translation_camera_to_front_mm": (
                translation.reshape(-1).tolist()
            ),
        }
    if right_pair is not None:
        rotation_front_to_right, translation_front_to_right = right_pair
        rotation_right_to_front = rotation_front_to_right.T
        translation_right_to_front = (
            -rotation_right_to_front @ translation_front_to_right
        )
        transforms["front_right"] = {
            "rotation_camera_to_front": rotation_right_to_front.tolist(),
            "translation_camera_to_front_mm": (
                translation_right_to_front.reshape(-1).tolist()
            ),
        }
    return transforms, all(camera in transforms for camera in CAMERA_KEYS)


def _latest_snapshot_frames(
    session: CalibrationSession,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = PROJECT_ROOT / "projects" / "calibration_snapshots"
    if root.exists():
        for directory in sorted(
            (path for path in root.iterdir() if path.is_dir()),
            reverse=True,
        ):
            metadata_path = directory / "metadata.yaml"
            if not metadata_path.exists():
                continue
            metadata = load_yaml(metadata_path)
            if metadata.get("topology") != session.data.get("topology"):
                continue
            frames: dict[str, np.ndarray] = {}
            raw_files = metadata.get("files", {}).get("raw", {})
            for camera in CAMERA_KEYS:
                filename = raw_files.get(camera, f"raw_{camera}.png")
                image = cv2.imread(str(directory / Path(filename).name))
                if image is not None:
                    frames[camera] = image
            if all(camera in frames for camera in CAMERA_KEYS):
                return frames, {
                    "source": "calibration_snapshot",
                    "snapshot_id": directory.name,
                    "synchronized": False,
                    "warning": (
                        "Snapshot frames are software-near-time only, not "
                        "hardware synchronized."
                    ),
                }
    frames = {}
    sample_ids = {}
    for camera in CAMERA_KEYS:
        records = [
            sample
            for sample in session.data["intrinsics"][camera]["samples"]
            if sample.get("accepted")
        ]
        if not records:
            continue
        record = records[-1]
        raw_path = session.directory / record.get("files", {}).get("raw", "")
        image = cv2.imread(str(raw_path))
        if image is not None:
            frames[camera] = image
            sample_ids[camera] = record.get("sample_id", "")
    return frames, {
        "source": "independent_intrinsic_samples",
        "sample_ids": sample_ids,
        "synchronized": False,
        "warning": (
            "No common three-camera snapshot was available; panorama frames "
            "come from different sample times."
        ),
    }


def _panorama_rays() -> np.ndarray:
    width, height = PANORAMA_SIZE
    yaw = np.deg2rad(
        np.linspace(
            PANORAMA_YAW_RANGE_DEG[0],
            PANORAMA_YAW_RANGE_DEG[1],
            width,
            dtype=np.float32,
        )
    )
    pitch = np.deg2rad(
        np.linspace(
            PANORAMA_PITCH_RANGE_DEG[0],
            PANORAMA_PITCH_RANGE_DEG[1],
            height,
            dtype=np.float32,
        )
    )
    yaw_grid, pitch_grid = np.meshgrid(yaw, pitch)
    cos_pitch = np.cos(pitch_grid)
    return np.stack(
        (
            np.sin(yaw_grid) * cos_pitch,
            np.sin(pitch_grid),
            np.cos(yaw_grid) * cos_pitch,
        ),
        axis=-1,
    ).astype(np.float64)


def _candidate_remap(
    rays_front: np.ndarray,
    rotation_camera_to_front: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = rays_front.shape[:2]
    rays_camera = (
        rays_front.reshape(-1, 3) @ rotation_camera_to_front
    )
    projected, _ = cv2.fisheye.projectPoints(
        rays_camera.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(height, width, 2)
    map_x = projected[:, :, 0].astype(np.float32)
    map_y = projected[:, :, 1].astype(np.float32)
    input_width, input_height = resolution
    forward = rays_camera[:, 2].reshape(height, width)
    valid = (
        (forward > 0.02)
        & (map_x >= 0.0)
        & (map_x < input_width - 1)
        & (map_y >= 0.0)
        & (map_y < input_height - 1)
    )
    return map_x, map_y, valid, forward.reshape(height, width)


def _overlap_suggestion(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> dict[str, Any]:
    shared = left_mask & right_mask
    minimum_rows = max(8, int(shared.shape[0] * 0.08))
    columns = np.where(np.count_nonzero(shared, axis=0) >= minimum_rows)[0]
    if not len(columns):
        return {
            "x_range": None,
            "seam_x": None,
            "feather_width": None,
            "warning": "No stable candidate overlap was found.",
        }
    start = int(columns[0])
    end = int(columns[-1])
    width = end - start + 1
    return {
        "x_range": [start, end],
        "seam_x": int(round((start + end) * 0.5)),
        "feather_width": min(120, max(16, width // 4)),
        "warning": (
            "Diagnostic suggestion only; translation/parallax and real "
            "scene depth were not optimized."
        ),
    }


def _pair_preview(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    seam_x: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.zeros_like(left_image)
    only_left = left_mask & ~right_mask
    only_right = right_mask & ~left_mask
    shared = left_mask & right_mask
    alpha[only_left] = left_image[only_left]
    alpha[only_right] = right_image[only_right]
    alpha[shared] = (
        left_image[shared].astype(np.float32) * 0.5
        + right_image[shared].astype(np.float32) * 0.5
    ).astype(np.uint8)

    hard = np.zeros_like(left_image)
    if seam_x is None:
        hard[:] = alpha
    else:
        hard[:, :seam_x][left_mask[:, :seam_x]] = (
            left_image[:, :seam_x][left_mask[:, :seam_x]]
        )
        hard[:, seam_x:][right_mask[:, seam_x:]] = (
            right_image[:, seam_x:][right_mask[:, seam_x:]]
        )
        left_fallback = left_mask & ~np.any(hard != 0, axis=2)
        right_fallback = right_mask & ~np.any(hard != 0, axis=2)
        hard[left_fallback] = left_image[left_fallback]
        hard[right_fallback] = right_image[right_fallback]
    return alpha, hard


def _generate_panorama_artifacts(
    output_dir: Path,
    session: CalibrationSession,
    calibrations: dict[str, tuple[np.ndarray, np.ndarray] | None],
    transforms: dict[str, dict[str, Any]],
    experimental: bool,
) -> dict[str, Any]:
    rays = _panorama_rays()
    frames, preview_source = _latest_snapshot_frames(session)
    remapped: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    forward_weights: dict[str, np.ndarray] = {}
    files: dict[str, Any] = {
        "remaps": {},
        "remap_previews": {},
        "pair_previews": {},
        "canvas": None,
    }
    for camera in CAMERA_KEYS:
        calibration = calibrations.get(camera)
        transform = transforms.get(camera)
        if calibration is None or transform is None:
            continue
        camera_matrix, distortion = calibration
        map_x, map_y, valid, forward = _candidate_remap(
            rays,
            np.asarray(
                transform["rotation_camera_to_front"],
                dtype=np.float64,
            ),
            camera_matrix,
            distortion,
            CANDIDATE_RESOLUTION,
        )
        remap_path = output_dir / "intrinsics" / f"{camera}_remap.npz"
        np.savez_compressed(
            remap_path,
            map_x=map_x,
            map_y=map_y,
            valid_mask=valid.astype(np.uint8),
        )
        files["remaps"][camera] = remap_path.relative_to(
            output_dir
        ).as_posix()
        masks[camera] = valid
        forward_weights[camera] = np.where(valid, forward, -np.inf)
        frame = frames.get(camera)
        if frame is None:
            continue
        preview = cv2.remap(
            frame,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        preview[~valid] = 0
        remapped[camera] = preview
        preview_path = output_dir / "previews" / f"remap_{camera}.png"
        save_image(preview_path, preview)
        files["remap_previews"][camera] = preview_path.relative_to(
            output_dir
        ).as_posix()

    suggestions = {}
    for left_camera, right_camera in PAIR_KEYS:
        pair_key = f"{left_camera}__{right_camera}"
        if left_camera not in masks or right_camera not in masks:
            continue
        suggestion = _overlap_suggestion(
            masks[left_camera],
            masks[right_camera],
        )
        suggestions[pair_key] = suggestion
        if left_camera not in remapped or right_camera not in remapped:
            continue
        alpha, hard = _pair_preview(
            remapped[left_camera],
            remapped[right_camera],
            masks[left_camera],
            masks[right_camera],
            suggestion["seam_x"],
        )
        alpha_path = (
            output_dir / "previews" / f"{pair_key}_alpha_50.png"
        )
        hard_path = output_dir / "previews" / f"{pair_key}_hard_seam.png"
        save_image(alpha_path, alpha)
        save_image(hard_path, hard)
        files["pair_previews"][pair_key] = {
            "alpha_50": alpha_path.relative_to(output_dir).as_posix(),
            "hard_seam": hard_path.relative_to(output_dir).as_posix(),
        }

    if remapped:
        camera_order = [
            camera for camera in CAMERA_KEYS if camera in remapped
        ]
        weights = np.stack(
            [forward_weights[camera] for camera in camera_order],
            axis=0,
        )
        selected = np.argmax(weights, axis=0)
        available = np.any(np.isfinite(weights), axis=0)
        canvas = np.zeros_like(next(iter(remapped.values())))
        for index, camera in enumerate(camera_order):
            selection = available & (selected == index) & masks[camera]
            canvas[selection] = remapped[camera][selection]
        cv2.putText(
            canvas,
            (
                "EXPERIMENTAL CANDIDATE - rotation-only virtual panorama"
                if experimental
                else "CALIBRATION CANDIDATE - rotation-only virtual panorama"
            ),
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (30, 220, 255),
            2,
            cv2.LINE_AA,
        )
        canvas_path = output_dir / "previews" / "candidate_canvas.png"
        save_image(canvas_path, canvas)
        files["canvas"] = canvas_path.relative_to(output_dir).as_posix()

    return {
        "projection": "equirectangular_rotation_only",
        "reference_camera": "front",
        "canvas_size": list(PANORAMA_SIZE),
        "yaw_range_degrees": list(PANORAMA_YAW_RANGE_DEG),
        "pitch_range_degrees": list(PANORAMA_PITCH_RANGE_DEG),
        "preview_source": preview_source,
        "suggested_overlaps": suggestions,
        "files": files,
        "limitations": [
            "Translation and scene depth are not represented by the "
            "rotation-only panorama.",
            "Near objects can retain physical parallax even when distant "
            "content aligns.",
            "Overlap, seam, and feather values are diagnostic suggestions "
            "and were not written to the formal profile.",
        ],
    }


def _candidate_output_directory(
    output_root: Path,
    session_id: str,
    generated_at: datetime,
) -> Path:
    session_root = output_root / session_id
    name = generated_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = session_root / name
    suffix = 1
    while output_dir.exists():
        output_dir = session_root / f"{name}_{suffix:02d}"
        suffix += 1
    for child in ("intrinsics", "stereo_pairs", "previews"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_calibration_candidate(
    session_directory: str | Path,
    output_root: str | Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Solve and save a candidate without reading or writing formal geometry."""
    formal_path = CONFIG_DIR / "calibration.yaml"
    formal_revision_before = file_revision(formal_path)
    session = CalibrationSession.load(session_directory)
    access = session.b2_candidate_access()
    if not access["can_enter"]:
        raise CandidateSolveError(
            "Session readiness is 'not recommended'; candidate solve blocked."
        )
    resolution = tuple(int(value) for value in session.data["resolution"])
    if resolution != CANDIDATE_RESOLUTION:
        raise CandidateSolveError(
            "Candidate solve requires session resolution 1920x1080."
        )
    generated_at = generated_at or datetime.now().astimezone()
    output_root = Path(
        output_root
        or PROJECT_ROOT / "projects" / "calibration_candidates"
    )
    output_dir = _candidate_output_directory(
        output_root,
        session.directory.name,
        generated_at,
    )
    lookup = _object_point_lookup(session.data["board_definition"])

    intrinsic_results = {}
    calibrations: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for camera in CAMERA_KEYS:
        result, calibration = _solve_intrinsics(
            session,
            camera,
            lookup,
            resolution,
        )
        intrinsic_results[camera] = result
        calibrations[camera] = calibration
        save_yaml(output_dir / "intrinsics" / f"{camera}.yaml", result)

    pair_results = {}
    pair_solutions = {}
    for left_camera, right_camera in PAIR_KEYS:
        pair_key = f"{left_camera}__{right_camera}"
        result, solution = _solve_stereo_pair(
            session,
            left_camera,
            right_camera,
            lookup,
            calibrations,
            resolution,
        )
        pair_results[pair_key] = result
        pair_solutions[pair_key] = solution
        save_yaml(
            output_dir / "stereo_pairs" / f"{pair_key}.yaml",
            result,
        )

    transforms, rig_complete = _rig_transforms(pair_solutions)
    panorama = None
    panorama_error = ""
    if rig_complete:
        try:
            panorama = _generate_panorama_artifacts(
                output_dir,
                session,
                calibrations,
                transforms,
                bool(access["experimental"]),
            )
        except Exception as exc:
            panorama_error = f"{type(exc).__name__}: {exc}"

    formal_revision_after = file_revision(formal_path)
    if formal_revision_after != formal_revision_before:
        raise RuntimeError(
            "Formal calibration.yaml changed during candidate solve."
        )
    readiness = deepcopy(access["readiness"])
    candidate = {
        "format": CANDIDATE_FORMAT,
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(timespec="milliseconds"),
        "session_id": session.directory.name,
        "session_directory": str(session.directory.resolve()),
        "session_format": session.data.get("format"),
        "session_schema_version": session.data.get("schema_version"),
        "topology": session.data.get("topology"),
        "resolution": list(resolution),
        "board_definition": deepcopy(session.data["board_definition"]),
        "source_coordinate_space": session.data.get(
            "source_coordinate_space"
        ),
        "source_reference_size": deepcopy(
            session.data.get("source_reference_size")
        ),
        "quality_level": (
            "experimental" if access["experimental"] else "candidate"
        ),
        "experimental": bool(access["experimental"]),
        "report_only": bool(access["report_only"]),
        "recommended": False,
        "apply_allowed": False,
        "formal_application_eligible": bool(access["apply_allowed"]),
        "writes_formal_profile": False,
        "formal_calibration_revision_before": formal_revision_before,
        "formal_calibration_revision_after": formal_revision_after,
        "readiness": readiness,
        "intrinsics": intrinsic_results,
        "stereo_pairs": pair_results,
        "rig": {
            "reference_camera": "front",
            "complete": rig_complete,
            "transforms": transforms,
        },
        "virtual_panorama": panorama,
        "virtual_panorama_error": panorama_error,
    }
    report = {
        "format": "DeepSharkFisheyeCalibrationCandidateReport",
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_file": "candidate.yaml",
        "quality_level": candidate["quality_level"],
        "experimental": candidate["experimental"],
        "apply_allowed": False,
        "summary": {
            "intrinsics": {
                camera: {
                    key: result.get(key)
                    for key in (
                        "status",
                        "accepted_input_count",
                        "used_sample_count",
                        "outlier_count",
                        "rms_px",
                        "error",
                    )
                }
                for camera, result in intrinsic_results.items()
            },
            "stereo_pairs": {
                pair: {
                    key: result.get(key)
                    for key in (
                        "status",
                        "accepted_input_count",
                        "used_sample_count",
                        "outlier_count",
                        "rms_px",
                        "time_delta_statistics",
                        "error",
                    )
                }
                for pair, result in pair_results.items()
            },
            "rig_complete": rig_complete,
            "panorama_generated": panorama is not None,
        },
        "quality_issues": readiness["quality_issues"],
        "blocking_reasons": readiness["blocking_reasons"],
        "limitations": [
            "This candidate is diagnostic and cannot be applied.",
            "Pair frames use software timestamps rather than hardware sync.",
            "Physical parallax cannot be removed by a single rotation-only "
            "virtual panorama.",
            "Additional pair position, distance, and tilt coverage is "
            "required before a formal application review.",
        ],
    }
    save_yaml(output_dir / "candidate.yaml", candidate)
    save_yaml(output_dir / "report.yaml", report)
    return output_dir


def load_calibration_candidate(
    candidate_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(candidate_directory)
    candidate = load_yaml(directory / "candidate.yaml")
    report = load_yaml(directory / "report.yaml")
    if candidate.get("format") != CANDIDATE_FORMAT:
        raise ValueError("Not a DeepShark fisheye calibration candidate.")
    if candidate.get("resolution") != list(CANDIDATE_RESOLUTION):
        raise ValueError("Candidate resolution is not 1920x1080.")
    if candidate.get("experimental") and candidate.get("apply_allowed"):
        raise ValueError("Experimental candidate cannot allow application.")
    return candidate, report


def latest_calibration_candidate(
    candidates_root: str | Path | None = None,
    topology: str = "triple_front_panorama",
) -> Path | None:
    """Return the newest complete candidate suitable for runtime trials."""
    root = Path(
        candidates_root
        or PROJECT_ROOT / "projects" / "calibration_candidates"
    )
    if not root.exists():
        return None
    directories = sorted(
        (
            run
            for session_dir in root.iterdir()
            if session_dir.is_dir()
            for run in session_dir.iterdir()
            if run.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for directory in directories:
        try:
            candidate, _ = load_calibration_candidate(directory)
        except (OSError, ValueError, TypeError):
            continue
        panorama = candidate.get("virtual_panorama") or {}
        canvas_file = panorama.get("files", {}).get("canvas")
        if (
            candidate.get("topology") == topology
            and candidate.get("rig", {}).get("complete")
            and canvas_file
            and (directory / canvas_file).exists()
        ):
            return directory
    return None


class CandidatePanoramaProcessor:
    """Apply a saved report-only candidate to live frames without write-back."""

    def __init__(self, candidate_directory: str | Path):
        self.directory = Path(candidate_directory)
        self.candidate, _ = load_calibration_candidate(self.directory)
        if not self.candidate.get("rig", {}).get("complete"):
            raise ValueError("Candidate rig is incomplete.")
        panorama = self.candidate.get("virtual_panorama") or {}
        self.camera_order = [
            camera
            for camera in CAMERA_KEYS
            if camera in panorama.get("files", {}).get("remaps", {})
        ]
        if self.camera_order != list(CAMERA_KEYS):
            raise ValueError(
                "Runtime candidate must contain all three camera remaps."
            )
        self.input_resolution = tuple(
            int(value) for value in self.candidate["resolution"]
        )
        self.output_width, self.output_height = (
            int(value) for value in panorama["canvas_size"]
        )
        self.topology_name = str(self.candidate.get("topology", ""))
        self.experimental = bool(self.candidate.get("experimental"))
        self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._masks: dict[str, np.ndarray] = {}
        for camera, relative_path in panorama["files"]["remaps"].items():
            with np.load(self.directory / relative_path) as data:
                self._maps[camera] = (
                    np.asarray(data["map_x"], dtype=np.float32),
                    np.asarray(data["map_y"], dtype=np.float32),
                )
                self._masks[camera] = np.asarray(
                    data["valid_mask"],
                    dtype=np.uint8,
                ).astype(bool)
        rays = _panorama_rays()
        transforms = self.candidate["rig"]["transforms"]
        self._weights = {}
        for camera in self.camera_order:
            rotation = np.asarray(
                transforms[camera]["rotation_camera_to_front"],
                dtype=np.float64,
            )
            rays_camera = rays.reshape(-1, 3) @ rotation
            forward = rays_camera[:, 2].reshape(
                self.output_height,
                self.output_width,
            )
            self._weights[camera] = np.where(
                self._masks[camera],
                forward,
                -np.inf,
            )

    def process(
        self,
        frames: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        warped: dict[str, np.ndarray] = {}
        expected_width, expected_height = self.input_resolution
        for camera in self.camera_order:
            frame = frames.get(camera)
            if frame is None:
                continue
            actual_size = (int(frame.shape[1]), int(frame.shape[0]))
            if actual_size != self.input_resolution:
                raise ValueError(
                    f"{camera} frame size {actual_size} does not match "
                    f"candidate resolution {self.input_resolution}."
                )
            map_x, map_y = self._maps[camera]
            image = cv2.remap(
                frame,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            image[~self._masks[camera]] = 0
            warped[camera] = image
        canvas = np.zeros(
            (self.output_height, self.output_width, 3),
            dtype=np.uint8,
        )
        if not warped:
            return warped, canvas
        available = [camera for camera in self.camera_order if camera in warped]
        weights = np.stack(
            [self._weights[camera] for camera in available],
            axis=0,
        )
        selected = np.argmax(weights, axis=0)
        valid_any = np.any(np.isfinite(weights), axis=0)
        for index, camera in enumerate(available):
            selection = (
                valid_any
                & (selected == index)
                & self._masks[camera]
            )
            canvas[selection] = warped[camera][selection]
        return warped, canvas
