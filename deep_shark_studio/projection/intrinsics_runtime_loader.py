"""Read-only fisheye intrinsics loader for experimental runtime projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deep_shark_studio.config import load_yaml


RUNTIME_CAMERA_KEYS = ("front_left", "front", "front_right")


class FisheyeIntrinsicsRuntimeError(ValueError):
    """Raised when a fisheye intrinsics source is not runtime-safe."""


@dataclass(frozen=True)
class FisheyeCameraIntrinsics:
    camera: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    model: str
    reprojection_error: float | None = None
    accepted_samples: int | None = None


@dataclass(frozen=True)
class FisheyeIntrinsicsRuntimeSource:
    path: Path
    source_type: str
    cameras: dict[str, FisheyeCameraIntrinsics]
    metadata: dict[str, Any]
    warnings: tuple[str, ...] = ()


def load_fisheye_intrinsics_source(
    path: str | Path,
) -> FisheyeIntrinsicsRuntimeSource:
    """Load a read-only three-camera OpenCV fisheye K/D source."""
    source_path = Path(path)
    candidate_path = _candidate_yaml_path(source_path)
    if candidate_path.exists():
        return _load_calibration_candidate(candidate_path)
    raise FisheyeIntrinsicsRuntimeError(
        "Unsupported fisheye intrinsics source. Expected a calibration candidate directory or candidate.yaml."
    )


def _candidate_yaml_path(path: Path) -> Path:
    return path if path.is_file() else path / "candidate.yaml"


def _load_calibration_candidate(path: Path) -> FisheyeIntrinsicsRuntimeSource:
    data = load_yaml(path)
    if data.get("format") != "DeepSharkFisheyeCalibrationCandidate":
        raise FisheyeIntrinsicsRuntimeError(
            "Fisheye runtime source must be a DeepShark fisheye calibration candidate."
        )
    intrinsics = data.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise FisheyeIntrinsicsRuntimeError("candidate.yaml missing intrinsics block.")
    cameras: dict[str, FisheyeCameraIntrinsics] = {}
    warnings: list[str] = []
    for camera in RUNTIME_CAMERA_KEYS:
        item = intrinsics.get(camera)
        if not isinstance(item, dict):
            raise FisheyeIntrinsicsRuntimeError(f"Missing fisheye intrinsics for {camera}.")
        cameras[camera] = _parse_candidate_camera(camera, item)
        if cameras[camera].reprojection_error is None:
            warnings.append(f"{camera}: reprojection_error/RMS is not recorded.")
        if cameras[camera].accepted_samples is None:
            warnings.append(f"{camera}: accepted_samples is not recorded.")
    metadata = {
        "format": data.get("format"),
        "topology": data.get("topology"),
        "resolution": data.get("resolution"),
        "experimental": bool(data.get("experimental", False)),
        "apply_allowed": bool(data.get("apply_allowed", False)),
    }
    return FisheyeIntrinsicsRuntimeSource(
        path=path,
        source_type="calibration_candidate",
        cameras=cameras,
        metadata=metadata,
        warnings=tuple(warnings),
    )


def _parse_candidate_camera(
    camera: str,
    item: dict[str, Any],
) -> FisheyeCameraIntrinsics:
    status = str(item.get("status", ""))
    if status and status != "success":
        raise FisheyeIntrinsicsRuntimeError(
            f"{camera}: fisheye calibration status is {status!r}, expected 'success'."
        )
    model = str(item.get("model", ""))
    if model != "opencv_fisheye":
        raise FisheyeIntrinsicsRuntimeError(
            f"{camera}: model must be opencv_fisheye, got {model!r}."
        )
    camera_matrix = item.get("camera_matrix")
    distortion = item.get("distortion_coefficients")
    if camera_matrix is None:
        raise FisheyeIntrinsicsRuntimeError(f"{camera}: missing camera_matrix.")
    if distortion is None:
        raise FisheyeIntrinsicsRuntimeError(f"{camera}: missing distortion_coefficients.")
    resolution = item.get("resolution")
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise FisheyeIntrinsicsRuntimeError(f"{camera}: missing resolution/image_size.")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    if matrix.shape != (3, 3):
        raise FisheyeIntrinsicsRuntimeError(f"{camera}: camera_matrix must be 3x3.")
    if dist.size != 4:
        raise FisheyeIntrinsicsRuntimeError(f"{camera}: OpenCV fisheye D must contain 4 values.")
    return FisheyeCameraIntrinsics(
        camera=camera,
        camera_matrix=matrix,
        distortion=dist.reshape(4, 1),
        image_size=(int(resolution[0]), int(resolution[1])),
        model=model,
        reprojection_error=_optional_float(item.get("rms_px")),
        accepted_samples=_optional_int(item.get("accepted_input_count")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
