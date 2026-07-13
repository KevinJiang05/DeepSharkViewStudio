"""Far-field custom layout candidate loader and saver."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.config import CONFIG_DIR, PROJECT_ROOT, file_revision, load_yaml, save_yaml
from deep_shark_studio.projection.runtime_providers import B2_FAR_FIELD_PROJECTION_SOURCE
from deep_shark_studio.stitcher import save_image
from deep_shark_studio.stitching.camera_layout_adjust import CameraAdjustParams


class FarFieldLayoutCandidateError(ValueError):
    """Raised when a Far-field layout candidate is not runtime-safe."""


@dataclass(frozen=True)
class FarFieldLayoutRuntimeCandidate:
    path: Path
    schema_version: int
    profile_id: str
    camera_adjust: dict[str, CameraAdjustParams]
    output_width_px: int
    output_height_px: int
    feather_width_override_px: int | None
    calibration_hash: str | None
    current_calibration_hash: str | None
    warnings: tuple[str, ...]
    raw: dict[str, Any]
    projection_source: str = "current_perspective"
    b2_candidate_directory: Path | None = None
    blend_mode: str | None = None

    @property
    def candidate_directory(self) -> Path:
        return self.path.parent if self.path.is_file() else self.path


def default_far_field_layout_candidate_root() -> Path:
    return PROJECT_ROOT / "layout_candidates"


def load_far_field_layout_candidate(
    path: str | Path,
    expected_profile_id: str = "triple_front_panorama",
    formal_calibration_path: str | Path | None = None,
) -> FarFieldLayoutRuntimeCandidate:
    candidate_path = _candidate_yaml_path(Path(path))
    if not candidate_path.exists():
        raise FarFieldLayoutCandidateError(f"candidate.yaml not found: {candidate_path}")
    data = load_yaml(candidate_path)
    schema_version = _required_int(data, "schema_version")
    if schema_version != 1:
        raise FarFieldLayoutCandidateError(
            f"Unsupported far_field_layout schema_version: {schema_version}"
        )
    if data.get("candidate_type") != "far_field_layout":
        raise FarFieldLayoutCandidateError(
            "Far-field custom runtime requires candidate_type=far_field_layout."
        )
    profile_id = _required_str(data, "profile_id")
    if profile_id != expected_profile_id:
        raise FarFieldLayoutCandidateError(
            f"Far-field candidate profile_id={profile_id!r} does not match {expected_profile_id!r}."
        )
    if bool(data.get("writes_calibration_yaml", True)):
        raise FarFieldLayoutCandidateError(
            "Far-field candidate is not runtime-safe: writes_calibration_yaml must be false."
        )
    if bool(data.get("formal_profile_modified", True)):
        raise FarFieldLayoutCandidateError(
            "Far-field candidate is not runtime-safe: formal_profile_modified must be false."
        )
    projection = _required_mapping(data, "projection")
    projection_source = str(projection.get("source", ""))
    if projection_source not in {"current_perspective", B2_FAR_FIELD_PROJECTION_SOURCE}:
        raise FarFieldLayoutCandidateError(
            "Far-field custom layout supports projection.source=current_perspective or b2_far_field_candidate."
        )
    b2_candidate_directory: Path | None = None
    if projection_source == B2_FAR_FIELD_PROJECTION_SOURCE:
        raw_directory = projection.get("candidate_directory") or projection.get(
            "b2_candidate_directory"
        )
        if not isinstance(raw_directory, str) or not raw_directory.strip():
            raise FarFieldLayoutCandidateError(
                "projection.candidate_directory is required for b2_far_field_candidate."
            )
        b2_candidate_directory = _resolve_candidate_path(
            raw_directory,
            candidate_path.parent,
        )
    output = _required_mapping(data, "output")
    output_width = _bounded_int(output, "width_px", 1, 5000, "output.width_px")
    output_height = _bounded_int(output, "height_px", 1, 3000, "output.height_px")
    blend = _required_mapping(data, "far_field_blend")
    blend_mode = str(blend.get("mode", ""))
    if blend_mode not in {"current_horizontal_feather", "b2_weight_selection"}:
        raise FarFieldLayoutCandidateError(
            "far_field_blend.mode must be current_horizontal_feather or b2_weight_selection."
        )
    expected_blend_mode = (
        "b2_weight_selection"
        if projection_source == B2_FAR_FIELD_PROJECTION_SOURCE
        else "current_horizontal_feather"
    )
    if blend_mode != expected_blend_mode:
        raise FarFieldLayoutCandidateError(
            f"projection.source={projection_source} requires "
            f"far_field_blend.mode={expected_blend_mode}; got {blend_mode}."
        )
    feather_override = blend.get("feather_width_override_px")
    if feather_override is not None:
        feather_override = _bounded_int(
            blend,
            "feather_width_override_px",
            1,
            1200,
            "far_field_blend.feather_width_override_px",
        )
    calibration_path = Path(formal_calibration_path or CONFIG_DIR / "calibration.yaml")
    current_hash = file_revision(calibration_path)
    source = data.get("source", {})
    calibration_hash = (
        str(source.get("calibration_hash"))
        if isinstance(source, dict) and source.get("calibration_hash")
        else None
    )
    warnings: list[str] = []
    if projection_source == "current_perspective":
        warnings.append(
            "Legacy Far-field layout candidate uses current_perspective; new Far-field layout tuning should use B-2 candidate projection."
        )
    if b2_candidate_directory is not None and not b2_candidate_directory.exists():
        warnings.append(
            f"B-2 candidate directory does not exist: {b2_candidate_directory}"
        )
    if calibration_hash and current_hash and calibration_hash != current_hash:
        warnings.append(
            "Far-field layout candidate calibration hash differs from current calibration.yaml."
        )
    return FarFieldLayoutRuntimeCandidate(
        path=candidate_path,
        schema_version=schema_version,
        profile_id=profile_id,
        camera_adjust=_parse_camera_adjust(_required_mapping(data, "camera_adjust")),
        output_width_px=output_width,
        output_height_px=output_height,
        feather_width_override_px=feather_override,
        calibration_hash=calibration_hash,
        current_calibration_hash=current_hash,
        warnings=tuple(warnings),
        raw=data,
        projection_source=projection_source,
        b2_candidate_directory=b2_candidate_directory,
        blend_mode=blend_mode,
    )


def save_far_field_layout_candidate(
    output_root: Path,
    profile_id: str,
    camera_adjust: dict[str, CameraAdjustParams],
    output_width_px: int,
    output_height_px: int,
    preview_result: Any,
    source: dict[str, Any] | None = None,
    feather_width_override_px: int | None = None,
) -> Path:
    calibration_path = CONFIG_DIR / "calibration.yaml"
    before_hash = file_revision(calibration_path)
    directory = _create_candidate_directory(output_root)
    debug_dir = directory / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    preview_image = np.asarray(preview_result.image)
    save_image(directory / "preview.png", preview_image)
    save_image(directory / "preview_with_overlay.png", _far_field_overlay(preview_result))
    adjusted = getattr(preview_result, "adjusted", None)
    debug_files = _save_adjusted_debug_images(debug_dir, adjusted)

    metrics = dict(getattr(preview_result, "metrics", {}) or {})
    projection_block = _projection_block_from_source(source)
    source_for_yaml = _sanitize_yaml_source(source)
    if not isinstance(source_for_yaml, dict):
        source_for_yaml = {}
    blend_mode = (
        "b2_weight_selection"
        if projection_block.get("source") == B2_FAR_FIELD_PROJECTION_SOURCE
        else "current_horizontal_feather"
    )
    candidate = {
        "schema_version": 1,
        "candidate_type": "far_field_layout",
        "profile_id": profile_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "source": {
            **source_for_yaml,
            "calibration_hash": before_hash,
        },
        "projection": projection_block,
        "camera_adjust": {
            camera: params.to_dict()
            for camera, params in camera_adjust.items()
        },
        "output": {
            "width_px": int(output_width_px),
            "height_px": int(output_height_px),
        },
        "far_field_blend": {
            "mode": blend_mode,
            "use_profile_overlaps": True,
            "feather_width_override_px": feather_width_override_px,
        },
        "status": "candidate",
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
        "metrics": metrics,
        "files": {
            "preview": "preview.png",
            "preview_with_overlay": "preview_with_overlay.png",
            "debug": debug_files,
        },
    }
    report = {
        "candidate_type": "far_field_layout",
        "profile_id": profile_id,
        "calibration_hash_before": before_hash,
        "calibration_hash_after": file_revision(calibration_path),
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
        "metrics": metrics,
        "camera_adjust_metadata": getattr(adjusted, "metadata", None),
    }
    if report["calibration_hash_before"] != report["calibration_hash_after"]:
        raise RuntimeError("calibration.yaml changed while saving Far-field layout candidate.")
    save_yaml(directory / "candidate.yaml", candidate)
    save_yaml(directory / "report.yaml", report)
    return directory


def _candidate_yaml_path(path: Path) -> Path:
    return path if path.is_file() else path / "candidate.yaml"


def _projection_block_from_source(source: dict[str, Any] | None) -> dict[str, Any]:
    source = source or {}
    projection = source.get("projection")
    if not isinstance(projection, dict):
        projection = {}
    projection_source = str(
        projection.get("projection_source")
        or projection.get("source")
        or "current_perspective"
    )
    if projection_source == B2_FAR_FIELD_PROJECTION_SOURCE:
        candidate_directory = projection.get("candidate_directory")
        if not isinstance(candidate_directory, str) or not candidate_directory.strip():
            raise FarFieldLayoutCandidateError(
                "Cannot save B-2 Far-field layout candidate without projection.candidate_directory."
            )
        block = {
            "source": B2_FAR_FIELD_PROJECTION_SOURCE,
            "candidate_directory": str(candidate_directory),
            "note": "Far-field custom layout uses read-only B-2 per-camera warped images with B-2 weight-selection composition.",
        }
        return block
    return {
        "source": "current_perspective",
        "note": "Legacy Far-field custom layout uses current perspective projection.",
    }


def _sanitize_yaml_source(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            sanitized[str(key)] = _sanitize_yaml_source(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_yaml_source(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return "<runtime-only-array>"
    return value


def _resolve_candidate_path(raw_path: str, base_directory: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_directory / path).resolve()


def _parse_camera_adjust(raw: dict[str, Any]) -> dict[str, CameraAdjustParams]:
    parsed: dict[str, CameraAdjustParams] = {}
    for camera in ("front_left", "front", "front_right"):
        values = raw.get(camera)
        if not isinstance(values, dict):
            raise FarFieldLayoutCandidateError(f"camera_adjust.{camera} is required.")
        parsed[camera] = CameraAdjustParams(
            x_offset_px=_bounded_int(values, "x_offset_px", -240, 240, f"camera_adjust.{camera}.x_offset_px"),
            y_offset_px=_bounded_int(values, "y_offset_px", -160, 160, f"camera_adjust.{camera}.y_offset_px"),
            scale=_bounded_float(values, "scale", 0.8, 1.2, f"camera_adjust.{camera}.scale"),
        )
    return parsed


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise FarFieldLayoutCandidateError(f"{key} is required.")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FarFieldLayoutCandidateError(f"{key} must be an integer.")
    return int(value)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise FarFieldLayoutCandidateError(f"{key} must be a non-empty string.")
    return value


def _bounded_int(
    data: dict[str, Any],
    key: str,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FarFieldLayoutCandidateError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FarFieldLayoutCandidateError(f"{label} must be finite.")
    result = int(round(numeric))
    if result < minimum or result > maximum:
        raise FarFieldLayoutCandidateError(
            f"{label}={result} is outside [{minimum}, {maximum}]."
        )
    return result


def _bounded_float(
    data: dict[str, Any],
    key: str,
    minimum: float,
    maximum: float,
    label: str,
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FarFieldLayoutCandidateError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise FarFieldLayoutCandidateError(f"{label} must be finite.")
    if result < minimum or result > maximum:
        raise FarFieldLayoutCandidateError(
            f"{label}={result} is outside [{minimum}, {maximum}]."
        )
    return result


def _create_candidate_directory(output_root: Path) -> Path:
    created = datetime.now().astimezone()
    session_id = created.strftime("%Y%m%d_%H%M%S")
    run_id = created.strftime("run_%H%M%S_%f")[:-3]
    directory = output_root / session_id / run_id
    suffix = 1
    while directory.exists():
        directory = output_root / session_id / f"{run_id}_{suffix:02d}"
        suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _far_field_overlay(preview_result: Any) -> np.ndarray:
    image = np.asarray(preview_result.image).copy()
    text = "Far-field Custom Layout"
    cv2.rectangle(image, (10, 10), (300, 44), (15, 23, 42), thickness=-1)
    cv2.putText(
        image,
        text,
        (22, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (248, 250, 252),
        2,
        cv2.LINE_AA,
    )
    return image


def _save_adjusted_debug_images(debug_dir: Path, adjusted: Any) -> dict[str, str]:
    files: dict[str, str] = {}
    if adjusted is None:
        return files
    for camera, image in getattr(adjusted, "warped_images", {}).items():
        name = f"adjusted_{camera}.png"
        save_image(debug_dir / name, image)
        files[f"adjusted_{camera}"] = f"debug/{name}"
    for camera, mask in getattr(adjusted, "valid_masks", {}).items():
        name = f"adjusted_valid_{camera}.png"
        save_image(debug_dir / name, np.dstack([mask.astype(np.uint8) * 255] * 3))
        files[f"adjusted_valid_{camera}"] = f"debug/{name}"
    overlay = _camera_adjust_overlay(adjusted)
    if overlay is not None:
        name = "camera_adjust_overlay.png"
        save_image(debug_dir / name, overlay)
        files["camera_adjust_overlay"] = f"debug/{name}"
    return files


def _camera_adjust_overlay(adjusted: Any) -> np.ndarray | None:
    images = getattr(adjusted, "warped_images", {})
    if not images:
        return None
    first = next(iter(images.values()))
    overlay = np.zeros_like(first)
    colors = {
        "front_left": (0, 0, 220),
        "front": (0, 180, 0),
        "front_right": (220, 0, 0),
    }
    for camera, mask in getattr(adjusted, "valid_masks", {}).items():
        color = np.zeros_like(overlay)
        color[:, :] = colors.get(camera, (180, 180, 180))
        overlay[mask.astype(bool)] = cv2.addWeighted(
            overlay[mask.astype(bool)],
            0.55,
            color[mask.astype(bool)],
            0.45,
            0,
        )
    return overlay
