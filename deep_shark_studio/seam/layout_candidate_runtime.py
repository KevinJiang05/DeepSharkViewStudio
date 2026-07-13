"""Read-only Layout Candidate V2 loader for near-field runtime stitching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from deep_shark_studio.config import CONFIG_DIR, file_revision, load_yaml
from deep_shark_studio.projection.intrinsics_runtime_loader import (
    FisheyeIntrinsicsRuntimeError,
    load_fisheye_intrinsics_source,
)
from deep_shark_studio.stitch_runtime_modes import ProjectionSource
from deep_shark_studio.stitching.camera_layout_adjust import CameraAdjustParams
from deep_shark_studio.topology import active_stitch_profile

from .layout_preview import LayoutPairCandidate
from .layout_tuner import (
    LayoutPreviewParamsV2,
    PairLayoutParams,
    layout_tuner_pair_candidates,
)


class LayoutCandidateRuntimeError(ValueError):
    """Raised when a layout candidate is not safe to use at runtime."""


@dataclass(frozen=True)
class LayoutRuntimeProjection:
    source: ProjectionSource
    intrinsics_source_path: Path | None = None
    balance: float = 0.6
    fov_scale: float = 1.0
    projection_pipeline: str = "current_template_perspective_warp"
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class LayoutRuntimeCandidate:
    path: Path
    schema_version: int
    profile_id: str
    camera_adjust: dict[str, CameraAdjustParams]
    left_pair: PairLayoutParams
    right_pair: PairLayoutParams
    output_width_px: int
    output_height_px: int
    vertical_safety_enabled: bool
    vertical_safe_ratio: float
    side_vertical_fade_px: int
    pair_candidates: list[LayoutPairCandidate]
    calibration_hash: str | None
    current_calibration_hash: str | None
    projection: LayoutRuntimeProjection
    warnings: tuple[str, ...]
    raw: dict[str, Any]

    def to_layout_params(self) -> LayoutPreviewParamsV2:
        return LayoutPreviewParamsV2(
            left_pair=self.left_pair,
            right_pair=self.right_pair,
            camera_adjust=self.camera_adjust,
            output_width_px=self.output_width_px,
            output_height_px=self.output_height_px,
            vertical_safe_ratio=self.vertical_safe_ratio,
            side_vertical_fade_px=self.side_vertical_fade_px,
            vertical_safety_enabled_override=self.vertical_safety_enabled,
        )

    @property
    def candidate_directory(self) -> Path:
        return self.path.parent if self.path.name == "candidate.yaml" else self.path


def load_layout_candidate_for_runtime(
    path: str | Path,
    expected_profile_id: str = "triple_front_panorama",
    formal_calibration_path: str | Path | None = None,
) -> LayoutRuntimeCandidate:
    candidate_path = _candidate_yaml_path(Path(path))
    if not candidate_path.exists():
        raise LayoutCandidateRuntimeError(f"candidate.yaml not found: {candidate_path}")
    data = load_yaml(candidate_path)
    schema_version = _required_int(data, "schema_version")
    if schema_version == 1:
        raise LayoutCandidateRuntimeError(
            "Layout Candidate V1 cannot be loaded by runtime; re-save it as a V2 layout candidate."
        )
    if schema_version not in (2, 3):
        raise LayoutCandidateRuntimeError(
            f"Unsupported layout candidate schema_version: {schema_version}"
        )
    if data.get("candidate_type") != "front_priority_layout":
        raise LayoutCandidateRuntimeError(
            "Layout runtime requires candidate_type=front_priority_layout."
        )
    profile_id = _required_str(data, "profile_id")
    if profile_id != expected_profile_id:
        raise LayoutCandidateRuntimeError(
            f"Layout candidate profile_id={profile_id!r} does not match {expected_profile_id!r}."
        )
    if bool(data.get("writes_calibration_yaml", True)):
        raise LayoutCandidateRuntimeError(
            "Layout candidate is not runtime-safe: writes_calibration_yaml must be false."
        )
    if bool(data.get("formal_profile_modified", True)):
        raise LayoutCandidateRuntimeError(
            "Layout candidate is not runtime-safe: formal_profile_modified must be false."
        )
    projection = _parse_projection(
        data,
        schema_version,
        candidate_path.parent,
    )

    camera_adjust = _parse_camera_adjust(_required_mapping(data, "camera_adjust"))
    left_pair = _parse_pair(_required_mapping(data, "left_pair"), "left_pair")
    right_pair = _parse_pair(_required_mapping(data, "right_pair"), "right_pair")
    output = _required_mapping(data, "output")
    output_width = _bounded_int(output, "width_px", 1, 5000, "output.width_px")
    output_height = _bounded_int(output, "height_px", 1, 3000, "output.height_px")
    vertical = _required_mapping(data, "vertical_safety")
    vertical_enabled = _required_bool(
        vertical,
        "enabled",
        "vertical_safety.enabled",
    )
    vertical_safe_ratio = _bounded_float(
        vertical,
        "vertical_safe_ratio",
        0.0,
        1.0,
        "vertical_safety.vertical_safe_ratio",
    )
    side_vertical_fade = _bounded_int(
        vertical,
        "side_vertical_fade_px",
        0,
        300,
        "vertical_safety.side_vertical_fade_px",
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
    if calibration_hash and current_hash and calibration_hash != current_hash:
        warnings.append(
            "Layout candidate calibration hash differs from current calibration.yaml."
        )

    profile = active_stitch_profile(load_yaml(calibration_path))
    canvas_width, canvas_height = _profile_canvas_size(profile)
    if output_width > canvas_width or output_height > canvas_height:
        raise LayoutCandidateRuntimeError(
            f"Candidate output {output_width}x{output_height} exceeds active profile "
            f"canvas {canvas_width}x{canvas_height}."
        )
    if not vertical_enabled and (
        output_height != canvas_height
        or not math.isclose(vertical_safe_ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-9)
        or side_vertical_fade != 0
    ):
        raise LayoutCandidateRuntimeError(
            "vertical_safety is disabled but output height, safe ratio, or side fade "
            "requests an active vertical adjustment."
        )
    pair_candidates = layout_tuner_pair_candidates(profile)
    if not pair_candidates:
        raise LayoutCandidateRuntimeError(
            "Current profile does not provide front-priority layout pair candidates."
        )

    return LayoutRuntimeCandidate(
        path=candidate_path,
        schema_version=schema_version,
        profile_id=profile_id,
        camera_adjust=camera_adjust,
        left_pair=left_pair,
        right_pair=right_pair,
        output_width_px=output_width,
        output_height_px=output_height,
        vertical_safety_enabled=vertical_enabled,
        vertical_safe_ratio=vertical_safe_ratio,
        side_vertical_fade_px=side_vertical_fade,
        pair_candidates=pair_candidates,
        calibration_hash=calibration_hash,
        current_calibration_hash=current_hash,
        projection=projection,
        warnings=tuple(warnings),
        raw=data,
    )


def _parse_projection(
    data: dict[str, Any],
    schema_version: int,
    candidate_dir: Path,
) -> LayoutRuntimeProjection:
    if schema_version == 2:
        return LayoutRuntimeProjection(source=ProjectionSource.CURRENT_PERSPECTIVE)
    raw = data.get("projection")
    if not isinstance(raw, dict):
        raise LayoutCandidateRuntimeError("schema_version 3 requires projection block.")
    source = _projection_source(raw.get("source"))
    if source == ProjectionSource.EQUIRECTANGULAR_CANDIDATE:
        raise LayoutCandidateRuntimeError("Equirectangular projection remains research-only.")
    projection_values = {"balance": 0.6, "fov_scale": 1.0, **raw}
    balance = _bounded_float(projection_values, "balance", 0.0, 1.0, "projection.balance")
    fov_scale = _bounded_float(projection_values, "fov_scale", 0.8, 1.2, "projection.fov_scale")
    pipeline = str(
        raw.get("projection_pipeline")
        or (
            "current_template_perspective_warp"
            if source == ProjectionSource.CURRENT_PERSPECTIVE
            else "fisheye_rectilinear_then_template_perspective_warp"
        )
    )
    intrinsics_path = None
    if source == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE:
        raw_path = raw.get("intrinsics_source_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise LayoutCandidateRuntimeError(
                "projection.intrinsics_source_path is required for fisheye_rectilinear."
            )
        intrinsics_path = Path(raw_path)
        if not intrinsics_path.is_absolute():
            intrinsics_path = (candidate_dir / intrinsics_path).resolve()
        try:
            load_fisheye_intrinsics_source(intrinsics_path)
        except FisheyeIntrinsicsRuntimeError as exc:
            raise LayoutCandidateRuntimeError(
                f"projection.intrinsics_source_path is not usable: {exc}"
            ) from exc
    return LayoutRuntimeProjection(
        source=source,
        intrinsics_source_path=intrinsics_path,
        balance=balance,
        fov_scale=fov_scale,
        projection_pipeline=pipeline,
        raw=dict(raw),
    )


def _projection_source(value: Any) -> ProjectionSource:
    text = str(value or "current_perspective")
    if text in {"current_perspective", ProjectionSource.CURRENT_PERSPECTIVE.value}:
        return ProjectionSource.CURRENT_PERSPECTIVE
    if text in {
        "fisheye_rectilinear",
        ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
    }:
        return ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE
    if text in {"equirectangular", ProjectionSource.EQUIRECTANGULAR_CANDIDATE.value}:
        return ProjectionSource.EQUIRECTANGULAR_CANDIDATE
    raise LayoutCandidateRuntimeError(f"Unsupported projection.source: {text}")


def _candidate_yaml_path(path: Path) -> Path:
    return path if path.name == "candidate.yaml" else path / "candidate.yaml"


def _parse_camera_adjust(raw: dict[str, Any]) -> dict[str, CameraAdjustParams]:
    parsed: dict[str, CameraAdjustParams] = {}
    for camera in ("front_left", "front", "front_right"):
        values = raw.get(camera)
        if not isinstance(values, dict):
            raise LayoutCandidateRuntimeError(f"camera_adjust.{camera} is required.")
        parsed[camera] = CameraAdjustParams(
            x_offset_px=_bounded_int(values, "x_offset_px", -240, 240, f"camera_adjust.{camera}.x_offset_px"),
            y_offset_px=_bounded_int(values, "y_offset_px", -160, 160, f"camera_adjust.{camera}.y_offset_px"),
            scale=_bounded_float(values, "scale", 0.8, 1.2, f"camera_adjust.{camera}.scale"),
        )
    return parsed


def _parse_pair(raw: dict[str, Any], label: str) -> PairLayoutParams:
    return PairLayoutParams(
        side_shift_px=_bounded_int(raw, "side_shift_px", 0, 240, f"{label}.side_shift_px"),
        side_visible_fraction=_bounded_float(
            raw,
            "side_visible_fraction",
            0.0,
            0.75,
            f"{label}.side_visible_fraction",
        ),
        feather_width_px=_bounded_int(raw, "feather_width_px", 0, 96, f"{label}.feather_width_px"),
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise LayoutCandidateRuntimeError(f"{key} is required.")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise LayoutCandidateRuntimeError(f"{key} must be an integer.")
    return int(value)


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise LayoutCandidateRuntimeError(f"{key} must be a non-empty string.")
    return value


def _required_bool(data: dict[str, Any], key: str, label: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise LayoutCandidateRuntimeError(f"{label} must be a boolean.")
    return value


def _profile_canvas_size(profile: dict[str, Any]) -> tuple[int, int]:
    canvas = profile.get("canvas")
    if not isinstance(canvas, dict):
        raise LayoutCandidateRuntimeError("Active profile canvas is missing.")
    width = canvas.get("width")
    height = canvas.get("height")
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or not math.isfinite(float(width))
        or not math.isfinite(float(height))
    ):
        raise LayoutCandidateRuntimeError("Active profile canvas dimensions must be finite.")
    width_px = int(round(float(width)))
    height_px = int(round(float(height)))
    if width_px <= 0 or height_px <= 0:
        raise LayoutCandidateRuntimeError("Active profile canvas dimensions must be positive.")
    return width_px, height_px


def _bounded_int(
    data: dict[str, Any],
    key: str,
    minimum: int,
    maximum: int,
    label: str,
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayoutCandidateRuntimeError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise LayoutCandidateRuntimeError(f"{label} must be finite.")
    result = int(round(numeric))
    if result < minimum or result > maximum:
        raise LayoutCandidateRuntimeError(
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
        raise LayoutCandidateRuntimeError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise LayoutCandidateRuntimeError(f"{label} must be finite.")
    if result < minimum or result > maximum:
        raise LayoutCandidateRuntimeError(
            f"{label}={result} is outside [{minimum}, {maximum}]."
        )
    return result
