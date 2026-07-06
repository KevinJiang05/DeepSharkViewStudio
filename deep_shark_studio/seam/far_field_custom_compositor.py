"""Far-field custom compositor with post-warp camera layout adjustment."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import cv2
import numpy as np

from deep_shark_studio.projection.runtime_providers import (
    B2_FAR_FIELD_PROJECTION_SOURCE,
    ProjectionResult,
)
from deep_shark_studio.stitching.camera_layout_adjust import (
    AdjustedWarpResult,
    apply_post_warp_camera_adjustments,
)
from deep_shark_studio.topology import compose_horizontal_feather

from .far_field_layout_candidate_runtime import FarFieldLayoutRuntimeCandidate


@dataclass(frozen=True)
class FarFieldCustomRenderResult:
    image: np.ndarray
    metrics: dict[str, Any]
    crop_x: tuple[int, int]
    crop_y: tuple[int, int]
    adjusted: AdjustedWarpResult


def render_far_field_custom_from_projection(
    projection_result: ProjectionResult,
    candidate: FarFieldLayoutRuntimeCandidate,
    stitch_config: Mapping[str, Any] | Any,
) -> FarFieldCustomRenderResult:
    """Render Far-field style canvas after post-warp camera adjustment."""
    total_start = time.perf_counter()
    adjust_start = time.perf_counter()
    adjusted = apply_post_warp_camera_adjustments(
        projection_result.warped_images,
        valid_masks=projection_result.valid_masks,
        camera_adjust=candidate.camera_adjust,
    )
    camera_adjust_ms = _elapsed_ms(adjust_start)

    compose_start = time.perf_counter()
    projection_source = str(projection_result.metadata.get("projection_source", ""))
    use_b2_selection = projection_source == B2_FAR_FIELD_PROJECTION_SOURCE
    output_width = int(_config_value(stitch_config, "output_width", "canvas.width", 0))
    output_height = int(_config_value(stitch_config, "output_height", "canvas.height", 0))
    if use_b2_selection:
        output_height, output_width = _first_image_shape(adjusted.warped_images)
    if output_width <= 0 or output_height <= 0:
        output_height, output_width = _first_image_shape(adjusted.warped_images)
    if use_b2_selection:
        full_canvas = _compose_b2_weight_selection(
            adjusted,
            projection_result.metadata.get("_b2_weight_maps", {}),
            output_width,
            output_height,
        )
        composition_mode = "b2_weight_selection"
    else:
        overlaps = _overlaps(stitch_config, candidate.feather_width_override_px)
        stitch_points = _config_value(stitch_config, "stitch_points", "stitch_points", {})
        default_feather_width = _default_feather_width(
            stitch_config,
            candidate.feather_width_override_px,
        )
        full_canvas = compose_horizontal_feather(
            adjusted.warped_images,
            output_width,
            output_height,
            stitch_points,
            overlaps,
            default_feather_width,
        )
        composition_mode = "current_horizontal_feather"
    composition_ms = _elapsed_ms(compose_start)

    crop_start = time.perf_counter()
    image, crop_x, crop_y = _center_crop(
        full_canvas,
        candidate.output_width_px,
        candidate.output_height_px,
    )
    crop_ms = _elapsed_ms(crop_start)
    valid = np.any(image != 0, axis=2)
    metrics = {
        "runtime_mode": "far_field_custom",
        "candidate_path": str(candidate.path),
        "candidate_profile_id": candidate.profile_id,
        "candidate_warnings": list(candidate.warnings),
        "crop_x": list(crop_x),
        "crop_y": list(crop_y),
        "black_pixel_ratio": float(1.0 - np.count_nonzero(valid) / max(1, valid.size)),
        "valid_pixel_ratio": float(np.count_nonzero(valid) / max(1, valid.size)),
        "camera_adjust_metadata": adjusted.metadata,
        "projection": _public_projection_metadata(projection_result.metadata),
        "composition_mode": composition_mode,
        "timing": {
            "far_field_custom_total_ms": _elapsed_ms(total_start),
            "projection_total_ms": float(
                projection_result.timings.get("projection_total_ms", 0.0)
            ),
            "camera_adjust_ms": camera_adjust_ms,
            "far_field_composition_ms": composition_ms,
            "crop_ms": crop_ms,
        },
    }
    return FarFieldCustomRenderResult(
        image=image,
        metrics=metrics,
        crop_x=crop_x,
        crop_y=crop_y,
        adjusted=adjusted,
    )


def _config_value(config: Mapping[str, Any] | Any, attr: str, dotted: str, default: Any) -> Any:
    if hasattr(config, attr):
        return getattr(config, attr)
    if isinstance(config, Mapping):
        current: Any = config
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current
    return default


def _overlaps(
    config: Mapping[str, Any] | Any,
    feather_override: int | None,
) -> list[dict[str, Any]]:
    raw = _config_value(config, "overlaps", "overlaps", [])
    overlaps = [dict(item) for item in raw]
    if feather_override is not None:
        for overlap in overlaps:
            overlap["feather_width"] = int(feather_override)
    return overlaps


def _default_feather_width(
    config: Mapping[str, Any] | Any,
    feather_override: int | None,
) -> int:
    if feather_override is not None:
        return int(feather_override)
    if hasattr(config, "composition"):
        return int(getattr(config, "composition", {}).get("feather_width", 120))
    if isinstance(config, Mapping):
        return int(config.get("composition", {}).get("feather_width", 120))
    return 120


def _first_image_shape(images: dict[str, np.ndarray]) -> tuple[int, int]:
    if not images:
        return 0, 0
    first = next(iter(images.values()))
    height, width = first.shape[:2]
    return int(height), int(width)


def _public_projection_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in metadata.items() if not str(key).startswith("_")}


def _compose_b2_weight_selection(
    adjusted: AdjustedWarpResult,
    weight_maps: Any,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    if not isinstance(weight_maps, Mapping) or not weight_maps:
        raise ValueError("B-2 Far-field custom composition requires B-2 weight maps.")
    available = [
        camera
        for camera in ("front_left", "front", "front_right")
        if camera in adjusted.warped_images and camera in weight_maps
    ]
    if not available:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8)
    adjusted_weights: list[np.ndarray] = []
    for camera in available:
        image = adjusted.warped_images[camera]
        if image.shape[:2] != (output_height, output_width):
            raise ValueError(f"Warped image size mismatch for {camera}")
        weight = np.asarray(weight_maps[camera], dtype=np.float32)
        if weight.shape != image.shape[:2]:
            raise ValueError(f"B-2 weight map size mismatch for {camera}")
        matrix = adjusted.transforms[camera]
        warped_weight = cv2.warpAffine(
            weight,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0e9,
        )
        mask = adjusted.valid_masks[camera].astype(bool)
        warped_weight = np.where(mask, warped_weight, -np.inf)
        adjusted_weights.append(warped_weight)
    weights = np.stack(adjusted_weights, axis=0)
    selected = np.argmax(weights, axis=0)
    valid_any = np.any(np.isfinite(weights), axis=0)
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    for index, camera in enumerate(available):
        selection = valid_any & (selected == index)
        canvas[selection] = adjusted.warped_images[camera][selection]
    return canvas


def _center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    height, width = image.shape[:2]
    crop_width = min(max(1, int(target_width)), width)
    crop_height = min(max(1, int(target_height)), height)
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    return image[y0:y1, x0:x1].copy(), (x0, x1), (y0, y1)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
