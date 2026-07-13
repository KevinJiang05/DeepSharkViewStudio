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
    blend_mode = _effective_blend_mode(candidate, projection_source)
    use_b2_selection = blend_mode == "b2_weight_selection"
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
            precomputed_selection_masks=projection_result.metadata.get(
                "_b2_selection_masks",
                {},
            ),
        )
        composition_mode = blend_mode
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
            valid_masks=adjusted.valid_masks,
        )
        composition_mode = blend_mode
    composition_ms = _elapsed_ms(compose_start)

    crop_start = time.perf_counter()
    image, crop_x, crop_y = _center_crop(
        full_canvas,
        candidate.output_width_px,
        candidate.output_height_px,
    )
    crop_ms = _elapsed_ms(crop_start)
    full_valid = np.zeros(full_canvas.shape[:2], dtype=bool)
    for mask in adjusted.valid_masks.values():
        full_valid |= np.asarray(mask, dtype=bool)
    cropped_valid, _mask_crop_x, _mask_crop_y = _center_crop(
        full_valid,
        candidate.output_width_px,
        candidate.output_height_px,
    )
    valid = np.asarray(cropped_valid, dtype=bool)
    black_pixels = np.all(image == 0, axis=2)
    valid_count = int(np.count_nonzero(valid))
    metrics = {
        "runtime_mode": "far_field_custom",
        "candidate_path": str(candidate.path),
        "candidate_profile_id": candidate.profile_id,
        "candidate_warnings": list(candidate.warnings),
        "crop_x": list(crop_x),
        "crop_y": list(crop_y),
        "black_pixel_ratio": float(np.count_nonzero(black_pixels) / max(1, black_pixels.size)),
        "valid_pixel_ratio": float(valid_count / max(1, valid.size)),
        "geometric_hole_ratio": float(1.0 - valid_count / max(1, valid.size)),
        "black_content_ratio": float(
            np.count_nonzero(black_pixels & valid) / max(1, valid_count)
        ),
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


def _effective_blend_mode(
    candidate: FarFieldLayoutRuntimeCandidate,
    actual_projection_source: str,
) -> str:
    expected_by_projection = {
        "current_perspective": "current_horizontal_feather",
        B2_FAR_FIELD_PROJECTION_SOURCE: "b2_weight_selection",
    }
    actual_blend = expected_by_projection.get(actual_projection_source)
    if actual_blend is None:
        raise ValueError(
            f"Unsupported Far-field custom projection source: {actual_projection_source!r}."
        )
    if candidate.blend_mode is None:
        # Layout Tuner preview objects predate the persisted blend field. They
        # are ephemeral and may infer composition from their cached projection.
        return actual_blend
    candidate_projection_source = str(candidate.projection_source)
    expected_candidate_blend = expected_by_projection.get(candidate_projection_source)
    if expected_candidate_blend != candidate.blend_mode:
        raise ValueError(
            f"Candidate projection source {candidate_projection_source!r} is incompatible "
            f"with blend mode {candidate.blend_mode!r}."
        )
    if actual_projection_source != candidate_projection_source:
        raise ValueError(
            f"Runtime projection source {actual_projection_source!r} does not match "
            f"candidate projection source {candidate_projection_source!r}."
        )
    return candidate.blend_mode


def _compose_b2_weight_selection(
    adjusted: AdjustedWarpResult,
    weight_maps: Any,
    output_width: int,
    output_height: int,
    *,
    precomputed_selection_masks: Any = None,
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
    if _can_use_precomputed_b2_selection(
        adjusted,
        available,
        precomputed_selection_masks,
        output_width,
        output_height,
    ):
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        for camera in available:
            selection = np.asarray(
                precomputed_selection_masks[camera],
                dtype=bool,
            )
            # The plan is generated from the B-2 validity masks.  Intersect it
            # with the current explicit mask as a fail-safe for custom
            # providers while retaining valid black source pixels.
            selection = selection & adjusted.valid_masks[camera]
            cv2.copyTo(
                adjusted.warped_images[camera],
                np.ascontiguousarray(selection, dtype=np.uint8),
                canvas,
            )
        return canvas
    adjusted_weights: list[np.ndarray] = []
    for camera in available:
        image = adjusted.warped_images[camera]
        if image.shape[:2] != (output_height, output_width):
            raise ValueError(f"Warped image size mismatch for {camera}")
        weight = np.asarray(weight_maps[camera], dtype=np.float32)
        if weight.shape != image.shape[:2]:
            raise ValueError(f"B-2 weight map size mismatch for {camera}")
        finite_weight = np.nan_to_num(
            weight,
            copy=True,
            nan=-1.0e9,
            posinf=-1.0e9,
            neginf=-1.0e9,
        )
        matrix = adjusted.transforms[camera]
        warped_weight = cv2.warpAffine(
            finite_weight,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0e9,
        )
        warped_weight = np.nan_to_num(
            warped_weight,
            copy=False,
            nan=-1.0e9,
            posinf=-1.0e9,
            neginf=-1.0e9,
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


def _can_use_precomputed_b2_selection(
    adjusted: AdjustedWarpResult,
    available: list[str],
    selection_masks: Any,
    output_width: int,
    output_height: int,
) -> bool:
    if not isinstance(selection_masks, Mapping):
        return False
    identity = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    expected_shape = (output_height, output_width)
    for camera in available:
        if camera not in selection_masks:
            return False
        if np.asarray(selection_masks[camera]).shape != expected_shape:
            return False
        transform = np.asarray(adjusted.transforms.get(camera), dtype=np.float64)
        if transform.shape != identity.shape or not np.array_equal(transform, identity):
            return False
    return True


def _center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    height, width = image.shape[:2]
    crop_width = int(target_width)
    crop_height = int(target_height)
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            f"Far-field custom crop must be positive; got {crop_width}x{crop_height}."
        )
    if crop_width > width or crop_height > height:
        raise ValueError(
            f"Far-field custom crop {crop_width}x{crop_height} exceeds source canvas "
            f"{width}x{height}."
        )
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    x1 = x0 + crop_width
    y1 = y0 + crop_height
    return image[y0:y1, x0:x1].copy(), (x0, x1), (y0, y1)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
