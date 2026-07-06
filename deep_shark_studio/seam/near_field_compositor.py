"""Near-field runtime compositor based on saved Layout Candidate V2 data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .front_priority_compositor_core import render_front_priority_layout_core
from .layout_candidate_runtime import LayoutRuntimeCandidate


@dataclass(frozen=True)
class NearFieldRenderResult:
    canvas: np.ndarray
    metrics: dict[str, Any]
    debug: dict[str, Any] | None = None


def render_near_field_from_warped(
    warped_images: dict[str, np.ndarray],
    candidate: LayoutRuntimeCandidate,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> NearFieldRenderResult:
    required = {"front_left", "front", "front_right"}
    missing = sorted(camera for camera in required if camera not in warped_images)
    if missing:
        raise ValueError(
            "Near-field runtime requires warped images for: " + ", ".join(missing)
        )
    normalized_masks = _normalize_valid_masks(warped_images, valid_masks, required)
    core = render_front_priority_layout_core(
        warped_images,
        candidate.pair_candidates,
        candidate.to_layout_params(),
        valid_masks=normalized_masks,
    )
    metrics = dict(core.metrics)
    metrics.update(
        {
            "runtime_mode": "near_field",
            "layout_id": core.layout_id,
            "candidate_path": str(candidate.path),
            "candidate_profile_id": candidate.profile_id,
            "candidate_warnings": list(candidate.warnings),
            "valid_mask_source": (
                "provider_valid_masks"
                if normalized_masks is not None
                else "nonzero_pixels_runtime_compatible"
            ),
        }
    )
    debug = {
        "crop_x": list(core.crop_x),
        "crop_y": list(core.crop_y),
        "vertical_safety_enabled": bool(core.vertical_safety_enabled),
        "camera_adjust_metadata": core.adjusted.metadata,
    }
    return NearFieldRenderResult(
        canvas=core.image,
        metrics=metrics,
        debug=debug,
    )


def _normalize_valid_masks(
    warped_images: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray] | None,
    required: set[str],
) -> dict[str, np.ndarray] | None:
    if valid_masks is None:
        return None
    extra = sorted(set(valid_masks) - set(warped_images))
    if extra:
        raise ValueError("valid_masks contains unknown cameras: " + ", ".join(extra))
    missing = sorted(camera for camera in required if camera not in valid_masks)
    if missing:
        raise ValueError("valid_masks missing required cameras: " + ", ".join(missing))
    normalized: dict[str, np.ndarray] = {}
    for camera, mask in valid_masks.items():
        image = warped_images[camera]
        mask_array = np.asarray(mask)
        if mask_array.shape != image.shape[:2]:
            raise ValueError(f"valid_mask size mismatch for {camera}")
        normalized[camera] = mask_array.astype(bool)
    return normalized
