"""Near-field runtime compositor based on saved Layout Candidate V2 data."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from threading import RLock
import time
from typing import Any

import numpy as np

from .front_priority_compositor_core import (
    RuntimeNearFieldPlan,
    build_runtime_near_field_plan,
    render_front_priority_layout_core,
)
from .layout_candidate_runtime import LayoutRuntimeCandidate


@dataclass(frozen=True)
class NearFieldRenderResult:
    canvas: np.ndarray
    metrics: dict[str, Any]
    debug: dict[str, Any] | None = None


_REQUIRED_CAMERAS = ("front_left", "front", "front_right")
_PLAN_CACHE_MAX_ENTRIES = 2
_PLAN_CACHE_MAX_MEMORY_BYTES = 128 * 1024 * 1024
_PLAN_CACHE: OrderedDict[tuple[Any, ...], RuntimeNearFieldPlan] = OrderedDict()
_PLAN_CACHE_LOCK = RLock()
_PLAN_CACHE_HITS = 0
_PLAN_CACHE_MISSES = 0
_PLAN_CACHE_EVICTIONS = 0
_PLAN_CACHE_BYPASSES = 0
_PLAN_CACHE_OVERSIZE_REJECTIONS = 0


def render_near_field_from_warped(
    warped_images: dict[str, np.ndarray],
    candidate: LayoutRuntimeCandidate,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> NearFieldRenderResult:
    render_start = time.perf_counter()
    required = set(_REQUIRED_CAMERAS)
    missing = sorted(camera for camera in required if camera not in warped_images)
    if missing:
        raise ValueError(
            "Near-field runtime requires warped images for: " + ", ".join(missing)
        )
    normalized_masks = _normalize_valid_masks(warped_images, valid_masks, required)
    cache_key = _runtime_plan_cache_key(
        warped_images,
        candidate,
        normalized_masks,
    )
    plan = _cached_plan(cache_key)
    plan_cache_hit = plan is not None
    core = render_front_priority_layout_core(
        warped_images,
        candidate.pair_candidates,
        candidate.to_layout_params(),
        valid_masks=normalized_masks,
        plan=plan,
    )
    plan_build_ms = 0.0
    plan_build_failed = False
    if cache_key is not None and plan is None:
        build_start = time.perf_counter()
        try:
            plan = build_runtime_near_field_plan(
                core.adjusted,
                candidate.pair_candidates,
                candidate.to_layout_params(),
            )
        except (AttributeError, TypeError, ValueError):
            # Plan creation is an optional optimization. A valid legacy render
            # must remain usable if an unusual input cannot be planned safely.
            plan_build_failed = True
        else:
            _store_plan(cache_key, plan)
        plan_build_ms = (time.perf_counter() - build_start) * 1000.0
    metrics = dict(core.metrics)
    timing = dict(metrics.get("timing", {}))
    near_field_core_ms = float(timing.get("near_field_total_ms", 0.0))
    timing.update(
        {
            "near_field_core_ms": near_field_core_ms,
            "near_field_total_ms": (time.perf_counter() - render_start) * 1000.0,
            "runtime_plan_cache_hit": plan_cache_hit,
            "runtime_plan_build_ms": plan_build_ms,
            "runtime_plan_build_failed": plan_build_failed,
        }
    )
    metrics["timing"] = timing
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
        normalized[camera] = mask_array.astype(bool, copy=False)
    return normalized


def clear_near_field_plan_cache() -> None:
    """Clear bounded process-local runtime geometry plans and counters."""
    global _PLAN_CACHE_HITS
    global _PLAN_CACHE_MISSES
    global _PLAN_CACHE_EVICTIONS
    global _PLAN_CACHE_BYPASSES
    global _PLAN_CACHE_OVERSIZE_REJECTIONS
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE.clear()
        _PLAN_CACHE_HITS = 0
        _PLAN_CACHE_MISSES = 0
        _PLAN_CACHE_EVICTIONS = 0
        _PLAN_CACHE_BYPASSES = 0
        _PLAN_CACHE_OVERSIZE_REJECTIONS = 0


def near_field_plan_cache_info() -> dict[str, int]:
    with _PLAN_CACHE_LOCK:
        return {
            "entries": len(_PLAN_CACHE),
            "max_entries": _PLAN_CACHE_MAX_ENTRIES,
            "max_memory_bytes": _PLAN_CACHE_MAX_MEMORY_BYTES,
            "hits": _PLAN_CACHE_HITS,
            "misses": _PLAN_CACHE_MISSES,
            "evictions": _PLAN_CACHE_EVICTIONS,
            "bypasses": _PLAN_CACHE_BYPASSES,
            "oversize_rejections": _PLAN_CACHE_OVERSIZE_REJECTIONS,
            "memory_bytes": int(
                sum(plan.memory_bytes for plan in _PLAN_CACHE.values())
            ),
        }


def _cached_plan(
    key: tuple[Any, ...] | None,
) -> RuntimeNearFieldPlan | None:
    global _PLAN_CACHE_HITS
    global _PLAN_CACHE_MISSES
    global _PLAN_CACHE_BYPASSES
    with _PLAN_CACHE_LOCK:
        if key is None:
            _PLAN_CACHE_BYPASSES += 1
            return None
        plan = _PLAN_CACHE.pop(key, None)
        if plan is None:
            _PLAN_CACHE_MISSES += 1
            return None
        _PLAN_CACHE[key] = plan
        _PLAN_CACHE_HITS += 1
        return plan


def _store_plan(key: tuple[Any, ...], plan: RuntimeNearFieldPlan) -> None:
    global _PLAN_CACHE_EVICTIONS
    global _PLAN_CACHE_OVERSIZE_REJECTIONS
    with _PLAN_CACHE_LOCK:
        if plan.memory_bytes > _PLAN_CACHE_MAX_MEMORY_BYTES:
            _PLAN_CACHE_OVERSIZE_REJECTIONS += 1
            return
        _PLAN_CACHE.pop(key, None)
        _PLAN_CACHE[key] = plan
        while (
            len(_PLAN_CACHE) > _PLAN_CACHE_MAX_ENTRIES
            or sum(item.memory_bytes for item in _PLAN_CACHE.values())
            > _PLAN_CACHE_MAX_MEMORY_BYTES
        ):
            _PLAN_CACHE.popitem(last=False)
            _PLAN_CACHE_EVICTIONS += 1


def _runtime_plan_cache_key(
    warped_images: dict[str, np.ndarray],
    candidate: LayoutRuntimeCandidate,
    valid_masks: dict[str, np.ndarray] | None,
) -> tuple[Any, ...] | None:
    """Key geometry only; frame pixels are deliberately not cached.

    Candidate layout/adjustment changes, canvas spec changes, and any explicit
    mask-content change produce a miss. Inputs without explicit provider masks
    stay on the legacy path because their validity depends on frame pixels.
    """
    if valid_masks is None:
        return None
    if set(warped_images) != set(_REQUIRED_CAMERAS):
        return None
    image_specs: list[tuple[str, tuple[int, ...], str]] = []
    mask_specs: list[tuple[str, tuple[int, ...], bytes]] = []
    reference_shape: tuple[int, ...] | None = None
    for camera in _REQUIRED_CAMERAS:
        image = np.asarray(warped_images[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            return None
        if reference_shape is None:
            reference_shape = image.shape
        elif image.shape != reference_shape:
            return None
        mask = np.asarray(valid_masks[camera], dtype=bool)
        image_specs.append((camera, tuple(image.shape), image.dtype.str))
        mask_specs.append((camera, tuple(mask.shape), _mask_digest(mask)))
    return (
        _candidate_plan_signature(candidate),
        tuple(image_specs),
        tuple(mask_specs),
    )


def _candidate_plan_signature(candidate: LayoutRuntimeCandidate) -> tuple[Any, ...]:
    camera_adjust = tuple(
        (
            camera,
            int(params.x_offset_px),
            int(params.y_offset_px),
            float(params.scale),
        )
        for camera, params in sorted(candidate.camera_adjust.items())
    )
    pair_geometry = tuple(
        (
            pair.pair_id,
            pair.side_camera,
            pair.side_position,
            pair.boundary_type,
            tuple((int(x), int(y)) for x, y in pair.seam_points),
        )
        for pair in candidate.pair_candidates
    )
    return (
        camera_adjust,
        (
            int(candidate.left_pair.side_shift_px),
            float(candidate.left_pair.side_visible_fraction),
            int(candidate.left_pair.feather_width_px),
        ),
        (
            int(candidate.right_pair.side_shift_px),
            float(candidate.right_pair.side_visible_fraction),
            int(candidate.right_pair.feather_width_px),
        ),
        int(candidate.output_width_px),
        int(candidate.output_height_px),
        bool(candidate.vertical_safety_enabled),
        float(candidate.vertical_safe_ratio),
        int(candidate.side_vertical_fade_px),
        pair_geometry,
    )


def _mask_digest(mask: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(mask, dtype=bool)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.digest()
