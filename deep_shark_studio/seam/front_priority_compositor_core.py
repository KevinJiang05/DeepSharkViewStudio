"""Label-free front-priority compositor core for near-field runtime."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .layout_preview import (
    FrontPriorityLayoutRuntimePlan,
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
    LayoutPreviewResult,
)
from deep_shark_studio.stitching.camera_layout_adjust import (
    AdjustedWarpResult,
    apply_post_warp_camera_adjustments,
    identity_camera_adjust,
)
from .layout_tuner import LayoutPreviewParamsV2
from .vertical_safety import (
    VerticalSafetyParams,
    VerticalSafetyRenderer,
    VerticalSafetyResult,
    VerticalSafetyRuntimePlan,
)


@dataclass(frozen=True)
class FrontPriorityCoreResult:
    image: np.ndarray
    side_visible_mask: np.ndarray
    front_preserved_mask: np.ndarray
    side_suppressed_mask: np.ndarray
    metrics: dict[str, Any]
    crop_x: tuple[int, int]
    crop_y: tuple[int, int]
    layout_id: str
    vertical_safety_enabled: bool
    adjusted: AdjustedWarpResult


@dataclass(frozen=True)
class RuntimeNearFieldPlan:
    """Internal geometry plan guarded by near_field_compositor's full key."""

    layout: FrontPriorityLayoutRuntimePlan
    vertical: VerticalSafetyRuntimePlan | None = None

    @property
    def memory_bytes(self) -> int:
        if self.vertical is not None:
            return int(self.vertical.memory_bytes)
        return int(self.layout.memory_bytes)


def render_front_priority_layout_core(
    warped_images: dict[str, np.ndarray],
    pair_candidates: list[LayoutPairCandidate],
    params: LayoutPreviewParamsV2,
    valid_masks: dict[str, np.ndarray] | None = None,
    plan: RuntimeNearFieldPlan | None = None,
) -> FrontPriorityCoreResult:
    """Render a clean runtime canvas without preview labels or overlays."""
    total_start = time.perf_counter()
    affine_start = time.perf_counter()
    adjusted = apply_post_warp_camera_adjustments(
        warped_images,
        valid_masks=valid_masks,
        camera_adjust=params.camera_adjust or identity_camera_adjust(),
    )
    post_warp_affine_ms = _elapsed_ms(affine_start)

    canvas_height = _front_canvas_height(adjusted.warped_images)
    layout_params = params.to_layout_params()
    pair_params = params.pair_layout_params()
    safety_params = params.to_vertical_params()
    use_vertical = params.vertical_safety_enabled(canvas_height)
    if plan is not None and use_vertical != (plan.vertical is not None):
        raise ValueError(
            "Runtime Near-field plan does not match vertical_safety parameters."
        )

    composition_start = time.perf_counter()
    if use_vertical:
        base_start = time.perf_counter()
        result = VerticalSafetyRenderer().render(
            adjusted.warped_images,
            pair_candidates,
            layout_params,
            safety_params,
            pair_params=pair_params,
            draw_label=False,
            valid_masks=adjusted.valid_masks,
            plan=plan.vertical if plan is not None else None,
        )
        composition_ms = _elapsed_ms(base_start)
        vertical_safety_ms = composition_ms
        core = _from_vertical_result(result, layout_params, params.layout_id, adjusted)
    else:
        result = FrontPriorityLayoutPreviewRenderer().render(
            adjusted.warped_images,
            pair_candidates,
            layout_params,
            pair_params=pair_params,
            draw_label=False,
            valid_masks=adjusted.valid_masks,
            plan=plan.layout if plan is not None else None,
        )
        composition_ms = _elapsed_ms(composition_start)
        vertical_safety_ms = 0.0
        core = _from_layout_result(result, params.layout_id, adjusted)

    metrics = dict(core.metrics)
    metrics["timing"] = {
        "near_field_total_ms": _elapsed_ms(total_start),
        "post_warp_affine_ms": post_warp_affine_ms,
        "composition_ms": composition_ms,
        "vertical_safety_ms": vertical_safety_ms,
        "runtime_plan_used": plan is not None,
    }
    return FrontPriorityCoreResult(
        image=core.image,
        side_visible_mask=core.side_visible_mask,
        front_preserved_mask=core.front_preserved_mask,
        side_suppressed_mask=core.side_suppressed_mask,
        metrics=metrics,
        crop_x=core.crop_x,
        crop_y=core.crop_y,
        layout_id=core.layout_id,
        vertical_safety_enabled=core.vertical_safety_enabled,
        adjusted=adjusted,
    )


def build_runtime_near_field_plan(
    adjusted: AdjustedWarpResult,
    pair_candidates: list[LayoutPairCandidate],
    params: LayoutPreviewParamsV2,
) -> RuntimeNearFieldPlan:
    """Build an internal plan from one already-rendered adjusted frame set.

    Callers must route reuse through ``near_field_compositor`` so candidate,
    seam, shape, and mask-content invalidation is enforced.
    """
    layout_params = params.to_layout_params()
    pair_params = params.pair_layout_params()
    layout_plan = FrontPriorityLayoutPreviewRenderer().build_runtime_plan(
        adjusted.warped_images,
        pair_candidates,
        layout_params,
        pair_params=pair_params,
        valid_masks=adjusted.valid_masks,
    )
    canvas_height = _front_canvas_height(adjusted.warped_images)
    vertical_plan = None
    if params.vertical_safety_enabled(canvas_height):
        vertical_plan = VerticalSafetyRenderer().build_runtime_plan(
            layout_plan,
            params.to_vertical_params(),
        )
    return RuntimeNearFieldPlan(
        layout=layout_plan,
        vertical=vertical_plan,
    )


def _from_layout_result(
    result: LayoutPreviewResult,
    layout_id: str,
    adjusted: AdjustedWarpResult,
) -> FrontPriorityCoreResult:
    return FrontPriorityCoreResult(
        image=result.image,
        side_visible_mask=result.side_visible_mask,
        front_preserved_mask=result.front_preserved_mask,
        side_suppressed_mask=result.side_suppressed_mask,
        metrics=dict(result.metrics),
        crop_x=result.crop,
        crop_y=(0, result.image.shape[0]),
        layout_id=layout_id,
        vertical_safety_enabled=False,
        adjusted=adjusted,
    )


def _from_vertical_result(
    result: VerticalSafetyResult,
    layout_params: LayoutPreviewParams,
    layout_id: str,
    adjusted: AdjustedWarpResult,
) -> FrontPriorityCoreResult:
    metrics = dict(result.metrics)
    metrics["base_layout_id"] = layout_params.layout_id
    return FrontPriorityCoreResult(
        image=result.image,
        side_visible_mask=result.side_visible_after,
        front_preserved_mask=result.front_preserved_mask,
        side_suppressed_mask=result.side_suppressed_top_bottom,
        metrics=metrics,
        crop_x=(0, result.image.shape[1]),
        crop_y=result.crop_y,
        layout_id=layout_id,
        vertical_safety_enabled=True,
        adjusted=adjusted,
    )


def _front_canvas_height(warped_images: dict[str, np.ndarray]) -> int | None:
    front = warped_images.get("front")
    return int(front.shape[0]) if front is not None else None


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
