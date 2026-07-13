"""Task-facing runtime view presets for the GUI.

The processing layer intentionally keeps separate mode, projection, custom
layout, and worker-strategy fields.  Presenting those fields independently in
the realtime workspace made several valid combinations look like unrelated
switches.  This module is the small UI coordinator that maps the five supported
user tasks onto the existing runtime contract without changing that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)


class RuntimeViewPreset(Enum):
    FAR_DEFAULT = "far_default"
    B2_VIEW = "b2_view"
    FAR_CUSTOM = "far_custom"
    NEAR_CURRENT = "near_current"
    NEAR_FISHEYE = "near_fisheye"


@dataclass(frozen=True)
class RuntimeViewSelection:
    preset: RuntimeViewPreset
    mode: StitchRuntimeMode
    projection_source: ProjectionSource
    use_far_field_custom_layout: bool
    live_stitch_strategy: str
    requires_b2_candidate: bool = False
    requires_far_layout_candidate: bool = False
    requires_near_layout_candidate: bool = False
    requires_fisheye_intrinsics: bool = False


_SELECTIONS = {
    RuntimeViewPreset.FAR_DEFAULT: RuntimeViewSelection(
        preset=RuntimeViewPreset.FAR_DEFAULT,
        mode=StitchRuntimeMode.FAR_FIELD,
        projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
        use_far_field_custom_layout=False,
        live_stitch_strategy="template",
    ),
    RuntimeViewPreset.B2_VIEW: RuntimeViewSelection(
        preset=RuntimeViewPreset.B2_VIEW,
        mode=StitchRuntimeMode.FAR_FIELD,
        projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
        use_far_field_custom_layout=False,
        live_stitch_strategy="candidate",
        requires_b2_candidate=True,
    ),
    RuntimeViewPreset.FAR_CUSTOM: RuntimeViewSelection(
        preset=RuntimeViewPreset.FAR_CUSTOM,
        mode=StitchRuntimeMode.FAR_FIELD,
        projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
        use_far_field_custom_layout=True,
        live_stitch_strategy="template",
        requires_b2_candidate=True,
        requires_far_layout_candidate=True,
    ),
    RuntimeViewPreset.NEAR_CURRENT: RuntimeViewSelection(
        preset=RuntimeViewPreset.NEAR_CURRENT,
        mode=StitchRuntimeMode.NEAR_FIELD,
        projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
        use_far_field_custom_layout=False,
        live_stitch_strategy="template",
        requires_near_layout_candidate=True,
    ),
    RuntimeViewPreset.NEAR_FISHEYE: RuntimeViewSelection(
        preset=RuntimeViewPreset.NEAR_FISHEYE,
        mode=StitchRuntimeMode.NEAR_FIELD,
        projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
        use_far_field_custom_layout=False,
        live_stitch_strategy="template",
        requires_near_layout_candidate=True,
        requires_fisheye_intrinsics=True,
    ),
}


def selection_for_preset(
    preset: RuntimeViewPreset | str,
) -> RuntimeViewSelection:
    if not isinstance(preset, RuntimeViewPreset):
        preset = RuntimeViewPreset(str(preset))
    return _SELECTIONS[preset]


def preset_for_runtime(
    config: RuntimeStitchConfig,
    live_stitch_strategy: str,
) -> RuntimeViewPreset:
    normalized = config.normalized()
    if (
        live_stitch_strategy == "candidate"
        and normalized.mode == StitchRuntimeMode.FAR_FIELD
        and not normalized.use_far_field_custom_layout
    ):
        return RuntimeViewPreset.B2_VIEW
    if normalized.mode == StitchRuntimeMode.NEAR_FIELD:
        if (
            normalized.projection_source
            == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE
        ):
            return RuntimeViewPreset.NEAR_FISHEYE
        return RuntimeViewPreset.NEAR_CURRENT
    if normalized.use_far_field_custom_layout:
        return RuntimeViewPreset.FAR_CUSTOM
    return RuntimeViewPreset.FAR_DEFAULT


def preset_for_effective_status(status: str) -> RuntimeViewPreset:
    mapping = {
        "far_field_default": RuntimeViewPreset.FAR_DEFAULT,
        "auto_fallback_far_field": RuntimeViewPreset.FAR_DEFAULT,
        "b2_candidate_view": RuntimeViewPreset.B2_VIEW,
        "far_field_custom": RuntimeViewPreset.FAR_CUSTOM,
        "near_field_current_perspective": RuntimeViewPreset.NEAR_CURRENT,
        "near_field_fisheye_rectilinear": RuntimeViewPreset.NEAR_FISHEYE,
    }
    return mapping.get(str(status), RuntimeViewPreset.FAR_DEFAULT)


__all__ = [
    "RuntimeViewPreset",
    "RuntimeViewSelection",
    "preset_for_effective_status",
    "preset_for_runtime",
    "selection_for_preset",
]
