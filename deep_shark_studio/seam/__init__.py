"""Static seam candidate generation for panorama overlap diagnostics."""

from .candidate import SeamCandidateStore
from .boundary import BoundarySearchParams, BoundarySearchResult, SideBoundaryCandidateFinder
from .cost import FrontPriorityCostParams, SeamCostBuilder, SeamCostParams, SeamCostResult
from .dp import DPSeamParams, SeamPath, VerticalDPSeamFinder
from .dynamic import DynamicBoundarySimulationResult, simulate_front_priority_dynamic_boundaries
from .front_priority_compositor_core import (
    FrontPriorityCoreResult,
    RuntimeNearFieldPlan,
    render_front_priority_layout_core,
)
from .far_field_custom_compositor import (
    FarFieldCustomRenderResult,
    render_far_field_custom_from_projection,
)
from .far_field_layout_candidate_runtime import (
    FarFieldLayoutCandidateError,
    FarFieldLayoutRuntimeCandidate,
    default_far_field_layout_candidate_root,
    load_far_field_layout_candidate,
    save_far_field_layout_candidate,
)
from .layout_preview import (
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
    LayoutPreviewResult,
)
from .layout_candidate_runtime import (
    LayoutCandidateRuntimeError,
    LayoutRuntimeProjection,
    LayoutRuntimeCandidate,
    load_layout_candidate_for_runtime,
)
from .layout_sweep import (
    LayoutSweepResult,
    load_layout_candidates_from_run,
    run_front_priority_layout_sweep,
)
from .layout_tuner import (
    LAYOUT_TUNER_PRESETS,
    LAYOUT_TUNER_PRESETS_V2,
    LayoutPreviewParamsV2,
    PairLayoutParams,
    LayoutTunerParams,
    LayoutTunerPreviewResult,
    default_layout_candidate_root,
    layout_tuner_pair_candidates,
    layout_tuner_params_v1_to_v2,
    normalize_layout_tuner_params,
    render_front_priority_layout_preview,
    save_layout_tuner_candidate,
)
from deep_shark_studio.stitching.camera_layout_adjust import (
    AdjustedWarpResult,
    CameraAdjustParams,
    apply_post_warp_camera_adjustments,
)
from .mask import SeamMaskBuilder, SeamMaskResult
from .near_field_compositor import NearFieldRenderResult, render_near_field_from_warped
from .projection_audit import write_projection_audit
from .projection_readiness_audit import write_projection_readiness_audit
from .preview import SeamPreviewRenderer
from .quality import FrontPriorityQualityGate, QualityGateParams, QualityGateResult
from .roles import PairRole, resolve_front_priority_pair_role
from .runner import (
    SeamCandidateRunResult,
    generate_front_priority_seam_candidates,
    generate_pairwise_seam_candidates,
)
from .vertical_safety import VerticalSafetyParams, VerticalSafetyRenderer, VerticalSafetyResult
from .vertical_safety_sweep import (
    VerticalSafetySweepResult,
    run_vertical_safety_sweep,
    run_vertical_safety_sweep_from_warped_images,
)

__all__ = [
    "DPSeamParams",
    "BoundarySearchParams",
    "BoundarySearchResult",
    "DynamicBoundarySimulationResult",
    "FrontPriorityCostParams",
    "FrontPriorityCoreResult",
    "FrontPriorityLayoutPreviewRenderer",
    "FrontPriorityQualityGate",
    "FarFieldCustomRenderResult",
    "FarFieldLayoutCandidateError",
    "FarFieldLayoutRuntimeCandidate",
    "LayoutPairCandidate",
    "LayoutCandidateRuntimeError",
    "LayoutPreviewParams",
    "LayoutPreviewParamsV2",
    "LayoutPreviewResult",
    "LayoutRuntimeProjection",
    "LayoutRuntimeCandidate",
    "LayoutSweepResult",
    "LayoutTunerParams",
    "LayoutTunerPreviewResult",
    "PairLayoutParams",
    "PairRole",
    "QualityGateParams",
    "QualityGateResult",
    "RuntimeNearFieldPlan",
    "SeamCandidateRunResult",
    "SeamCandidateStore",
    "SeamCostBuilder",
    "SeamCostParams",
    "SeamCostResult",
    "SeamMaskBuilder",
    "SeamMaskResult",
    "SeamPath",
    "SeamPreviewRenderer",
    "SideBoundaryCandidateFinder",
    "NearFieldRenderResult",
    "VerticalDPSeamFinder",
    "VerticalSafetyParams",
    "VerticalSafetyRenderer",
    "VerticalSafetyResult",
    "VerticalSafetySweepResult",
    "LAYOUT_TUNER_PRESETS",
    "LAYOUT_TUNER_PRESETS_V2",
    "AdjustedWarpResult",
    "CameraAdjustParams",
    "apply_post_warp_camera_adjustments",
    "default_layout_candidate_root",
    "default_far_field_layout_candidate_root",
    "generate_front_priority_seam_candidates",
    "generate_pairwise_seam_candidates",
    "layout_tuner_pair_candidates",
    "layout_tuner_params_v1_to_v2",
    "load_layout_candidates_from_run",
    "load_layout_candidate_for_runtime",
    "load_far_field_layout_candidate",
    "normalize_layout_tuner_params",
    "render_near_field_from_warped",
    "render_front_priority_layout_core",
    "render_front_priority_layout_preview",
    "render_far_field_custom_from_projection",
    "resolve_front_priority_pair_role",
    "run_front_priority_layout_sweep",
    "run_vertical_safety_sweep",
    "run_vertical_safety_sweep_from_warped_images",
    "save_layout_tuner_candidate",
    "save_far_field_layout_candidate",
    "simulate_front_priority_dynamic_boundaries",
    "write_projection_audit",
    "write_projection_readiness_audit",
]
