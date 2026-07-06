"""Report-only fisheye projection candidates."""

from typing import TYPE_CHECKING

from .fisheye_projection import (
    EquirectangularParams,
    build_equirectangular_projection,
    build_rectilinear_projection,
)
from .intrinsics_candidate import (
    FisheyeCameraModel,
    IntrinsicsCandidateResult,
    consolidate_intrinsics_candidates,
    find_latest_calibration_candidate,
)
from .intrinsics_runtime_loader import (
    FisheyeCameraIntrinsics,
    FisheyeIntrinsicsRuntimeError,
    FisheyeIntrinsicsRuntimeSource,
    load_fisheye_intrinsics_source,
)
from .projection_runner import (
    ProjectionCandidateRunResult,
    generate_projection_candidate_preview,
)
from .runtime_providers import (
    B2CandidateProjectionProvider,
    B2_FAR_FIELD_PROJECTION_SOURCE,
    CurrentPerspectiveProjectionProvider,
    FisheyeRectilinearParams,
    FisheyeRectilinearProjectionProvider,
    ProjectionProvider,
    ProjectionResult,
    derive_nonzero_valid_masks,
)

if TYPE_CHECKING:
    from .runtime_ab_comparison import (
        RuntimeABComparisonResult,
        generate_runtime_ab_comparison,
    )

__all__ = [
    "B2CandidateProjectionProvider",
    "B2_FAR_FIELD_PROJECTION_SOURCE",
    "CurrentPerspectiveProjectionProvider",
    "EquirectangularParams",
    "FisheyeCameraModel",
    "FisheyeCameraIntrinsics",
    "FisheyeIntrinsicsRuntimeError",
    "FisheyeIntrinsicsRuntimeSource",
    "FisheyeRectilinearParams",
    "FisheyeRectilinearProjectionProvider",
    "IntrinsicsCandidateResult",
    "ProjectionProvider",
    "ProjectionCandidateRunResult",
    "ProjectionResult",
    "RuntimeABComparisonResult",
    "build_equirectangular_projection",
    "build_rectilinear_projection",
    "consolidate_intrinsics_candidates",
    "derive_nonzero_valid_masks",
    "find_latest_calibration_candidate",
    "generate_projection_candidate_preview",
    "generate_runtime_ab_comparison",
    "load_fisheye_intrinsics_source",
]


def __getattr__(name: str):
    if name in {"RuntimeABComparisonResult", "generate_runtime_ab_comparison"}:
        from .runtime_ab_comparison import (
            RuntimeABComparisonResult,
            generate_runtime_ab_comparison,
        )

        values = {
            "RuntimeABComparisonResult": RuntimeABComparisonResult,
            "generate_runtime_ab_comparison": generate_runtime_ab_comparison,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
