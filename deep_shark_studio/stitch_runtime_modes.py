"""Runtime stitching mode definitions.

These modes describe the stitching algorithm used after the user enters the
Stitched view. They are intentionally separate from GUI layout modes such as
Grid, Focus, and Stitched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StitchRuntimeMode(Enum):
    FAR_FIELD = "far_field"
    NEAR_FIELD = "near_field"
    AUTO = "auto"


class ProjectionSource(Enum):
    CURRENT_PERSPECTIVE = "current_perspective"
    FISHEYE_RECTILINEAR_CANDIDATE = "fisheye_rectilinear_candidate"
    EQUIRECTANGULAR_CANDIDATE = "equirectangular_candidate"


@dataclass(frozen=True)
class RuntimeStitchConfig:
    mode: StitchRuntimeMode = StitchRuntimeMode.FAR_FIELD
    projection_source: ProjectionSource = ProjectionSource.CURRENT_PERSPECTIVE
    layout_candidate_path: Path | None = None
    use_far_field_custom_layout: bool = False
    far_field_layout_candidate_path: Path | None = None
    projection_intrinsics_source_path: Path | None = None
    fisheye_balance: float = 0.6
    fisheye_fov_scale: float = 1.0
    auto_enabled: bool = False

    def normalized(self) -> "RuntimeStitchConfig":
        return RuntimeStitchConfig(
            mode=coerce_stitch_runtime_mode(self.mode),
            projection_source=coerce_projection_source(self.projection_source),
            layout_candidate_path=(
                Path(self.layout_candidate_path)
                if self.layout_candidate_path is not None
                else None
            ),
            use_far_field_custom_layout=bool(self.use_far_field_custom_layout),
            far_field_layout_candidate_path=(
                Path(self.far_field_layout_candidate_path)
                if self.far_field_layout_candidate_path is not None
                else None
            ),
            projection_intrinsics_source_path=(
                Path(self.projection_intrinsics_source_path)
                if self.projection_intrinsics_source_path is not None
                else None
            ),
            fisheye_balance=float(self.fisheye_balance),
            fisheye_fov_scale=float(self.fisheye_fov_scale),
            auto_enabled=bool(self.auto_enabled),
        )

    def runtime_warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.projection_source == ProjectionSource.EQUIRECTANGULAR_CANDIDATE:
            warnings.append(
                "Projection candidate is research-only and not connected to runtime yet."
            )
        if (
            self.projection_source == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE
            and self.projection_intrinsics_source_path is None
        ):
            warnings.append("Fisheye Rectilinear projection requires a loaded fisheye intrinsics source.")
        if self.mode == StitchRuntimeMode.NEAR_FIELD and self.layout_candidate_path is None:
            warnings.append("Near-field mode requires a loaded Layout Candidate V2.")
        if (
            self.mode == StitchRuntimeMode.FAR_FIELD
            and self.use_far_field_custom_layout
            and self.far_field_layout_candidate_path is None
        ):
            warnings.append("Far-field custom layout requires a loaded Far-field Layout Candidate.")
        if self.mode == StitchRuntimeMode.AUTO and not self.auto_enabled:
            warnings.append("Auto stitch runtime mode is not implemented; falling back to Far-field.")
        return warnings

    def can_enter_near_field(self) -> bool:
        return (
            self.mode == StitchRuntimeMode.NEAR_FIELD
            and self.projection_source
            in {
                ProjectionSource.CURRENT_PERSPECTIVE,
                ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
            }
            and self.layout_candidate_path is not None
            and (
                self.projection_source == ProjectionSource.CURRENT_PERSPECTIVE
                or self.projection_intrinsics_source_path is not None
            )
        )


def coerce_stitch_runtime_mode(value: StitchRuntimeMode | str) -> StitchRuntimeMode:
    if isinstance(value, StitchRuntimeMode):
        return value
    return StitchRuntimeMode(str(value))


def coerce_projection_source(value: ProjectionSource | str) -> ProjectionSource:
    if isinstance(value, ProjectionSource):
        return value
    return ProjectionSource(str(value))


def resolve_effective_runtime_status(
    runtime_config: RuntimeStitchConfig,
    *,
    processor_mode: str = "template",
) -> str:
    """Return the concrete processing path selected by a worker configuration."""
    if processor_mode == "candidate":
        return "b2_candidate_view"

    config = runtime_config.normalized()
    if config.mode == StitchRuntimeMode.AUTO:
        return "auto_fallback_far_field"
    if config.mode == StitchRuntimeMode.NEAR_FIELD:
        if (
            config.projection_source
            == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE
        ):
            return "near_field_fisheye_rectilinear"
        return "near_field_current_perspective"
    if config.use_far_field_custom_layout:
        return "far_field_custom"
    return "far_field_default"
