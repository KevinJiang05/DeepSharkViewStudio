"""Runtime stitching controller for Far-field, Near-field, and Auto modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from .projection.runtime_providers import (
    B2CandidateProjectionProvider,
    B2_FAR_FIELD_PROJECTION_SOURCE,
    CurrentPerspectiveProjectionProvider,
    FisheyeRectilinearParams,
    FisheyeRectilinearProjectionProvider,
    ProjectionProvider,
)
from .projection.intrinsics_runtime_loader import load_fisheye_intrinsics_source
from .seam.layout_candidate_runtime import (
    LayoutRuntimeCandidate,
    load_layout_candidate_for_runtime,
)
from .seam.far_field_custom_compositor import render_far_field_custom_from_projection
from .seam.far_field_layout_candidate_runtime import (
    FarFieldLayoutRuntimeCandidate,
    load_far_field_layout_candidate,
)
from .seam.near_field_compositor import render_near_field_from_warped
from .stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
    resolve_effective_runtime_status,
)


@dataclass(frozen=True)
class RuntimeStitchResult:
    warped: dict[str, np.ndarray]
    canvas: np.ndarray | None
    mode: StitchRuntimeMode
    status: str
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] | None = None


class RuntimeStitchController:
    """Small wrapper that keeps runtime mode branching out of GUI code."""

    def __init__(
        self,
        stitcher: Any,
        config: RuntimeStitchConfig | None = None,
        layout_candidate: LayoutRuntimeCandidate | None = None,
        far_field_layout_candidate: FarFieldLayoutRuntimeCandidate | None = None,
        projection_provider: ProjectionProvider | None = None,
    ):
        self.stitcher = stitcher
        self.config = (config or RuntimeStitchConfig()).normalized()
        self.layout_candidate = layout_candidate
        self.far_field_layout_candidate = far_field_layout_candidate
        self._projection_provider = projection_provider
        self._far_field_projection_provider: ProjectionProvider | None = None
        if self.layout_candidate is None and self.config.layout_candidate_path is not None:
            self.layout_candidate = load_layout_candidate_for_runtime(
                Path(self.config.layout_candidate_path)
            )
        if (
            self.far_field_layout_candidate is None
            and self.config.far_field_layout_candidate_path is not None
        ):
            self.far_field_layout_candidate = load_far_field_layout_candidate(
                Path(self.config.far_field_layout_candidate_path)
            )
        if self.layout_candidate is not None:
            self._validate_candidate_projection_selection()

    def process(
        self,
        frames: Mapping[str, np.ndarray],
    ) -> RuntimeStitchResult:
        config = self.config.normalized()
        if config.mode == StitchRuntimeMode.NEAR_FIELD:
            return self._process_near_field(dict(frames))
        if config.mode == StitchRuntimeMode.AUTO:
            return self._process_auto(dict(frames))
        return self._process_far_field(dict(frames), warnings=tuple(config.runtime_warnings()))

    def _process_far_field(
        self,
        frames: dict[str, np.ndarray],
        warnings: tuple[str, ...] = (),
        status: str = "far_field_default",
    ) -> RuntimeStitchResult:
        if self.config.use_far_field_custom_layout:
            return self._process_far_field_custom(frames, warnings=warnings)
        start = time.perf_counter()
        warped, canvas = self.stitcher.process(frames)
        far_field_total_ms = _elapsed_ms(start)
        return RuntimeStitchResult(
            warped=warped,
            canvas=canvas,
            mode=StitchRuntimeMode.FAR_FIELD,
            status=status,
            warnings=warnings,
            metrics={
                "runtime_mode": status,
                "timing": {"far_field_total_ms": far_field_total_ms},
            },
        )

    def _process_far_field_custom(
        self,
        frames: dict[str, np.ndarray],
        warnings: tuple[str, ...] = (),
    ) -> RuntimeStitchResult:
        if self.far_field_layout_candidate is None:
            raise RuntimeError(
                "Far-field custom layout requires a loaded Far-field Layout Candidate."
            )
        projection = self._far_field_custom_projection_provider().project(frames)
        render = render_far_field_custom_from_projection(
            projection,
            self.far_field_layout_candidate,
            self.stitcher,
        )
        all_warnings = (
            tuple(warnings)
            + tuple(self.far_field_layout_candidate.warnings)
            + tuple(projection.metadata.get("warning_reasons", []))
        )
        metrics = dict(render.metrics)
        metrics["runtime_mode"] = "far_field_custom"
        timing = dict(metrics.get("timing", {}))
        timing.update(projection.timings)
        metrics["timing"] = timing
        return RuntimeStitchResult(
            warped=projection.warped_images,
            canvas=render.image,
            mode=StitchRuntimeMode.FAR_FIELD,
            status="far_field_custom",
            warnings=all_warnings,
            metrics=metrics,
        )

    def _far_field_custom_projection_provider(self) -> ProjectionProvider:
        if self.far_field_layout_candidate is None:
            raise RuntimeError(
                "Far-field custom layout requires a loaded Far-field Layout Candidate."
            )
        if self._far_field_projection_provider is not None:
            return self._far_field_projection_provider
        source = self.far_field_layout_candidate.projection_source
        if source == B2_FAR_FIELD_PROJECTION_SOURCE:
            directory = self.far_field_layout_candidate.b2_candidate_directory
            if directory is None:
                raise RuntimeError(
                    "Far-field custom B-2 layout requires projection.candidate_directory."
                )
            self._far_field_projection_provider = B2CandidateProjectionProvider(directory)
        elif source == "current_perspective":
            self._far_field_projection_provider = CurrentPerspectiveProjectionProvider(
                self.stitcher
            )
        else:
            raise RuntimeError(f"Unsupported Far-field custom projection source: {source}")
        return self._far_field_projection_provider

    def _process_auto(self, frames: dict[str, np.ndarray]) -> RuntimeStitchResult:
        warnings = (
            "Auto stitch runtime mode is not implemented; falling back to Far-field.",
        )
        return self._process_far_field(
            frames,
            warnings=warnings,
            status="auto_fallback_far_field",
        )

    def _process_near_field(self, frames: dict[str, np.ndarray]) -> RuntimeStitchResult:
        if self.config.projection_source == ProjectionSource.EQUIRECTANGULAR_CANDIDATE:
            raise RuntimeError(
                "Projection candidate is research-only and not connected to runtime yet."
            )
        if self.config.projection_source not in {
            ProjectionSource.CURRENT_PERSPECTIVE,
            ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
        }:
            raise RuntimeError(
                f"Unsupported Near-field projection source: {self.config.projection_source.value}"
            )
        if self.layout_candidate is None:
            raise RuntimeError(
                "Near-field mode requires a loaded Layout Candidate V2."
            )
        projection = self._near_field_projection_provider().project(frames)
        self._validate_near_field_projected_canvas(projection.warped_images)
        compositor_start = time.perf_counter()
        near = render_near_field_from_warped(
            projection.warped_images,
            self.layout_candidate,
            valid_masks=projection.valid_masks,
        )
        near_field_compositor_ms = _elapsed_ms(compositor_start)
        warnings = tuple(self.layout_candidate.warnings) + tuple(
            projection.metadata.get("warning_reasons", [])
        )
        metrics = dict(near.metrics)
        status = resolve_effective_runtime_status(self.config)
        metrics["runtime_mode"] = status
        timing = dict(metrics.get("timing", {}))
        timing.update(projection.timings)
        timing["near_field_compositor_ms"] = near_field_compositor_ms
        metrics["timing"] = timing
        metrics["projection"] = projection.metadata
        return RuntimeStitchResult(
            warped=projection.warped_images,
            canvas=near.canvas,
            mode=StitchRuntimeMode.NEAR_FIELD,
            status=status,
            warnings=warnings,
            metrics=metrics,
        )

    def _near_field_projection_provider(self) -> ProjectionProvider:
        if self._projection_provider is None:
            if self.config.projection_source == ProjectionSource.CURRENT_PERSPECTIVE:
                self._projection_provider = CurrentPerspectiveProjectionProvider(self.stitcher)
            elif self.config.projection_source == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE:
                if self.config.projection_intrinsics_source_path is None:
                    raise RuntimeError(
                        "Fisheye Rectilinear projection requires a loaded fisheye intrinsics source."
                    )
                source = load_fisheye_intrinsics_source(
                    self.config.projection_intrinsics_source_path
                )
                self._projection_provider = FisheyeRectilinearProjectionProvider(
                    self.stitcher,
                    source,
                    FisheyeRectilinearParams(
                        balance=float(self.config.fisheye_balance),
                        fov_scale=float(self.config.fisheye_fov_scale),
                    ),
                )
            else:
                raise RuntimeError(
                    "Projection candidate is research-only and not connected to runtime yet."
                )
        return self._projection_provider

    def _validate_candidate_projection_selection(self) -> None:
        if self.layout_candidate is None or self.config.mode != StitchRuntimeMode.NEAR_FIELD:
            return
        selected = self.config.projection_source
        if selected not in {
            ProjectionSource.CURRENT_PERSPECTIVE,
            ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
        }:
            return
        candidate_source = self.layout_candidate.projection.source
        if selected != candidate_source:
            raise RuntimeError(
                f"Near-field candidate projection {candidate_source.value!r} does not "
                f"match selected projection {selected.value!r}. Select a matching view "
                "or load a matching candidate."
            )

    def _validate_near_field_projected_canvas(
        self,
        warped_images: Mapping[str, np.ndarray],
    ) -> None:
        if self.layout_candidate is None:
            return
        required = ("front_left", "front", "front_right")
        missing = [camera for camera in required if camera not in warped_images]
        if missing:
            raise RuntimeError(
                "Near-field projection is missing warped images for: " + ", ".join(missing)
            )
        shapes = {
            camera: tuple(np.asarray(warped_images[camera]).shape[:2])
            for camera in required
        }
        if len(set(shapes.values())) != 1:
            raise RuntimeError(f"Near-field projected canvas sizes do not match: {shapes}")
        canvas_height, canvas_width = next(iter(shapes.values()))
        if (
            self.layout_candidate.output_width_px > canvas_width
            or self.layout_candidate.output_height_px > canvas_height
        ):
            raise RuntimeError(
                f"Near-field candidate output {self.layout_candidate.output_width_px}x"
                f"{self.layout_candidate.output_height_px} exceeds projected canvas "
                f"{canvas_width}x{canvas_height}."
            )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
