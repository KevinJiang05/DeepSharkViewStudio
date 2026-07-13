"""Runtime projection provider abstractions.

Far-field keeps the formal perspective runtime. Near-field can select the
current perspective provider or the experimental fisheye rectilinear provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

import cv2
import numpy as np

from deep_shark_studio.calibration_candidate import CandidatePanoramaProcessor
from deep_shark_studio.stitch_runtime_modes import ProjectionSource

from .intrinsics_runtime_loader import (
    FisheyeCameraIntrinsics,
    FisheyeIntrinsicsRuntimeSource,
)


@dataclass(frozen=True)
class ProjectionResult:
    warped_images: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray]
    metadata: dict[str, Any]
    timings: dict[str, float]


class ProjectionProvider(Protocol):
    def project(self, frames: Mapping[str, np.ndarray]) -> ProjectionResult:
        ...


class CurrentPerspectiveProjectionProvider:
    """Wrap the existing SurroundStitcher.warp_all() runtime behavior."""

    provider_name = "CurrentPerspectiveProjectionProvider"

    def __init__(self, stitcher: Any):
        self.stitcher = stitcher

    def project(self, frames: Mapping[str, np.ndarray]) -> ProjectionResult:
        total_start = time.perf_counter()
        warp_start = time.perf_counter()
        geometry_masks_available = hasattr(self.stitcher, "warp_all_with_masks")
        if geometry_masks_available:
            warped, valid_masks = self.stitcher.warp_all_with_masks(dict(frames))
        else:
            warped = self.stitcher.warp_all(dict(frames))
            valid_masks = derive_nonzero_valid_masks(warped)
        warp_all_ms = _elapsed_ms(warp_start)
        warnings: list[str] = []
        uses_intrinsics = bool(getattr(self.stitcher, "use_intrinsics", False))
        if uses_intrinsics:
            warnings.append(
                "CurrentPerspectiveProvider may include pinhole cv2.undistort, not fisheye remap."
            )
        metadata = {
            "projection_source": ProjectionSource.CURRENT_PERSPECTIVE.value,
            "provider_name": self.provider_name,
            "canvas_size": _canvas_size(self.stitcher, warped),
            "uses_intrinsics": uses_intrinsics,
            "uses_fisheye": False,
            "uses_remap": False,
            "valid_mask_source": (
                "perspective_warp_geometry_mask"
                if geometry_masks_available
                else "nonzero_pixels_legacy_fallback"
            ),
            "warning_reasons": warnings,
        }
        return ProjectionResult(
            warped_images=warped,
            valid_masks=valid_masks,
            metadata=metadata,
            timings={
                "projection_total_ms": _elapsed_ms(total_start),
                "warp_all_ms": warp_all_ms,
            },
        )


B2_FAR_FIELD_PROJECTION_SOURCE = "b2_far_field_candidate"


class B2CandidateProjectionProvider:
    """Wrap a read-only B-2 candidate as per-camera warped projection input."""

    provider_name = "B2CandidateProjectionProvider"

    def __init__(self, candidate_directory: str | Path):
        self.candidate_directory = Path(candidate_directory)
        self.processor = CandidatePanoramaProcessor(self.candidate_directory)

    def project(self, frames: Mapping[str, np.ndarray]) -> ProjectionResult:
        total_start = time.perf_counter()
        process_start = time.perf_counter()
        if hasattr(self.processor, "warp_all"):
            warped = self.processor.warp_all(dict(frames))
        else:
            warped, _canvas = self.processor.process(dict(frames))
        process_ms = _elapsed_ms(process_start)
        processor_timing = getattr(self.processor, "last_timings", {})
        valid_masks = self._valid_masks_for_warped(warped)
        metadata = {
            "projection_source": B2_FAR_FIELD_PROJECTION_SOURCE,
            "provider_name": self.provider_name,
            "candidate_directory": str(self.candidate_directory),
            "topology": str(getattr(self.processor, "topology_name", "")),
            "canvas_size": [
                int(getattr(self.processor, "output_width", 0)),
                int(getattr(self.processor, "output_height", 0)),
            ],
            "uses_intrinsics": True,
            "uses_fisheye": True,
            "uses_remap": True,
            "uses_equirectangular": False,
            "valid_mask_source": "b2_candidate_remap_valid_mask",
            "warning_reasons": [
                "B-2 candidate projection is read-only and experimental.",
                "Far-field layout tuning uses B-2 per-camera warped images, not the flattened B-2 final canvas.",
            ],
            "_b2_weight_maps": self._weight_maps_for_warped(warped),
            "_b2_selection_masks": self._selection_masks_for_warped(warped),
        }
        return ProjectionResult(
            warped_images=warped,
            valid_masks=valid_masks,
            metadata=metadata,
            timings={
                "projection_total_ms": _elapsed_ms(total_start),
                "b2_candidate_process_ms": process_ms,
                "b2_candidate_remap_ms": float(
                    processor_timing.get("candidate_remap_ms", 0.0)
                ) if isinstance(processor_timing, dict) else 0.0,
                "b2_candidate_compose_ms": float(
                    processor_timing.get("candidate_compose_ms", 0.0)
                ) if isinstance(processor_timing, dict) else 0.0,
            },
        )

    def _valid_masks_for_warped(
        self,
        warped: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        processor_masks = getattr(self.processor, "_masks", {})
        for camera, image in warped.items():
            mask = processor_masks.get(camera)
            if mask is not None and mask.shape == image.shape[:2]:
                masks[camera] = np.asarray(mask, dtype=bool)
            else:
                raise RuntimeError(
                    "B-2 candidate projection is missing its geometric valid mask "
                    f"for {camera}; pixel color cannot be used as geometry."
                )
        return masks

    def _weight_maps_for_warped(
        self,
        warped: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        weights: dict[str, np.ndarray] = {}
        processor_weights = getattr(self.processor, "_weights", {})
        for camera, image in warped.items():
            weight = processor_weights.get(camera)
            if weight is not None and weight.shape == image.shape[:2]:
                weights[camera] = np.asarray(weight, dtype=np.float32)
        return weights

    def _selection_masks_for_warped(
        self,
        warped: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expose the processor's immutable winner plan for identity layouts.

        CandidatePanoramaProcessor builds these masks once from the same B-2
        per-camera weights used by its final canvas.  Keeping the mapping in
        private ProjectionResult metadata lets Far Custom reuse that plan
        without allocating and reducing a three-map float stack every frame.
        """
        camera_order = tuple(getattr(self.processor, "camera_order", ()))
        available = tuple(camera for camera in camera_order if camera in warped)
        plans = getattr(self.processor, "_selection_masks_by_available", {})
        selection = plans.get(available, {}) if isinstance(plans, Mapping) else {}
        result: dict[str, np.ndarray] = {}
        for camera in available:
            mask = selection.get(camera) if isinstance(selection, Mapping) else None
            image = warped[camera]
            if mask is not None and np.asarray(mask).shape == image.shape[:2]:
                result[camera] = np.asarray(mask, dtype=bool)
        return result


@dataclass(frozen=True)
class FisheyeRectilinearParams:
    balance: float = 0.6
    fov_scale: float = 1.0


@dataclass(frozen=True)
class FisheyeRemapCacheEntry:
    map_x: np.ndarray
    map_y: np.ndarray
    rectified_valid_mask: np.ndarray
    rectified_size: tuple[int, int]
    build_ms: float


class FisheyeRectilinearProjectionProvider:
    """Experimental fisheye rectilinear remap followed by template perspective warp."""

    provider_name = "FisheyeRectilinearProjectionProvider"

    def __init__(
        self,
        stitcher: Any,
        intrinsics_source: FisheyeIntrinsicsRuntimeSource,
        params: FisheyeRectilinearParams | None = None,
    ):
        self.stitcher = stitcher
        self.intrinsics_source = intrinsics_source
        self.params = params or FisheyeRectilinearParams()
        self._maps: dict[str, FisheyeRemapCacheEntry] = {}
        self.map_build_count = 0
        self._build_all_maps()

    def project(self, frames: Mapping[str, np.ndarray]) -> ProjectionResult:
        total_start = time.perf_counter()
        remap_ms = 0.0
        warp_ms = 0.0
        valid_mask_ms = 0.0
        warped: dict[str, np.ndarray] = {}
        valid_masks: dict[str, np.ndarray] = {}
        warnings = [
            "Fisheye Rectilinear projection is experimental and only affects Near-field.",
            "Template source_points were calibrated on raw/template images; after rectilinear remap, layout may need re-tuning.",
        ]
        warnings.extend(self.intrinsics_source.warnings)
        for camera in ("front_left", "front", "front_right"):
            frame = frames.get(camera)
            if frame is None:
                continue
            model = self.intrinsics_source.cameras[camera]
            actual_size = (int(frame.shape[1]), int(frame.shape[0]))
            if actual_size != model.image_size:
                raise ValueError(
                    f"{camera} frame size {actual_size} does not match fisheye intrinsics image_size {model.image_size}."
                )
            entry = self._maps[camera]
            remap_start = time.perf_counter()
            rectified = cv2.remap(
                frame,
                entry.map_x,
                entry.map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            remap_ms += _elapsed_ms(remap_start)
            warp_start = time.perf_counter()
            warped[camera], valid_masks[camera] = self._warp_rectified_camera(
                camera,
                rectified,
                entry.rectified_valid_mask,
            )
            warp_ms += _elapsed_ms(warp_start)
        valid_mask_ms = warp_ms
        map_build_ms = sum(entry.build_ms for entry in self._maps.values())
        metadata = {
            "projection_source": ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
            "provider_name": self.provider_name,
            "projection_pipeline": "fisheye_rectilinear_then_template_perspective_warp",
            "uses_intrinsics": True,
            "uses_fisheye": True,
            "uses_remap": True,
            "uses_equirectangular": False,
            "intrinsics_source_path": str(self.intrinsics_source.path),
            "intrinsics_source_type": self.intrinsics_source.source_type,
            "balance": float(self.params.balance),
            "fov_scale": float(self.params.fov_scale),
            "rectified_size": list(self._common_rectified_size()),
            "canvas_size": _canvas_size(self.stitcher, warped),
            "valid_mask_source": "fisheye_remap_geometry_mask_then_template_warp",
            "map_cache": {
                "created": True,
                "reused": True,
                "entry_count": len(self._maps),
                "build_count": self.map_build_count,
            },
            "warning_reasons": warnings,
        }
        return ProjectionResult(
            warped_images=warped,
            valid_masks=valid_masks,
            metadata=metadata,
            timings={
                "projection_total_ms": _elapsed_ms(total_start),
                "map_build_ms": 0.0,
                "map_cache_build_ms": map_build_ms,
                "fisheye_remap_ms": remap_ms,
                "template_warp_ms": warp_ms,
                "valid_mask_ms": valid_mask_ms,
            },
        )

    def _build_all_maps(self) -> None:
        for camera, model in self.intrinsics_source.cameras.items():
            self._maps[camera] = self._build_map(model)

    def _build_map(self, model: FisheyeCameraIntrinsics) -> FisheyeRemapCacheEntry:
        start = time.perf_counter()
        width, height = model.image_size
        new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            model.camera_matrix,
            model.distortion,
            (width, height),
            np.eye(3),
            balance=float(self.params.balance),
            fov_scale=float(self.params.fov_scale),
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            model.camera_matrix,
            model.distortion,
            np.eye(3),
            new_matrix,
            (width, height),
            cv2.CV_32FC1,
        )
        raw_valid = np.ones((height, width), dtype=np.uint8)
        rectified_valid = cv2.remap(
            raw_valid,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        self.map_build_count += 1
        return FisheyeRemapCacheEntry(
            map_x=map_x,
            map_y=map_y,
            rectified_valid_mask=rectified_valid,
            rectified_size=(width, height),
            build_ms=_elapsed_ms(start),
        )

    def _warp_rectified_camera(
        self,
        camera: str,
        rectified: np.ndarray,
        rectified_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.stitcher.cameras[camera]
        matrix = calibration.matrix.copy()
        image = rectified
        mask = rectified_valid.astype(np.uint8)
        max_input_width = getattr(self.stitcher, "max_input_width", None)
        if max_input_width and image.shape[1] > int(max_input_width):
            scale = int(max_input_width) / float(image.shape[1])
            image = cv2.resize(
                image,
                (int(max_input_width), int(image.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
            mask = cv2.resize(
                mask,
                (int(max_input_width), int(mask.shape[0] * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
            scale_matrix = np.array(
                [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            matrix = matrix @ scale_matrix
        size = (
            int(getattr(self.stitcher, "output_width")),
            int(getattr(self.stitcher, "output_height")),
        )
        warped = cv2.warpPerspective(
            image,
            matrix,
            size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        canvas_mask = cv2.warpPerspective(
            mask.astype(np.uint8),
            matrix,
            size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        warped[~canvas_mask] = 0
        return warped, canvas_mask

    def _common_rectified_size(self) -> tuple[int, int]:
        sizes = [entry.rectified_size for entry in self._maps.values()]
        return sizes[0] if sizes else (0, 0)

    def cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.params).encode("utf-8"))
        for camera in sorted(self.intrinsics_source.cameras):
            model = self.intrinsics_source.cameras[camera]
            digest.update(camera.encode("utf-8"))
            digest.update(str(model.image_size).encode("utf-8"))
            digest.update(model.camera_matrix.tobytes())
            digest.update(model.distortion.tobytes())
        return digest.hexdigest()


def derive_nonzero_valid_masks(
    warped_images: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Derive temporary masks from current black-filled perspective canvases."""
    masks: dict[str, np.ndarray] = {}
    for camera, image in warped_images.items():
        masks[str(camera)] = np.any(image != 0, axis=2).astype(bool)
    return masks


def _canvas_size(stitcher: Any, warped_images: Mapping[str, np.ndarray]) -> list[int]:
    width = getattr(stitcher, "output_width", None)
    height = getattr(stitcher, "output_height", None)
    if width is not None and height is not None:
        return [int(width), int(height)]
    if warped_images:
        first = next(iter(warped_images.values()))
        image_height, image_width = first.shape[:2]
        return [int(image_width), int(image_height)]
    return [0, 0]


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
