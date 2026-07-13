"""Perspective warping and surround-view stitching."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import scale_camera_matrix_for_resolution
from .masks import crop_image_by_line
from .topology import (
    compose_horizontal_feather,
    resolve_stitch_config,
    source_coordinate_diagnostics,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    source_points: np.ndarray
    target_points: np.ndarray
    _matrix: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source_points.shape != (4, 2) or self.target_points.shape != (4, 2):
            raise ValueError(f"{self.name} perspective points must both have shape (4, 2).")
        if not np.all(np.isfinite(self.source_points)) or not np.all(
            np.isfinite(self.target_points)
        ):
            raise ValueError(f"{self.name} perspective points must be finite.")
        object.__setattr__(
            self,
            "_matrix",
            cv2.getPerspectiveTransform(self.source_points, self.target_points),
        )

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "CameraCalibration":
        return cls(
            name=name,
            source_points=np.asarray(config["source_points"], dtype=np.float32),
            target_points=np.asarray(config["target_points"], dtype=np.float32),
        )

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class SurroundStitcher:
    """Generate bird's-eye warped images and a stitched surround canvas."""

    def __init__(self, config: dict[str, Any], max_input_width: int | None = None, use_intrinsics: bool = True):
        config = resolve_stitch_config(config)
        self.topology_name = str(config.get("_topology_name", "legacy_surround_5"))
        self.source_coordinate_space = str(
            config.get("source_coordinate_space", "")
        )
        self.source_reference_size = config.get("source_reference_size")
        self.source_contract_version = config.get("source_contract_version")
        canvas = config.get("canvas", {})
        self.output_width = int(canvas.get("width", 2440))
        self.output_height = int(canvas.get("height", 1800))
        self.max_input_width = max_input_width
        self.use_intrinsics = use_intrinsics
        self.transition_width = int(config.get("transition_width", 5))
        self.cameras = {
            name: CameraCalibration.from_config(name, camera_config)
            for name, camera_config in config.get("cameras", {}).items()
        }
        self.stitch_points = {
            name: [tuple(point) for point in points]
            for name, points in config.get("stitch_points", {}).items()
        }
        self.mask_rules = config.get("mask_rules", [])
        self.active_camera_keys = [str(key) for key in config.get("active_cameras", self.cameras)]
        self.overlaps = config.get("overlaps", [])
        self.composition = config.get("composition", {})
        self.camera_intrinsics = config.get("camera_intrinsics", {})
        self._masks: dict[str, np.ndarray] = {}
        self._source_diagnostic_sizes: dict[str, tuple[int, int]] = {}
        self._intrinsics_remap_cache: dict[
            tuple[str, int, int],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._geometry_mask_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self._runtime_geometry_cache_limit = max(4, len(self.active_camera_keys) * 2)
        self._validate_intrinsics_contract()

    def source_diagnostics(
        self,
        camera_name: str,
        actual_raw_size: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        return source_coordinate_diagnostics(
            {
                "source_coordinate_space": self.source_coordinate_space,
                "source_reference_size": self.source_reference_size,
                "source_contract_version": self.source_contract_version,
                "cameras": {
                    name: {
                        "source_points": calibration.source_points.tolist(),
                    }
                    for name, calibration in self.cameras.items()
                },
            },
            camera_name,
            actual_raw_size,
        )

    @property
    def geometry_mask_cache_entries(self) -> int:
        return len(self._geometry_mask_cache)

    @property
    def intrinsics_remap_cache_entries(self) -> int:
        return len(self._intrinsics_remap_cache)

    def warp(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        warped, _mask = self.warp_with_mask(image, camera_name)
        return warped

    def warp_with_mask(
        self,
        image: np.ndarray,
        camera_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        calibration = self.cameras[camera_name]
        raw_size = (int(image.shape[1]), int(image.shape[0]))
        if (
            (self.source_coordinate_space or self.source_reference_size)
            and self._source_diagnostic_sizes.get(camera_name) != raw_size
        ):
            self._source_diagnostic_sizes[camera_name] = raw_size
            for warning in self.source_diagnostics(
                camera_name,
                raw_size,
            )["warnings"]:
                LOGGER.warning("Source coordinate warning: %s", warning)
        matrix = calibration.matrix.copy()
        processed = image
        if self.max_input_width and image.shape[1] > self.max_input_width:
            scale_x = self.max_input_width / float(image.shape[1])
            resized_height = max(1, int(round(image.shape[0] * scale_x)))
            scale_y = resized_height / float(image.shape[0])
            processed = cv2.resize(
                image,
                (self.max_input_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )
            scale_matrix = np.array(
                [
                    [1.0 / scale_x, 0.0, 0.0],
                    [0.0, 1.0 / scale_y, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            matrix = matrix @ scale_matrix
        source_valid: np.ndarray | None = None
        if self.use_intrinsics:
            map_x, map_y, source_valid = self._intrinsics_remap(
                camera_name,
                (int(processed.shape[1]), int(processed.shape[0])),
            )
            processed = cv2.remap(
                processed,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        warped = cv2.warpPerspective(
            processed,
            matrix,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cache_key = (camera_name, raw_size[0], raw_size[1])
        valid_mask = self._geometry_mask_cache.get(cache_key)
        if valid_mask is None:
            if source_valid is None:
                source_valid = np.ones(processed.shape[:2], dtype=np.uint8)
            valid_mask = cv2.warpPerspective(
                np.asarray(source_valid, dtype=np.uint8),
                matrix,
                (self.output_width, self.output_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            valid_mask.flags.writeable = False
            self._store_bounded_runtime_geometry(
                self._geometry_mask_cache,
                cache_key,
                valid_mask,
            )
        warped = cv2.copyTo(
            warped,
            np.ascontiguousarray(valid_mask, dtype=np.uint8),
        )
        return warped, valid_mask

    def warp_all(self, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        warped, _valid_masks = self.warp_all_with_masks(frames)
        return warped

    def warp_all_with_masks(
        self,
        frames: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        warped: dict[str, np.ndarray] = {}
        valid_masks: dict[str, np.ndarray] = {}
        for camera_name, frame in frames.items():
            if camera_name not in self.cameras or camera_name not in self.active_camera_keys:
                continue
            warped[camera_name], valid_masks[camera_name] = self.warp_with_mask(
                frame,
                camera_name,
            )
        return warped, valid_masks

    def stitch(
        self,
        warped_images: dict[str, np.ndarray],
        valid_masks: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if self.composition.get("mode") == "horizontal_feather":
            return compose_horizontal_feather(
                warped_images,
                self.output_width,
                self.output_height,
                self.stitch_points,
                self.overlaps,
                int(self.composition.get("feather_width", 120)),
                valid_masks=valid_masks,
            )

        self._ensure_masks()
        canvas = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        for rule in self.mask_rules:
            camera_name = rule["camera"]
            if camera_name not in warped_images:
                continue
            mask = self._masks[camera_name]
            if mask.ndim == 2:
                mask = np.stack([mask] * 3, axis=-1)
            cropped = cv2.bitwise_and(warped_images[camera_name], mask)
            canvas = cv2.bitwise_or(canvas, cropped)

        return canvas

    def process(self, frames: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        warped, valid_masks = self.warp_all_with_masks(frames)
        return warped, self.stitch(warped, valid_masks=valid_masks)

    def _validate_intrinsics_contract(self) -> None:
        if not self.use_intrinsics:
            return
        missing = [
            camera
            for camera in self.active_camera_keys
            if camera not in self.camera_intrinsics
        ]
        if missing:
            raise ValueError(
                "Pinhole intrinsics are enabled but missing for active cameras: "
                + ", ".join(missing)
            )
        reference_sizes: set[tuple[int, int]] = set()
        for camera in self.active_camera_keys:
            intrinsics = self.camera_intrinsics[camera]
            image_size = intrinsics.get("image_size")
            if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
                raise ValueError(f"{camera} intrinsics.image_size must contain width and height.")
            size = (int(image_size[0]), int(image_size[1]))
            if min(size) <= 0:
                raise ValueError(f"{camera} intrinsics.image_size must be positive.")
            reference_sizes.add(size)
            matrix = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
            distortion = np.asarray(intrinsics.get("distortion"), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{camera} intrinsics.camera_matrix must be finite 3x3.")
            if distortion.size < 4 or not np.all(np.isfinite(distortion)):
                raise ValueError(f"{camera} intrinsics.distortion must be finite.")
        if len(reference_sizes) != 1:
            raise ValueError(
                "Active-camera intrinsics must use one consistent calibration image_size."
            )

    def _intrinsics_remap(
        self,
        camera_name: str,
        runtime_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (camera_name, int(runtime_size[0]), int(runtime_size[1]))
        cached = self._intrinsics_remap_cache.get(key)
        if cached is not None:
            return cached
        intrinsics = self.camera_intrinsics[camera_name]
        calibration_size = tuple(int(value) for value in intrinsics["image_size"])
        camera_matrix = scale_camera_matrix_for_resolution(
            np.asarray(intrinsics["camera_matrix"], dtype=np.float64),
            calibration_size=calibration_size,
            runtime_size=runtime_size,
        )
        distortion = np.asarray(intrinsics["distortion"], dtype=np.float64)
        new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            runtime_size,
            1,
            runtime_size,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            new_matrix,
            runtime_size,
            cv2.CV_32FC1,
        )
        source_valid = cv2.remap(
            np.ones((runtime_size[1], runtime_size[0]), dtype=np.uint8),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cached = (map_x, map_y, source_valid)
        self._store_bounded_runtime_geometry(
            self._intrinsics_remap_cache,
            key,
            cached,
        )
        return cached

    def _store_bounded_runtime_geometry(
        self,
        cache: dict[Any, Any],
        key: Any,
        value: Any,
    ) -> None:
        if key not in cache and len(cache) >= self._runtime_geometry_cache_limit:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def _ensure_masks(self) -> None:
        if self._masks:
            return

        dummy = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        for rule in self.mask_rules:
            points_a = self.stitch_points[rule["line_a"]]
            points_b = self.stitch_points[rule["line_b"]]
            _, mask = crop_image_by_line(
                dummy,
                points_a[0],
                points_a[1],
                points_b[0],
                points_b[1],
                rule["side"],
            )
            self._masks[rule["camera"]] = mask


def load_images_from_directory(image_dir: str | Path) -> dict[str, np.ndarray]:
    """Load images whose filenames contain a known camera direction."""
    directory = Path(image_dir)
    frames: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lower_name = path.stem.lower()
        for camera_name in ("front_left", "front_right", "behind", "left", "right", "front"):
            if camera_name in lower_name:
                image = cv2.imread(str(path))
                if image is not None:
                    frames[camera_name] = image
                break
    return frames


def save_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write image: {output_path}")
