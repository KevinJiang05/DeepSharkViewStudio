"""Perspective warping and surround-view stitching."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import undistort_image
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

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> "CameraCalibration":
        return cls(
            name=name,
            source_points=np.asarray(config["source_points"], dtype=np.float32),
            target_points=np.asarray(config["target_points"], dtype=np.float32),
        )

    @property
    def matrix(self) -> np.ndarray:
        return cv2.getPerspectiveTransform(self.source_points, self.target_points)


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

    def warp(self, image: np.ndarray, camera_name: str) -> np.ndarray:
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
        if self.max_input_width and image.shape[1] > self.max_input_width:
            scale = self.max_input_width / float(image.shape[1])
            image = cv2.resize(image, (self.max_input_width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            scale_matrix = np.array(
                [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            matrix = matrix @ scale_matrix
        if self.use_intrinsics and camera_name in self.camera_intrinsics:
            image = undistort_image(image, self.camera_intrinsics[camera_name])
        return cv2.warpPerspective(
            image,
            matrix,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def warp_all(self, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        warped: dict[str, np.ndarray] = {}
        for camera_name, frame in frames.items():
            if camera_name not in self.cameras or camera_name not in self.active_camera_keys:
                continue
            warped[camera_name] = self.warp(frame, camera_name)
        return warped

    def stitch(self, warped_images: dict[str, np.ndarray]) -> np.ndarray:
        if self.composition.get("mode") == "horizontal_feather":
            return compose_horizontal_feather(
                warped_images,
                self.output_width,
                self.output_height,
                self.stitch_points,
                self.overlaps,
                int(self.composition.get("feather_width", 120)),
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
        warped = self.warp_all(frames)
        return warped, self.stitch(warped)

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
