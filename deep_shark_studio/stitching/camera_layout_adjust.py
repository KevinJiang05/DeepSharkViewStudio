"""Shared post-warp camera layout adjustment helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraAdjustParams:
    x_offset_px: int = 0
    y_offset_px: int = 0
    scale: float = 1.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class AdjustedWarpResult:
    warped_images: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray]
    transforms: dict[str, np.ndarray]
    metadata: dict[str, Any]


def identity_camera_adjust() -> dict[str, CameraAdjustParams]:
    return {
        "front_left": CameraAdjustParams(),
        "front": CameraAdjustParams(),
        "front_right": CameraAdjustParams(),
    }


def apply_post_warp_camera_adjustments(
    warped_images: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray] | None,
    camera_adjust: dict[str, CameraAdjustParams],
) -> AdjustedWarpResult:
    """Apply preview/runtime-only affine x/y/scale adjustment after formal warp."""
    adjusted: dict[str, np.ndarray] = {}
    adjusted_masks: dict[str, np.ndarray] = {}
    transforms: dict[str, np.ndarray] = {}
    cameras: dict[str, Any] = {}
    input_masks = valid_masks or {}
    fallback_cameras: list[str] = []
    for camera, image in warped_images.items():
        params = camera_adjust.get(camera, CameraAdjustParams())
        mask = input_masks.get(camera)
        has_explicit_valid_mask = mask is not None
        if mask is None:
            # Temporary runtime-compatible validity heuristic: current formal
            # warped images use black fill for invalid canvas regions.
            mask = np.any(image != 0, axis=2)
            fallback_cameras.append(str(camera))
        else:
            mask_array = np.asarray(mask)
            if mask_array.shape != image.shape[:2]:
                raise ValueError(f"valid_mask size mismatch for {camera}")
            mask = mask_array.astype(bool)
        height, width = image.shape[:2]
        try:
            scale = float(params.scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera_adjust.{camera}.scale must be numeric.") from exc
        if not math.isfinite(scale) or scale < 0.80 or scale > 1.20:
            raise ValueError(
                f"camera_adjust.{camera}.scale={scale} is outside the finite "
                "runtime range [0.8, 1.2]."
            )
        x_offset = int(params.x_offset_px)
        y_offset = int(params.y_offset_px)
        identity_fast_path = (
            has_explicit_valid_mask
            and x_offset == 0
            and y_offset == 0
            and scale == 1.0
        )
        if identity_fast_path:
            # For an identity transform the scale center is mathematically
            # invariant. Use the canvas center so the fast path does not scan
            # the full validity mask merely to find its bounding box.
            center = ((width - 1) / 2.0, (height - 1) / 2.0)
        else:
            center = _valid_bbox_center(mask, width, height)
        matrix = _camera_adjust_matrix(
            center=center,
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        if identity_fast_path:
            # Preserve the old non-aliasing result contract and invalid-region
            # black fill without paying for two identity affine warps.
            adjusted_image = np.zeros_like(image)
            cv2.copyTo(
                image,
                np.ascontiguousarray(mask, dtype=np.uint8),
                adjusted_image,
            )
            adjusted_mask = mask.copy()
        else:
            adjusted_image = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            adjusted_mask = cv2.warpAffine(
                mask.astype(np.uint8),
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
        if not identity_fast_path:
            adjusted_image[~adjusted_mask] = 0
        valid_count_before = int(np.count_nonzero(mask))
        valid_count_after = (
            valid_count_before
            if identity_fast_path
            else int(np.count_nonzero(adjusted_mask))
        )
        adjusted[camera] = adjusted_image
        adjusted_masks[camera] = adjusted_mask
        transforms[camera] = matrix
        cameras[camera] = {
            "x_offset_px": x_offset,
            "y_offset_px": y_offset,
            "scale": scale,
            "scale_center": [float(center[0]), float(center[1])],
            "matrix": matrix.tolist(),
            "valid_ratio_before": float(valid_count_before / max(1, mask.size)),
            "valid_ratio_after": float(valid_count_after / max(1, adjusted_mask.size)),
            "identity_fast_path": identity_fast_path,
        }
    return AdjustedWarpResult(
        warped_images=adjusted,
        valid_masks=adjusted_masks,
        transforms=transforms,
        metadata={
            "mode": "post_warp_camera_adjustment",
            "note": "Preview/runtime-only affine display adjustment; not camera extrinsics calibration.",
            "valid_mask_source": (
                "mixed_provider_and_nonzero_fallback"
                if fallback_cameras and valid_masks
                else "nonzero_pixels_runtime_compatible"
                if fallback_cameras
                else "provider_valid_masks"
            ),
            "fallback_mask_cameras": fallback_cameras,
            "cameras": cameras,
        },
    )


def _valid_bbox_center(mask: np.ndarray, width: int, height: int) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return ((width - 1) / 2.0, (height - 1) / 2.0)
    return (
        float((int(xs.min()) + int(xs.max())) / 2.0),
        float((int(ys.min()) + int(ys.max())) / 2.0),
    )


def _camera_adjust_matrix(
    center: tuple[float, float],
    scale: float,
    x_offset: int,
    y_offset: int,
) -> np.ndarray:
    cx, cy = center
    return np.asarray(
        [
            [scale, 0.0, cx - scale * cx + float(x_offset)],
            [0.0, scale, cy - scale * cy + float(y_offset)],
        ],
        dtype=np.float32,
    )
