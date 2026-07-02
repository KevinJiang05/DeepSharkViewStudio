"""Fisheye rectilinear and equirectangular projection candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.stitcher import save_image

from .intrinsics_candidate import CAMERA_KEYS, FisheyeCameraModel


@dataclass(frozen=True)
class EquirectangularParams:
    canvas_width: int = 2200
    canvas_height: int = 700
    yaw_range_deg: tuple[float, float] = (-120.0, 120.0)
    pitch_range_deg: tuple[float, float] = (-45.0, 45.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["yaw_range_deg"] = list(self.yaw_range_deg)
        data["pitch_range_deg"] = list(self.pitch_range_deg)
        return data


@dataclass(frozen=True)
class ProjectionResult:
    warped_images: dict[str, np.ndarray]
    valid_masks: dict[str, np.ndarray]
    metadata: dict[str, Any]


def build_rectilinear_projection(
    frames: dict[str, np.ndarray],
    models: dict[str, FisheyeCameraModel],
    output_dir: str | Path,
    balance_values: list[float] | None = None,
    fov_scale_values: list[float] | None = None,
) -> dict[str, Any]:
    """Generate single-camera OpenCV fisheye rectilinear previews."""
    balance_values = list(balance_values or [0.0, 0.3, 0.6, 1.0])
    fov_scale_values = list(fov_scale_values or [1.0])
    root = Path(output_dir)
    previews = root / "previews"
    masks_dir = root / "valid_masks"
    maps_dir = root / "maps"
    for path in (previews, masks_dir, maps_dir):
        path.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "fisheye_rectilinear_preview",
        "balance_values": balance_values,
        "fov_scale_values": fov_scale_values,
        "cameras": {},
        "warnings": [],
    }
    for camera in CAMERA_KEYS:
        frame = frames.get(camera)
        model = models.get(camera)
        item: dict[str, Any] = {"status": "skipped", "warnings": []}
        if frame is None:
            item["warnings"].append("missing raw frame")
            report["cameras"][camera] = item
            continue
        if model is None:
            item["warnings"].append("missing opencv_fisheye K/D")
            report["cameras"][camera] = item
            continue
        height, width = frame.shape[:2]
        item.update(
            {
                "status": "generated",
                "source": model.source,
                "image_size": [width, height],
                "map_shape": [height, width],
                "map_dtype": "float32",
                "previews": {},
                "valid_masks": {},
                "maps": {},
            }
        )
        for balance in balance_values:
            for fov_scale in fov_scale_values:
                key = f"balance_{int(round(balance * 10)):02d}"
                new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    model.camera_matrix,
                    model.distortion,
                    (width, height),
                    np.eye(3),
                    balance=float(balance),
                    fov_scale=float(fov_scale),
                )
                map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                    model.camera_matrix,
                    model.distortion,
                    np.eye(3),
                    new_matrix,
                    (width, height),
                    cv2.CV_32FC1,
                )
                valid = _map_valid_mask(map_x, map_y, width, height)
                preview = cv2.remap(
                    frame,
                    map_x,
                    map_y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                preview[~valid] = 0
                preview_path = previews / f"{camera}_{key}.png"
                mask_path = masks_dir / f"{camera}_{key}_valid_mask.png"
                map_path = maps_dir / f"{camera}_{key}_map.npz"
                save_image(preview_path, preview)
                save_image(mask_path, _mask_image(valid))
                np.savez_compressed(map_path, map_x=map_x, map_y=map_y, valid_mask=valid.astype(np.uint8))
                item["previews"][key] = preview_path.relative_to(root).as_posix()
                item["valid_masks"][key] = mask_path.relative_to(root).as_posix()
                item["maps"][key] = map_path.relative_to(root).as_posix()
        report["cameras"][camera] = item
    return report


def build_equirectangular_projection(
    frames: dict[str, np.ndarray],
    models: dict[str, FisheyeCameraModel],
    transforms: dict[str, dict[str, Any]],
    output_dir: str | Path,
    params: EquirectangularParams | None = None,
) -> ProjectionResult:
    """Project fisheye frames into a rotation-only equirectangular canvas."""
    params = params or EquirectangularParams()
    root = Path(output_dir)
    previews = root / "previews"
    masks_dir = root / "valid_masks"
    maps_dir = root / "maps"
    for path in (previews, masks_dir, maps_dir):
        path.mkdir(parents=True, exist_ok=True)
    rays = _panorama_rays(params)
    remapped: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    forward_weights: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "mode": "equirectangular_rotation_only_preview",
        "params": params.to_dict(),
        "cameras": {},
        "warnings": [
            "rotation-only projection does not solve non-common-center near-field parallax",
        ],
    }
    for camera in CAMERA_KEYS:
        frame = frames.get(camera)
        model = models.get(camera)
        transform = transforms.get(camera)
        item: dict[str, Any] = {"status": "skipped", "warnings": []}
        if frame is None:
            item["warnings"].append("missing raw frame")
            metadata["cameras"][camera] = item
            continue
        if model is None:
            item["warnings"].append("missing opencv_fisheye K/D")
            metadata["cameras"][camera] = item
            continue
        if transform is None or not transform.get("rotation_camera_to_front"):
            item["warnings"].append("missing rotation_camera_to_front")
            metadata["cameras"][camera] = item
            continue
        map_x, map_y, valid, forward = _candidate_remap(
            rays,
            np.asarray(transform["rotation_camera_to_front"], dtype=np.float64),
            model.camera_matrix,
            model.distortion,
            model.image_size,
        )
        preview = cv2.remap(
            frame,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        preview[~valid] = 0
        remapped[camera] = preview
        masks[camera] = valid
        forward_weights[camera] = np.where(valid, forward, -np.inf)
        preview_path = previews / f"{camera}_equi.png"
        mask_path = masks_dir / f"{camera}_equi_valid_mask.png"
        map_path = maps_dir / f"{camera}_equi_map.npz"
        save_image(preview_path, preview)
        save_image(mask_path, _mask_image(valid))
        np.savez_compressed(map_path, map_x=map_x, map_y=map_y, valid_mask=valid.astype(np.uint8))
        item.update(
            {
                "status": "generated",
                "source": model.source,
                "preview": preview_path.relative_to(root).as_posix(),
                "valid_mask": mask_path.relative_to(root).as_posix(),
                "map": map_path.relative_to(root).as_posix(),
                "map_shape": list(map_x.shape),
                "map_dtype": str(map_x.dtype),
                "valid_ratio": float(np.count_nonzero(valid) / valid.size),
            }
        )
        metadata["cameras"][camera] = item

    if remapped:
        canvas = _winner_take_forward_canvas(remapped, masks, forward_weights)
        canvas_path = previews / "equi_overlay_debug.png"
        save_image(canvas_path, canvas)
        metadata["overlay_debug"] = canvas_path.relative_to(root).as_posix()
    metadata["suggested_overlaps"] = suggested_overlaps(masks)
    return ProjectionResult(remapped, masks, metadata)


def suggested_overlaps(valid_masks: dict[str, np.ndarray]) -> dict[str, Any]:
    pairs = {
        "left_front": ("front_left", "front"),
        "front_right": ("front", "front_right"),
    }
    output: dict[str, Any] = {}
    for pair_id, (left, right) in pairs.items():
        if left not in valid_masks or right not in valid_masks:
            output[pair_id] = {"x_range": None, "seam_x": None}
            continue
        shared = valid_masks[left] & valid_masks[right]
        min_rows = max(8, int(shared.shape[0] * 0.08))
        columns = np.where(np.count_nonzero(shared, axis=0) >= min_rows)[0]
        if not len(columns):
            output[pair_id] = {"x_range": None, "seam_x": None}
            continue
        start, end = int(columns[0]), int(columns[-1])
        output[pair_id] = {
            "x_range": [start, end],
            "seam_x": int(round((start + end) / 2.0)),
        }
    return output


def _panorama_rays(params: EquirectangularParams) -> np.ndarray:
    yaw = np.deg2rad(
        np.linspace(
            params.yaw_range_deg[0],
            params.yaw_range_deg[1],
            params.canvas_width,
            dtype=np.float32,
        )
    )
    pitch = np.deg2rad(
        np.linspace(
            params.pitch_range_deg[0],
            params.pitch_range_deg[1],
            params.canvas_height,
            dtype=np.float32,
        )
    )
    yaw_grid, pitch_grid = np.meshgrid(yaw, pitch)
    cos_pitch = np.cos(pitch_grid)
    return np.stack(
        (
            np.sin(yaw_grid) * cos_pitch,
            np.sin(pitch_grid),
            np.cos(yaw_grid) * cos_pitch,
        ),
        axis=-1,
    ).astype(np.float64)


def _candidate_remap(
    rays_front: np.ndarray,
    rotation_camera_to_front: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = rays_front.shape[:2]
    rays_camera = rays_front.reshape(-1, 3) @ rotation_camera_to_front
    projected, _ = cv2.fisheye.projectPoints(
        rays_camera.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(height, width, 2)
    map_x = projected[:, :, 0].astype(np.float32)
    map_y = projected[:, :, 1].astype(np.float32)
    input_width, input_height = resolution
    forward = rays_camera[:, 2].reshape(height, width)
    valid = (
        (forward > 0.02)
        & (map_x >= 0.0)
        & (map_x < input_width - 1)
        & (map_y >= 0.0)
        & (map_y < input_height - 1)
    )
    return map_x, map_y, valid, forward.reshape(height, width)


def _map_valid_mask(map_x: np.ndarray, map_y: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        (map_x >= 0.0)
        & (map_x < width - 1)
        & (map_y >= 0.0)
        & (map_y < height - 1)
    )


def _winner_take_forward_canvas(
    remapped: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    camera_order = [camera for camera in CAMERA_KEYS if camera in remapped]
    if not camera_order:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    stack = np.stack([weights[camera] for camera in camera_order], axis=0)
    selected = np.argmax(stack, axis=0)
    available = np.any(np.isfinite(stack), axis=0)
    canvas = np.zeros_like(remapped[camera_order[0]])
    for index, camera in enumerate(camera_order):
        mask = available & (selected == index) & masks[camera]
        canvas[mask] = remapped[camera][mask]
    cv2.putText(
        canvas,
        "EXPERIMENTAL equirectangular rotation-only candidate",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _mask_image(mask: np.ndarray) -> np.ndarray:
    return np.dstack([mask.astype(np.uint8) * 255] * 3)
