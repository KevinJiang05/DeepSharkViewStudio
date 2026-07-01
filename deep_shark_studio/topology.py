"""Stitch-topology profile resolution and generic horizontal composition."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


LEGACY_TOPOLOGY_NAME = "legacy_surround_5"


def selected_topology_name(config: dict[str, Any]) -> str:
    """Return the explicitly selected topology, or the legacy fallback."""
    return str(config.get("stitch_topology") or LEGACY_TOPOLOGY_NAME)


def active_stitch_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable geometry mapping edited by the current profile."""
    name = selected_topology_name(config)
    profile = config.get("topologies", {}).get(name)
    if not isinstance(profile, dict) or profile.get("legacy_root"):
        return config
    return profile


def active_topology_camera_keys(config: dict[str, Any]) -> list[str]:
    profile = active_stitch_profile(config)
    declared = profile.get("active_cameras")
    if isinstance(declared, list):
        return [str(key) for key in declared]
    order = profile.get("camera_order", config.get("camera_order", []))
    if order:
        return [str(key) for key in order]
    return [str(key) for key in profile.get("cameras", {})]


def resolve_stitch_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve one runtime stitch config while preserving legacy YAML support."""
    name = selected_topology_name(config)
    profile = active_stitch_profile(config)
    if profile is config:
        resolved = deepcopy(config)
    else:
        resolved = deepcopy(profile)
        resolved["camera_intrinsics"] = deepcopy(config.get("camera_intrinsics", {}))
        resolved["transition_width"] = int(
            profile.get("transition_width", config.get("transition_width", 5))
        )
    resolved["_topology_name"] = name
    resolved["active_cameras"] = active_topology_camera_keys(config)
    return resolved


def seam_x(stitch_points: dict[str, Any], seam_name: str) -> float:
    points = stitch_points.get(seam_name, [])
    if len(points) != 2:
        raise ValueError(f"Seam '{seam_name}' must contain exactly two points")
    return (float(points[0][0]) + float(points[1][0])) / 2.0


def compose_horizontal_feather(
    warped_images: dict[str, np.ndarray],
    output_width: int,
    output_height: int,
    stitch_points: dict[str, Any],
    overlaps: list[dict[str, Any]],
    default_feather_width: int,
) -> np.ndarray:
    """Blend available left-to-right cameras using profile-declared overlaps."""
    if not warped_images:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8)

    weights: dict[str, np.ndarray] = {}
    for camera_name, image in warped_images.items():
        if image.shape[:2] != (output_height, output_width):
            raise ValueError(f"Warped image size mismatch for {camera_name}")
        weights[camera_name] = np.any(image != 0, axis=2).astype(np.float32)

    x_coordinates = np.arange(output_width, dtype=np.float32)
    for overlap in overlaps:
        cameras = overlap.get("cameras", [])
        if len(cameras) != 2:
            raise ValueError("Each horizontal overlap must declare two cameras")
        left_camera, right_camera = str(cameras[0]), str(cameras[1])
        if left_camera not in weights or right_camera not in weights:
            continue

        x_range = overlap.get("x_range", [])
        if len(x_range) != 2:
            raise ValueError("Each horizontal overlap must declare x_range")
        overlap_start, overlap_end = sorted((float(x_range[0]), float(x_range[1])))
        seam_position = seam_x(stitch_points, str(overlap.get("seam", "")))
        if not overlap_start <= seam_position <= overlap_end:
            raise ValueError(
                f"Seam for {left_camera}/{right_camera} is outside its overlap"
            )

        feather_width = max(
            1.0,
            float(overlap.get("feather_width", default_feather_width)),
        )
        transition_start = max(overlap_start, seam_position - feather_width / 2.0)
        transition_end = min(overlap_end, seam_position + feather_width / 2.0)
        if transition_end <= transition_start:
            transition_start, transition_end = overlap_start, overlap_end

        in_overlap = (x_coordinates >= overlap_start) & (x_coordinates <= overlap_end)
        right_factor = np.clip(
            (x_coordinates - transition_start)
            / max(1.0, transition_end - transition_start),
            0.0,
            1.0,
        )
        left_factor = 1.0 - right_factor
        left_band = np.where(in_overlap, left_factor, 1.0)[None, :]
        right_band = np.where(in_overlap, right_factor, 1.0)[None, :]
        weights[left_camera] *= left_band
        weights[right_camera] *= right_band

    accumulator = np.zeros((output_height, output_width, 3), dtype=np.float32)
    total_weight = np.zeros((output_height, output_width), dtype=np.float32)
    for camera_name, image in warped_images.items():
        weight = weights[camera_name]
        accumulator += image.astype(np.float32) * weight[:, :, None]
        total_weight += weight

    canvas = np.zeros_like(accumulator)
    valid = total_weight > 1e-6
    canvas[valid] = accumulator[valid] / total_weight[valid][:, None]
    return np.clip(canvas, 0, 255).astype(np.uint8)
