"""Stitch-topology profile resolution and generic horizontal composition."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

import cv2
import numpy as np


LEGACY_TOPOLOGY_NAME = "legacy_surround_5"
RAW_FRAME_PIXEL_SPACE = "raw_frame_pixels"
MIN_SOURCE_COVERAGE = 0.8


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


def validate_overlap_seams(
    profile: dict[str, Any],
    stitch_points: dict[str, Any] | None = None,
) -> list[str]:
    """Return actionable errors for seams outside declared overlaps."""
    points = stitch_points or profile.get("stitch_points", {})
    errors: list[str] = []
    for overlap in profile.get("overlaps", []):
        cameras = [str(key) for key in overlap.get("cameras", [])]
        camera_pair = "/".join(cameras) or "unknown cameras"
        x_range = overlap.get("x_range", [])
        seam_name = str(overlap.get("seam", ""))
        if len(x_range) != 2:
            errors.append(
                f"{camera_pair}: overlap '{overlap.get('name', '')}' "
                "must declare two X values."
            )
            continue
        overlap_start, overlap_end = sorted(
            (float(x_range[0]), float(x_range[1]))
        )
        try:
            position = seam_x(points, seam_name)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{camera_pair}: invalid seam '{seam_name}': {exc}")
            continue
        endpoints = points.get(seam_name, [])
        canvas_height = float(profile.get("canvas", {}).get("height", 0))
        for index, point in enumerate(endpoints[:2], start=1):
            try:
                point_x, point_y = float(point[0]), float(point[1])
            except (IndexError, TypeError, ValueError):
                errors.append(
                    f"{camera_pair}: seam '{seam_name}' point {index} is invalid."
                )
                continue
            if not overlap_start <= point_x <= overlap_end:
                errors.append(
                    f"{camera_pair}: seam '{seam_name}' point {index} "
                    f"X={point_x:.3f} is outside overlap "
                    f"[{overlap_start:.3f}, {overlap_end:.3f}]."
                )
            if canvas_height > 0 and not 0.0 <= point_y <= canvas_height:
                errors.append(
                    f"{camera_pair}: seam '{seam_name}' point {index} "
                    f"Y={point_y:.3f} is outside canvas [0.000, "
                    f"{canvas_height:.3f}]."
                )
        if not overlap_start <= position <= overlap_end:
            errors.append(
                f"{camera_pair}: seam '{seam_name}' X={position:.3f} is "
                f"outside overlap [{overlap_start:.3f}, {overlap_end:.3f}]."
            )
    return errors


def source_coordinate_diagnostics(
    profile: dict[str, Any],
    camera_name: str,
    actual_raw_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Describe and validate the source-point coordinate contract."""
    camera = profile.get("cameras", {}).get(camera_name, {})
    coordinate_space = str(
        camera.get(
            "source_coordinate_space",
            profile.get("source_coordinate_space", ""),
        )
    )
    reference_value = camera.get(
        "source_reference_size",
        profile.get("source_reference_size"),
    )
    reference_size: list[int] | None = None
    if (
        isinstance(reference_value, (list, tuple))
        and len(reference_value) == 2
        and int(reference_value[0]) > 0
        and int(reference_value[1]) > 0
    ):
        reference_size = [int(reference_value[0]), int(reference_value[1])]

    points = camera.get("source_points", [])
    bounds: list[float] | None = None
    coverage: list[float] | None = None
    warnings: list[str] = []
    if coordinate_space and coordinate_space != RAW_FRAME_PIXEL_SPACE:
        warnings.append(
            f"{camera_name}: source coordinate space '{coordinate_space}' is "
            f"not the required '{RAW_FRAME_PIXEL_SPACE}' contract."
        )
    if not reference_size:
        warnings.append(
            f"{camera_name}: source reference size is missing or invalid; "
            "raw-frame source coordinates cannot be scale-checked."
        )
    if isinstance(points, list) and len(points) >= 4:
        try:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bounds = [min(xs), min(ys), max(xs), max(ys)]
        except (IndexError, TypeError, ValueError):
            warnings.append(f"{camera_name}: source points are invalid.")
    else:
        warnings.append(f"{camera_name}: source quadrilateral is incomplete.")

    if bounds is not None and reference_size is not None:
        reference_width, reference_height = reference_size
        coverage = [
            (bounds[2] - bounds[0]) / max(1.0, reference_width - 1.0),
            (bounds[3] - bounds[1]) / max(1.0, reference_height - 1.0),
        ]
        if (
            coverage[0] < MIN_SOURCE_COVERAGE
            or coverage[1] < MIN_SOURCE_COVERAGE
        ):
            warnings.append(
                f"{camera_name}: source quadrilateral covers only "
                f"{coverage[0] * 100:.1f}% width x "
                f"{coverage[1] * 100:.1f}% height of the "
                f"{reference_width}x{reference_height} reference frame; "
                "suspected normalized/raw coordinate mix."
            )

    actual_size = (
        [int(actual_raw_size[0]), int(actual_raw_size[1])]
        if actual_raw_size is not None
        else None
    )
    if (
        actual_size is not None
        and reference_size is not None
        and actual_size != reference_size
    ):
        warnings.append(
            f"{camera_name}: actual raw frame is "
            f"{actual_size[0]}x{actual_size[1]}, but source points reference "
            f"{reference_size[0]}x{reference_size[1]}; source coordinates "
            "must be rescaled or recalibrated before stitching."
        )

    return {
        "camera": camera_name,
        "coordinate_space": coordinate_space,
        "reference_size": reference_size,
        "contract_version": profile.get("source_contract_version"),
        "source_bounds": bounds,
        "coverage_fraction": coverage,
        "actual_raw_size": actual_size,
        "warnings": warnings,
    }


def _horizontal_feather_specs(
    camera_names: tuple[str, ...],
    stitch_points: dict[str, Any],
    overlaps: list[dict[str, Any]],
    default_feather_width: int,
) -> tuple[tuple[str, str, float, float, float, float], ...]:
    """Validate dynamic config and return an immutable cache key."""
    available = set(camera_names)
    specs: list[tuple[str, str, float, float, float, float]] = []
    for overlap in overlaps:
        cameras = overlap.get("cameras", [])
        if len(cameras) != 2:
            raise ValueError("Each horizontal overlap must declare two cameras")
        left_camera, right_camera = str(cameras[0]), str(cameras[1])
        if left_camera not in available or right_camera not in available:
            continue

        x_range = overlap.get("x_range", [])
        if len(x_range) != 2:
            raise ValueError("Each horizontal overlap must declare x_range")
        overlap_start, overlap_end = sorted(
            (float(x_range[0]), float(x_range[1]))
        )
        seam_position = seam_x(stitch_points, str(overlap.get("seam", "")))
        if not overlap_start <= seam_position <= overlap_end:
            raise ValueError(
                f"Seam for {left_camera}/{right_camera} is outside its overlap"
            )

        feather_width = max(
            1.0,
            float(overlap.get("feather_width", default_feather_width)),
        )
        transition_start = max(
            overlap_start,
            seam_position - feather_width / 2.0,
        )
        transition_end = min(
            overlap_end,
            seam_position + feather_width / 2.0,
        )
        if transition_end <= transition_start:
            transition_start, transition_end = overlap_start, overlap_end
        specs.append(
            (
                left_camera,
                right_camera,
                overlap_start,
                overlap_end,
                transition_start,
                transition_end,
            )
        )
    return tuple(specs)


def _feather_column_regions(
    affected_columns: np.ndarray,
) -> tuple[tuple[int, int, bool], ...]:
    """Partition columns into base-weight and feather-weight runs."""
    width = int(affected_columns.size)
    if width == 0:
        return ()
    changes = np.flatnonzero(affected_columns[1:] != affected_columns[:-1]) + 1
    boundaries = (0, *(int(value) for value in changes), width)
    return tuple(
        (
            boundaries[index],
            boundaries[index + 1],
            bool(affected_columns[boundaries[index]]),
        )
        for index in range(len(boundaries) - 1)
    )


@lru_cache(maxsize=32)
def _cached_horizontal_feather_plan(
    output_width: int,
    camera_names: tuple[str, ...],
    specs: tuple[tuple[str, str, float, float, float, float], ...],
) -> dict[
    str,
    tuple[
        np.ndarray,
        tuple[np.ndarray, ...],
        tuple[tuple[int, int, bool], ...],
    ],
]:
    """Build immutable 1-D feather bands shared by successive frames."""
    x_coordinates = np.arange(output_width, dtype=np.float32)
    combined_bands = {
        camera_name: np.ones(output_width, dtype=np.float32)
        for camera_name in camera_names
    }
    band_operations: dict[str, list[np.ndarray]] = {
        camera_name: [] for camera_name in camera_names
    }
    affected_columns = {
        camera_name: np.zeros(output_width, dtype=bool)
        for camera_name in camera_names
    }

    for (
        left_camera,
        right_camera,
        overlap_start,
        overlap_end,
        transition_start,
        transition_end,
    ) in specs:
        in_overlap = (x_coordinates >= overlap_start) & (
            x_coordinates <= overlap_end
        )
        right_factor = np.clip(
            (x_coordinates - transition_start)
            / max(1.0, transition_end - transition_start),
            0.0,
            1.0,
        )
        left_factor = 1.0 - right_factor
        left_band = np.where(in_overlap, left_factor, 1.0)
        right_band = np.where(in_overlap, right_factor, 1.0)
        combined_bands[left_camera] *= left_band
        combined_bands[right_camera] *= right_band
        band_operations[left_camera].append(left_band)
        band_operations[right_camera].append(right_band)
        affected_columns[left_camera] |= in_overlap
        affected_columns[right_camera] |= in_overlap

    plan = {}
    for camera_name in camera_names:
        combined_band = combined_bands[camera_name]
        operations = tuple(band_operations[camera_name])
        combined_band.flags.writeable = False
        for operation in operations:
            operation.flags.writeable = False
        plan[camera_name] = (
            combined_band,
            operations,
            _feather_column_regions(affected_columns[camera_name]),
        )
    return plan


def _can_use_boolean_mask_roi_path(
    image: np.ndarray,
    mask: np.ndarray,
) -> bool:
    return bool(
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and image.shape[0] > 0
        and image.shape[1] > 0
        and image.flags.c_contiguous
        and mask.dtype == np.bool_
        and mask.flags.c_contiguous
    )


def compose_horizontal_feather(
    warped_images: dict[str, np.ndarray],
    output_width: int,
    output_height: int,
    stitch_points: dict[str, Any],
    overlaps: list[dict[str, Any]],
    default_feather_width: int,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Blend available left-to-right cameras using profile-declared overlaps."""
    if not warped_images:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8)

    prepared_masks: dict[str, np.ndarray] = {}
    masks = valid_masks or {}
    for camera_name, image in warped_images.items():
        if image.shape[:2] != (output_height, output_width):
            raise ValueError(f"Warped image size mismatch for {camera_name}")
        mask = masks.get(camera_name)
        if mask is None:
            mask = np.any(image != 0, axis=2)
        else:
            mask = np.asarray(mask)
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Valid mask size mismatch for {camera_name}")
        prepared_masks[camera_name] = mask

    camera_names = tuple(str(camera_name) for camera_name in warped_images)
    specs = _horizontal_feather_specs(
        camera_names,
        stitch_points,
        overlaps,
        default_feather_width,
    )
    plan = _cached_horizontal_feather_plan(output_width, camera_names, specs)

    accumulator = np.zeros((output_height, output_width, 3), dtype=np.float32)
    total_weight = np.zeros((output_height, output_width), dtype=np.float32)
    used_generic_path = False
    for camera_name, image in warped_images.items():
        mask = prepared_masks[camera_name]
        combined_band, band_operations, column_regions = plan[camera_name]
        if _can_use_boolean_mask_roi_path(image, mask):
            mask_u8 = mask.view(np.uint8)
            for start, end, is_feather_region in column_regions:
                if end <= start:
                    continue
                image_region = image[:, start:end]
                mask_region = mask[:, start:end]
                accumulator_region = accumulator[:, start:end]
                total_weight_region = total_weight[:, start:end]
                if not is_feather_region:
                    cv2.accumulate(
                        image_region,
                        accumulator_region,
                        mask=mask_u8[:, start:end],
                    )
                    cv2.accumulate(
                        mask_u8[:, start:end],
                        total_weight_region,
                    )
                    continue

                weight = mask_region.astype(np.float32)
                weight *= combined_band[None, start:end]
                product = image_region.astype(np.float32)
                product *= weight[:, :, None]
                accumulator_region += product
                total_weight_region += weight
            continue

        # Preserve the generic ndarray behavior for uncommon dtypes/layouts.
        # Individual cached operations retain the original multiplication
        # order, including when overlap bands intersect.
        used_generic_path = True
        weight = mask.astype(np.float32, copy=True)
        for band in band_operations:
            weight *= band[None, :]
        accumulator += image.astype(np.float32) * weight[:, :, None]
        total_weight += weight

    valid = total_weight > 1e-6
    np.divide(
        accumulator,
        total_weight[:, :, None],
        out=accumulator,
        where=valid[:, :, None],
    )
    if used_generic_path:
        accumulator[~valid] = 0.0
    np.clip(accumulator, 0, 255, out=accumulator)
    return accumulator.astype(np.uint8)
