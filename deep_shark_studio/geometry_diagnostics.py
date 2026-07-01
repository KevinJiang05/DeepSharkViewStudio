"""Read-only geometry diagnostics for multi-camera stitch profiles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .stitcher import SurroundStitcher
from .topology import (
    active_stitch_profile,
    seam_x,
    source_coordinate_diagnostics,
)


ALPHA_50 = "alpha_50"
NO_FEATHER = "no_feather"


@dataclass(frozen=True)
class GeometryDiagnosticResult:
    warped: dict[str, np.ndarray]
    pair_only: dict[str, np.ndarray]
    final_canvas: np.ndarray | None
    final_error: str
    metadata: dict[str, Any]


def normalized_input_size(
    frame: np.ndarray,
    max_input_width: int | None,
) -> tuple[int, int, float]:
    """Mirror the resize performed by SurroundStitcher.warp()."""
    raw_height, raw_width = frame.shape[:2]
    if max_input_width and raw_width > max_input_width:
        scale = max_input_width / float(raw_width)
        return max_input_width, int(raw_height * scale), scale
    return raw_width, raw_height, 1.0


def compose_pair_only(
    left: np.ndarray,
    right: np.ndarray,
    mode: str,
    seam_position: float,
) -> np.ndarray:
    """Compose exactly two warped cameras without profile feathering."""
    if left.shape != right.shape:
        raise ValueError("Pair-only warped images must have matching sizes")
    left_valid = np.any(left != 0, axis=2)
    right_valid = np.any(right != 0, axis=2)
    both = left_valid & right_valid
    result = np.zeros_like(left)
    result[left_valid] = left[left_valid]
    result[right_valid & ~left_valid] = right[right_valid & ~left_valid]

    if mode == ALPHA_50:
        blended = (
            left[both].astype(np.float32) * 0.5
            + right[both].astype(np.float32) * 0.5
        )
        result[both] = np.clip(blended, 0, 255).astype(np.uint8)
        return result
    if mode != NO_FEATHER:
        raise ValueError(f"Unknown geometry diagnostic mode: {mode}")

    split = int(round(seam_position))
    split = max(0, min(result.shape[1], split))
    columns = np.arange(result.shape[1])[None, :]
    use_right = right_valid & ((columns >= split) | ~left_valid)
    result[use_right] = right[use_right]
    return result


def _annotate_geometry(
    image: np.ndarray,
    title: str,
    overlaps: list[dict[str, Any]],
    stitch_points: dict[str, Any],
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    colors = ((244, 63, 94), (249, 115, 22), (34, 197, 94))
    cv2.putText(
        annotated,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, overlap in enumerate(overlaps):
        x_range = overlap.get("x_range", [])
        if len(x_range) != 2:
            continue
        start, end = sorted((int(round(x_range[0])), int(round(x_range[1]))))
        color = colors[index % len(colors)]
        cv2.rectangle(
            annotated,
            (max(0, start), 0),
            (min(width - 1, end), height - 1),
            color,
            2,
        )
        seam_name = str(overlap.get("seam", ""))
        try:
            seam_position = int(round(seam_x(stitch_points, seam_name)))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= seam_position < width:
            cv2.line(
                annotated,
                (seam_position, 0),
                (seam_position, height - 1),
                color,
                2,
            )
        pair = " <-> ".join(str(key) for key in overlap.get("cameras", []))
        cv2.putText(
            annotated,
            f"{pair} X=[{start},{end}] seam={seam_position}",
            (max(4, start + 4), 62 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_geometry_diagnostics(
    frames: dict[str, np.ndarray],
    calibration_config: dict[str, Any],
    max_input_width: int | None,
    use_intrinsics: bool,
    pair_mode: str = ALPHA_50,
) -> GeometryDiagnosticResult:
    """Generate diagnostic images without mutating frames or profile data."""
    profile = deepcopy(active_stitch_profile(calibration_config))
    stitcher = SurroundStitcher(
        calibration_config,
        max_input_width=max_input_width,
        use_intrinsics=use_intrinsics,
    )
    active_keys = list(stitcher.active_camera_keys)
    selected_frames = {
        key: frames[key]
        for key in active_keys
        if key in frames and frames[key] is not None
    }
    warped_base = stitcher.warp_all(selected_frames)
    overlaps = deepcopy(profile.get("overlaps", []))
    stitch_points = deepcopy(profile.get("stitch_points", {}))

    camera_metadata: dict[str, Any] = {}
    warped: dict[str, np.ndarray] = {}
    for key in active_keys:
        frame = selected_frames.get(key)
        calibration = profile.get("cameras", {}).get(key, {})
        if frame is None:
            source_diagnostics = source_coordinate_diagnostics(profile, key)
            camera_metadata[key] = {
                "available": False,
                "source_point_space": source_diagnostics["coordinate_space"],
                "source_reference_size": source_diagnostics["reference_size"],
                "source_contract_version": source_diagnostics["contract_version"],
                "source_bounds": source_diagnostics["source_bounds"],
                "source_coverage_fraction": source_diagnostics["coverage_fraction"],
                "coordinate_warnings": source_diagnostics["warnings"],
                "editor_point_space": "loaded_image_pixels",
            }
            continue
        raw_height, raw_width = frame.shape[:2]
        normalized_width, normalized_height, scale = normalized_input_size(
            frame,
            max_input_width,
        )
        source_points = calibration.get("source_points", [])
        source_diagnostics = source_coordinate_diagnostics(
            profile,
            key,
            (raw_width, raw_height),
        )
        intrinsics = stitcher.camera_intrinsics.get(key)
        intrinsics_reference = (
            intrinsics.get("image_size")
            if isinstance(intrinsics, dict)
            else None
        )
        effective_intrinsics = bool(use_intrinsics and intrinsics)
        intrinsics_scale_to_raw = None
        intrinsics_scale_to_normalized = None
        intrinsics_uniform_scale_compatible = False
        intrinsics_warning = ""
        if (
            isinstance(intrinsics_reference, (list, tuple))
            and len(intrinsics_reference) == 2
            and float(intrinsics_reference[0]) > 0
            and float(intrinsics_reference[1]) > 0
        ):
            reference_width = float(intrinsics_reference[0])
            reference_height = float(intrinsics_reference[1])
            intrinsics_scale_to_raw = [
                raw_width / reference_width,
                raw_height / reference_height,
            ]
            intrinsics_scale_to_normalized = [
                normalized_width / reference_width,
                normalized_height / reference_height,
            ]
            intrinsics_uniform_scale_compatible = (
                abs(
                    intrinsics_scale_to_raw[0]
                    - intrinsics_scale_to_raw[1]
                )
                < 1e-6
            )
            if effective_intrinsics and (
                normalized_width != int(reference_width)
                or normalized_height != int(reference_height)
            ):
                intrinsics_warning = (
                    "Saved camera matrix reference size differs from the "
                    "normalized image passed to undistort_image(); the current "
                    "runtime does not rescale the camera matrix."
                )
        camera_metadata[key] = {
            "available": True,
            "raw_size": [raw_width, raw_height],
            "normalized_size": [normalized_width, normalized_height],
            "raw_to_normalized_scale": scale,
            "normalized_to_source_factor": 1.0 / scale,
            "source_point_space": source_diagnostics["coordinate_space"],
            "source_reference_size": source_diagnostics["reference_size"],
            "source_contract_version": source_diagnostics["contract_version"],
            "editor_point_space": "loaded_image_pixels",
            "source_points": deepcopy(source_points),
            "source_extent": (
                source_diagnostics["source_bounds"][2:]
                if source_diagnostics["source_bounds"] is not None
                else None
            ),
            "source_bounds": source_diagnostics["source_bounds"],
            "source_coverage_fraction": source_diagnostics["coverage_fraction"],
            "intrinsics_available": bool(intrinsics),
            "intrinsics_enabled": bool(use_intrinsics),
            "intrinsics_effective": effective_intrinsics,
            "intrinsics_reference_size": deepcopy(intrinsics_reference),
            "intrinsics_scale_to_raw": intrinsics_scale_to_raw,
            "intrinsics_scale_to_normalized": intrinsics_scale_to_normalized,
            "intrinsics_uniform_scale_compatible": (
                intrinsics_uniform_scale_compatible
            ),
            "intrinsics_runtime_matrix_scaled": False,
            "intrinsics_warning": intrinsics_warning,
            "coordinate_warnings": source_diagnostics["warnings"],
        }
        relevant = [
            overlap
            for overlap in overlaps
            if key in overlap.get("cameras", [])
        ]
        warped[key] = _annotate_geometry(
            warped_base[key],
            f"Warp: {key}",
            relevant,
            stitch_points,
        )

    pair_only: dict[str, np.ndarray] = {}
    pair_metadata: list[dict[str, Any]] = []
    for overlap in overlaps:
        cameras = [str(key) for key in overlap.get("cameras", [])]
        if len(cameras) != 2:
            continue
        left_key, right_key = cameras
        pair_name = str(overlap.get("name") or f"{left_key}_{right_key}")
        pair_metadata.append(
            {
                "name": pair_name,
                "cameras": cameras,
                "x_range": deepcopy(overlap.get("x_range", [])),
                "seam": overlap.get("seam", ""),
                "seam_x": seam_x(
                    stitch_points,
                    str(overlap.get("seam", "")),
                ),
            }
        )
        if left_key not in warped_base or right_key not in warped_base:
            continue
        base = compose_pair_only(
            warped_base[left_key],
            warped_base[right_key],
            pair_mode,
            pair_metadata[-1]["seam_x"],
        )
        pair_only[pair_name] = _annotate_geometry(
            base,
            f"Pair only: {left_key} <-> {right_key} ({pair_mode})",
            [overlap],
            stitch_points,
        )

    final_canvas: np.ndarray | None = None
    final_error = ""
    try:
        final_canvas = stitcher.stitch(warped_base)
        final_canvas = _annotate_geometry(
            final_canvas,
            "Current final canvas",
            overlaps,
            stitch_points,
        )
    except Exception as exc:
        final_error = str(exc)

    metadata = {
        "topology": stitcher.topology_name,
        "camera_order": active_keys,
        "canvas": {
            "width": stitcher.output_width,
            "height": stitcher.output_height,
        },
        "max_input_width": max_input_width,
        "runtime_stage_order": [
            "resize_to_max_input_width",
            "optional_undistort",
            "perspective_warp",
        ],
        "source_point_space": profile.get("source_coordinate_space", ""),
        "source_reference_size": deepcopy(profile.get("source_reference_size")),
        "source_contract_version": profile.get("source_contract_version"),
        "editor_point_space": "loaded_image_pixels",
        "use_intrinsics": bool(use_intrinsics),
        "cameras": camera_metadata,
        "pairs": pair_metadata,
        "overlaps": overlaps,
        "stitch_points": stitch_points,
        "pair_mode": pair_mode,
    }
    return GeometryDiagnosticResult(
        warped=warped,
        pair_only=pair_only,
        final_canvas=final_canvas,
        final_error=final_error,
        metadata=metadata,
    )
