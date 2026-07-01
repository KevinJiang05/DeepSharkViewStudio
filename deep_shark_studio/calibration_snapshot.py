"""Manual calibration snapshot artifacts for topology tuning."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import save_yaml
from .stitcher import save_image
from .topology import (
    active_stitch_profile,
    active_topology_camera_keys,
    seam_x,
    selected_topology_name,
)


SNAPSHOT_FORMAT = "DeepSharkCalibrationSnapshot"
SNAPSHOT_SCHEMA_VERSION = 1


def topology_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Return the current profile geometry without connection information."""
    profile = active_stitch_profile(config)
    stitch_points = deepcopy(profile.get("stitch_points", {}))
    overlaps: list[dict[str, Any]] = []
    for overlap in profile.get("overlaps", []):
        item = deepcopy(overlap)
        seam_name = str(item.get("seam", ""))
        try:
            item["seam_x"] = seam_x(stitch_points, seam_name)
        except (KeyError, TypeError, ValueError):
            item["seam_x"] = None
        overlaps.append(item)

    cameras = {}
    for key in active_topology_camera_keys(config):
        camera = profile.get("cameras", {}).get(key, {})
        cameras[key] = {
            "source_points": deepcopy(camera.get("source_points", [])),
            "target_points": deepcopy(camera.get("target_points", [])),
        }

    return {
        "topology": selected_topology_name(config),
        "camera_order": active_topology_camera_keys(config),
        "canvas": deepcopy(profile.get("canvas", {})),
        "cameras": cameras,
        "overlaps": overlaps,
        "seams": stitch_points,
        "composition": deepcopy(profile.get("composition", {})),
        "calibration_origin": deepcopy(
            profile.get(
                "calibration_origin",
                {
                    "source_points": "unmarked",
                    "target_points": "unmarked",
                    "seams": "unmarked",
                    "feather": "unmarked",
                },
            )
        ),
    }


def build_snapshot_metadata(
    calibration_config: dict[str, Any],
    available_camera_keys: list[str],
    frame_ages_seconds: dict[str, float | None],
    snapshot_time: str,
    processing_error: str = "",
) -> dict[str, Any]:
    """Build a portable, connection-secret-free snapshot description."""
    diagnostics = topology_diagnostics(calibration_config)
    camera_order = diagnostics["camera_order"]
    available = [key for key in camera_order if key in available_camera_keys]
    missing = [key for key in camera_order if key not in available]
    raw_files = {key: f"raw_{key}.png" for key in available}
    return {
        "format": SNAPSHOT_FORMAT,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_time": snapshot_time,
        "topology": diagnostics["topology"],
        "camera_order": camera_order,
        "available_cameras": available,
        "missing_cameras": missing,
        "frame_age_seconds": {
            key: frame_ages_seconds.get(key)
            for key in camera_order
        },
        "canvas": diagnostics["canvas"],
        "cameras": diagnostics["cameras"],
        "overlaps": diagnostics["overlaps"],
        "seams": diagnostics["seams"],
        "composition": diagnostics["composition"],
        "calibration_origin": diagnostics["calibration_origin"],
        "processing_error": processing_error,
        "files": {
            "raw": raw_files,
            "canvas": "canvas.png",
            "metadata": "metadata.yaml",
        },
    }


def save_calibration_snapshot(
    snapshots_root: str | Path,
    frames: dict[str, np.ndarray],
    canvas: np.ndarray,
    calibration_config: dict[str, Any],
    frame_ages_seconds: dict[str, float | None],
    processing_error: str = "",
    captured_at: datetime | None = None,
) -> Path:
    """Write one user-triggered snapshot into a never-overwritten directory."""
    captured_at = captured_at or datetime.now().astimezone()
    snapshot_time = captured_at.isoformat(timespec="milliseconds")
    directory_name = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    root = Path(snapshots_root)
    snapshot_dir = root / directory_name
    suffix = 1
    while snapshot_dir.exists():
        snapshot_dir = root / f"{directory_name}_{suffix:02d}"
        suffix += 1
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    camera_order = active_topology_camera_keys(calibration_config)
    available_frames = {
        key: frames[key]
        for key in camera_order
        if key in frames and frames[key] is not None
    }
    for key, frame in available_frames.items():
        save_image(snapshot_dir / f"raw_{key}.png", frame)
    save_image(snapshot_dir / "canvas.png", canvas)

    metadata = build_snapshot_metadata(
        calibration_config=calibration_config,
        available_camera_keys=list(available_frames),
        frame_ages_seconds=frame_ages_seconds,
        snapshot_time=snapshot_time,
        processing_error=processing_error,
    )
    save_yaml(snapshot_dir / "metadata.yaml", metadata)
    return snapshot_dir
