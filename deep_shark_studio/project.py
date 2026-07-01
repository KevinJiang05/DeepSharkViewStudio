"""Project file and configuration backup helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR, load_config, load_yaml, save_config, save_yaml


PROJECT_FORMAT = "DeepSharkViewStudioProject"
PROJECT_VERSION = 1


def current_project_payload() -> dict[str, Any]:
    """Return a portable project payload containing all editable configs."""
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "configs": {
            "calibration.yaml": load_config("calibration.yaml"),
            "cameras.yaml": load_config("cameras.yaml"),
            "network.yaml": load_config("network.yaml"),
        },
    }


def save_project(path: str | Path) -> Path:
    project_path = Path(path)
    if project_path.suffix.lower() not in {".yaml", ".yml"}:
        project_path = project_path.with_suffix(".dsvs.yaml")
    save_yaml(project_path, current_project_payload())
    return project_path


def load_project(path: str | Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if payload.get("format") != PROJECT_FORMAT:
        raise ValueError("Not a DeepShark View Studio project file.")
    configs = payload.get("configs", {})
    for name in ("calibration.yaml", "cameras.yaml", "network.yaml"):
        if name in configs:
            save_config(name, configs[name])
    return payload


def backup_configs(backups_dir: str | Path | None = None) -> Path:
    backup_root = Path(backups_dir) if backups_dir else CONFIG_DIR.parent / "projects" / "backups"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_root / f"backup_{timestamp}.dsvs.yaml"
    return save_project(backup_path)


def export_runtime_config(path: str | Path) -> Path:
    """Export the compact runtime subset expected by a future service/QGC bridge."""
    calibration = load_config("calibration.yaml")
    cameras = load_config("cameras.yaml")
    network = load_config("network.yaml")
    runtime = {
        "format": "DeepSharkViewRuntimeConfig",
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "active_camera_count": cameras.get("active_camera_count", calibration.get("active_camera_count", 5)),
        "camera_order": cameras.get("camera_order", calibration.get("camera_order", [])),
        "canvas": calibration.get("canvas", {}),
        "cameras": cameras.get("cameras", {}),
        "perspective_cameras": calibration.get("cameras", {}),
        "camera_intrinsics": calibration.get("camera_intrinsics", {}),
        "stitch_points": calibration.get("stitch_points", {}),
        "mask_rules": calibration.get("mask_rules", []),
        "network": network,
    }
    export_path = Path(path)
    save_yaml(export_path, runtime)
    return export_path
