"""Manual runtime A/B comparison exports for Near-field projection sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.config import PROJECT_ROOT, save_yaml
from deep_shark_studio.seam.layout_candidate_runtime import LayoutRuntimeCandidate
from deep_shark_studio.stitch_runtime_controller import RuntimeStitchController
from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)
from deep_shark_studio.stitcher import save_image


@dataclass(frozen=True)
class RuntimeABComparisonResult:
    output_dir: Path
    files: dict[str, Path]
    report: dict[str, Any]


def generate_runtime_ab_comparison(
    frames: dict[str, np.ndarray],
    stitcher: Any,
    layout_candidate: LayoutRuntimeCandidate,
    fisheye_intrinsics_source_path: str | Path,
    output_root: str | Path | None = None,
    fisheye_balance: float = 0.6,
    fisheye_fov_scale: float = 1.0,
) -> RuntimeABComparisonResult:
    """Export Far-field, Near-field current, and Near-field fisheye canvases."""
    root = Path(output_root or PROJECT_ROOT / "projects" / "runtime_ab_comparisons")
    run_dir = _create_run_directory(root)

    far = RuntimeStitchController(
        stitcher,
        RuntimeStitchConfig(mode=StitchRuntimeMode.FAR_FIELD),
    ).process(frames)
    near_current = RuntimeStitchController(
        stitcher,
        RuntimeStitchConfig(
            mode=StitchRuntimeMode.NEAR_FIELD,
            projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
        ),
        layout_candidate=layout_candidate,
    ).process(frames)
    near_fisheye = RuntimeStitchController(
        stitcher,
        RuntimeStitchConfig(
            mode=StitchRuntimeMode.NEAR_FIELD,
            projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
            projection_intrinsics_source_path=Path(fisheye_intrinsics_source_path),
            fisheye_balance=float(fisheye_balance),
            fisheye_fov_scale=float(fisheye_fov_scale),
        ),
        layout_candidate=layout_candidate,
    ).process(frames)

    files = {
        "far_field": run_dir / "far_field.png",
        "near_current_perspective": run_dir / "near_current_perspective.png",
        "near_fisheye_rectilinear": run_dir / "near_fisheye_rectilinear.png",
        "side_by_side": run_dir / "side_by_side.png",
        "report": run_dir / "comparison_report.yaml",
    }
    if far.canvas is not None:
        save_image(files["far_field"], far.canvas)
    if near_current.canvas is not None:
        save_image(files["near_current_perspective"], near_current.canvas)
    if near_fisheye.canvas is not None:
        save_image(files["near_fisheye_rectilinear"], near_fisheye.canvas)
    side_by_side = _side_by_side(
        [
            far.canvas,
            near_current.canvas,
            near_fisheye.canvas,
        ]
    )
    save_image(files["side_by_side"], side_by_side)

    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
        "files": {key: path.name for key, path in files.items()},
        "timing": {
            "far_field_total_ms": _timing(far.metrics, "far_field_total_ms"),
            "near_current_total_ms": _timing(near_current.metrics, "near_field_total_ms"),
            "near_fisheye_total_ms": _timing(near_fisheye.metrics, "near_field_total_ms"),
            "near_fisheye_remap_ms": _timing(near_fisheye.metrics, "fisheye_remap_ms"),
            "near_fisheye_template_warp_ms": _timing(near_fisheye.metrics, "template_warp_ms"),
        },
        "projection": {
            "fisheye_balance": float(fisheye_balance),
            "fov_scale": float(fisheye_fov_scale),
            "intrinsics_source": str(fisheye_intrinsics_source_path),
            "fisheye_metadata": (near_fisheye.metrics or {}).get("projection", {}),
        },
        "warnings": list(far.warnings)
        + list(near_current.warnings)
        + list(near_fisheye.warnings),
    }
    save_yaml(files["report"], report)
    return RuntimeABComparisonResult(output_dir=run_dir, files=files, report=report)


def _create_run_directory(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = root / timestamp
    parent.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        run = parent / f"run_{timestamp[-6:]}_{index:03d}"
        if not run.exists():
            run.mkdir(parents=True)
            return run
        index += 1


def _side_by_side(images: list[np.ndarray | None]) -> np.ndarray:
    available = [image for image in images if image is not None]
    if not available:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height = min(image.shape[0] for image in available)
    normalized = []
    for image in available:
        if image.shape[0] != height:
            width = max(1, int(round(image.shape[1] * height / image.shape[0])))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        normalized.append(image)
    return np.concatenate(normalized, axis=1)


def _timing(metrics: dict[str, Any] | None, key: str) -> float | None:
    timing = metrics.get("timing", {}) if isinstance(metrics, dict) else {}
    value = timing.get(key) if isinstance(timing, dict) else None
    return float(value) if isinstance(value, (int, float)) else None
