"""Runner for report-only fisheye projection candidate previews."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.config import CONFIG_DIR, PROJECT_ROOT, file_revision, load_config, load_yaml, save_yaml
from deep_shark_studio.seam.layout_preview import LayoutPairCandidate, LayoutPreviewParams
from deep_shark_studio.seam.layout_sweep import (
    load_layout_candidates_from_run,
    run_front_priority_layout_sweep,
)
from deep_shark_studio.stitcher import SurroundStitcher

from .fisheye_projection import (
    EquirectangularParams,
    build_equirectangular_projection,
    build_rectilinear_projection,
)
from .intrinsics_candidate import CAMERA_KEYS, consolidate_intrinsics_candidates
from .projection_preview import (
    projection_metrics,
    render_fixed_layout,
    write_projection_comparison,
)


@dataclass(frozen=True)
class ProjectionCandidateRunResult:
    output_dir: Path
    report_path: Path
    intrinsics_report_path: Path
    generated_modes: dict[str, str]


def generate_projection_candidate_preview(
    session_dir: Path,
    output_root: Path,
    projection_mode: str = "equirectangular_rotation_only",
    profile_id: str = "triple_front_panorama",
    baseline_candidate_run_dir: Path | None = None,
    snapshot_dir: Path | None = None,
) -> ProjectionCandidateRunResult:
    """Generate projection previews without modifying formal runtime/config."""
    before = file_revision(CONFIG_DIR / "calibration.yaml")
    run_dir = _create_run_directory(output_root)
    config = load_config("calibration.yaml")
    frames, frame_source = _load_preview_frames(session_dir, snapshot_dir)
    intrinsics = consolidate_intrinsics_candidates(
        session_dir=session_dir,
        output_dir=run_dir,
        profile_id=profile_id,
    )
    candidate_data = {}
    if intrinsics.source_candidate_dir is not None:
        candidate_data = load_yaml(intrinsics.source_candidate_dir / "candidate.yaml")
    all_intrinsics = intrinsics.report["all_required_intrinsics_available"]
    generated_modes: dict[str, str] = {}

    rectilinear_status = "skipped"
    rectilinear_report: dict[str, Any] = {
        "status": "skipped",
        "warnings": [],
    }
    if all_intrinsics:
        rectilinear_report = build_rectilinear_projection(
            frames,
            intrinsics.models,
            run_dir / "rectilinear",
        )
        rectilinear_status = "generated"
        save_yaml(run_dir / "rectilinear" / "rectilinear_report.yaml", rectilinear_report)
        generated_modes["rectilinear"] = rectilinear_status
    else:
        rectilinear_report["warnings"].append("missing opencv_fisheye K/D for at least one active camera")

    equirect_status = "skipped"
    equirect_metadata: dict[str, Any] = {
        "status": "skipped",
        "warnings": [],
    }
    projection_result = None
    if all_intrinsics and projection_mode == "equirectangular_rotation_only":
        transforms = candidate_data.get("rig_transforms", {})
        if not transforms:
            transforms = candidate_data.get("camera_to_front_transforms", {})
        if not transforms:
            transforms = candidate_data.get("rig", {}).get("transforms", {})
        if not transforms:
            equirect_metadata["warnings"].append("missing B-2 rotation transforms")
        else:
            projection_result = build_equirectangular_projection(
                frames,
                intrinsics.models,
                transforms,
                run_dir / "equirectangular",
                EquirectangularParams(),
            )
            equirect_metadata = projection_result.metadata
            equirect_status = "generated"
            save_yaml(run_dir / "equirectangular" / "equirectangular_report.yaml", equirect_metadata)
            generated_modes["equirectangular"] = equirect_status
    else:
        equirect_metadata["warnings"].append("missing K/D or unsupported projection mode")

    baseline_candidate_run_dir = baseline_candidate_run_dir or _latest_front_priority_candidate_run()
    layout_params = LayoutPreviewParams(40, 1800, 0.30, 24)
    comparison_files: dict[str, str] = {}
    fixed_metrics: dict[str, Any] = {}
    mini_sweep_report: str | None = None
    projection_preview_available = projection_result is not None and bool(projection_result.warped_images)
    perspective_baseline_available = False

    if projection_preview_available:
        projection_candidates = _pair_candidates_from_equirectangular_metadata(
            projection_result.metadata,
            projection_result.warped_images["front"].shape[0],
        )
        fixed_image, fixed_metrics = render_fixed_layout(
            projection_result.warped_images,
            projection_candidates,
            layout_params,
        )
        fixed_dir = run_dir / "front_priority_layout" / "fixed_layout"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fixed_dir / "projection_front_priority_fixed_layout.png"), fixed_image)

        mini = run_front_priority_layout_sweep(
            projection_result.warped_images,
            projection_candidates,
            run_dir / "front_priority_layout" / "mini_sweep" / "projection_layout_sweep",
            profile_id=profile_id,
            side_shifts_px=[0, 40, 80],
            output_widths_px=[1800, 1900, 2000],
            side_visible_fractions=[0.25, 0.30, 0.35],
            feather_widths=[16, 24],
            source_note="Projection candidate warped_images; canvas may differ from formal perspective baseline.",
        )
        mini_sweep_report = str(mini.report_path.relative_to(run_dir))
        _copy_layout_report_name(mini.report_path)

        baseline_image = None
        if baseline_candidate_run_dir is not None:
            baseline_candidates = load_layout_candidates_from_run(baseline_candidate_run_dir)
            stitcher = SurroundStitcher(config, max_input_width=None, use_intrinsics=False)
            perspective_warped = stitcher.warp_all(frames)
            if all(camera in perspective_warped for camera in CAMERA_KEYS):
                baseline_image, _ = render_fixed_layout(
                    perspective_warped,
                    baseline_candidates,
                    layout_params,
                )
                perspective_baseline_available = True
        if baseline_image is not None:
            comparison_files = write_projection_comparison(
                run_dir / "comparison",
                baseline_image,
                fixed_image,
            )

    report = {
        "schema_version": 1,
        "profile_id": profile_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
        "frame_source": frame_source,
        "intrinsics_status": {
            "all_required_intrinsics_available": all_intrinsics,
            "warnings": _intrinsics_warnings(intrinsics.report),
        },
        "projection_modes": {
            "rectilinear": {
                "status": rectilinear_status,
                "warnings": rectilinear_report.get("warnings", []),
            },
            "equirectangular": {
                "status": equirect_status,
                "warnings": equirect_metadata.get("warnings", []),
            },
        },
        "front_priority_layout": {
            "fixed_layout_generated": projection_preview_available,
            "fixed_layout_file": "front_priority_layout/fixed_layout/projection_front_priority_fixed_layout.png" if projection_preview_available else None,
            "mini_sweep_report": mini_sweep_report,
        },
        "comparison": {
            "perspective_baseline_available": perspective_baseline_available,
            "projection_preview_available": projection_preview_available,
            "subjective_review_required": True,
            "files": comparison_files,
        },
        "metrics": _combined_metrics(projection_result, fixed_metrics),
        "limitations": [
            "Projection candidate is report-only and not applied to formal runtime.",
            "Rotation-only equirectangular projection does not solve near-field non-common-center parallax.",
            "Software-near-time snapshots are not hardware synchronized.",
        ],
    }
    save_yaml(run_dir / "projection_candidate_report.yaml", report)
    after = file_revision(CONFIG_DIR / "calibration.yaml")
    if after != before:
        raise RuntimeError("Formal calibration.yaml changed during projection candidate generation.")
    return ProjectionCandidateRunResult(
        output_dir=run_dir,
        report_path=run_dir / "projection_candidate_report.yaml",
        intrinsics_report_path=run_dir / "intrinsics_candidates_report.yaml",
        generated_modes=generated_modes,
    )


def _create_run_directory(output_root: Path) -> Path:
    created = datetime.now().astimezone()
    session_id = created.strftime("%Y%m%d_%H%M%S")
    run_id = created.strftime("run_%H%M%S_%f")[:-3]
    root = Path(output_root)
    directory = root / session_id / run_id
    suffix = 1
    while directory.exists():
        directory = root / session_id / f"{run_id}_{suffix:02d}"
        suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _load_preview_frames(session_dir: Path, snapshot_dir: Path | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = snapshot_dir or _latest_snapshot_dir()
    if directory is not None:
        metadata_path = directory / "metadata.yaml"
        metadata = load_yaml(metadata_path) if metadata_path.exists() else {}
        raw_files = metadata.get("files", {}).get("raw", {})
        frames: dict[str, np.ndarray] = {}
        for camera in CAMERA_KEYS:
            filename = raw_files.get(camera, f"raw_{camera}.png")
            image = cv2.imread(str(directory / filename))
            if image is not None:
                frames[camera] = image
        if all(camera in frames for camera in CAMERA_KEYS):
            return frames, {
                "source": "calibration_snapshot",
                "snapshot_dir": str(directory),
                "synchronized": False,
                "frame_age_seconds": metadata.get("frame_age_seconds", {}),
                "warning": "Software-near-time snapshot, not hardware synchronized.",
            }
    # Fallback to latest independent intrinsic accepted raw frames.
    frames = {}
    sample_ids = {}
    for camera in CAMERA_KEYS:
        accepted = sorted((Path(session_dir) / "intrinsics" / camera / "accepted").glob("*_raw.png"))
        if not accepted:
            continue
        image = cv2.imread(str(accepted[-1]))
        if image is not None:
            frames[camera] = image
            sample_ids[camera] = accepted[-1].stem.replace("_raw", "")
    return frames, {
        "source": "independent_intrinsic_samples",
        "sample_ids": sample_ids,
        "synchronized": False,
        "warning": "Frames are from independent sample times.",
    }


def _latest_snapshot_dir() -> Path | None:
    root = PROJECT_ROOT / "projects" / "calibration_snapshots"
    if not root.exists():
        return None
    snapshots = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.yaml").exists()
    ]
    return sorted(snapshots, key=lambda path: path.name)[-1] if snapshots else None


def _latest_front_priority_candidate_run() -> Path | None:
    root = PROJECT_ROOT / "projects" / "seam_candidates"
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.glob("*/*")
        if (path / "left_front" / "candidate.yaml").exists()
        and (path / "front_right" / "candidate.yaml").exists()
    ]
    return sorted(candidates, key=lambda path: str(path))[-1] if candidates else None


def _pair_candidates_from_equirectangular_metadata(
    metadata: dict[str, Any],
    height: int,
) -> list[LayoutPairCandidate]:
    overlaps = metadata.get("suggested_overlaps", {})
    mapping = {
        "left_front": ("front_left", "left"),
        "front_right": ("front_right", "right"),
    }
    output: list[LayoutPairCandidate] = []
    for pair_id, (side_camera, side_position) in mapping.items():
        item = overlaps.get(pair_id, {})
        seam_x = item.get("seam_x")
        if seam_x is None:
            continue
        points = [(int(seam_x), y) for y in range(height)]
        output.append(
            LayoutPairCandidate(
                pair_id=pair_id,
                side_camera=side_camera,
                side_position=side_position,
                boundary_type="projection_overlap_midline",
                seam_points=points,
            )
        )
    return output


def _copy_layout_report_name(report_path: Path) -> None:
    data = load_yaml(report_path)
    save_yaml(report_path.parent / "projection_layout_sweep_report.yaml", data)


def _intrinsics_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for camera, item in report.get("cameras", {}).items():
        for reason in item.get("warning_reasons", []):
            warnings.append(f"{camera}: {reason}")
    return warnings


def _combined_metrics(projection_result: Any, fixed_metrics: dict[str, Any]) -> dict[str, float]:
    if projection_result is None or not fixed_metrics:
        return {}
    image = next(iter(projection_result.warped_images.values()))
    metrics = projection_metrics(image, projection_result.valid_masks)
    metrics.update(
        {
            "side_visible_ratio": float(fixed_metrics.get("side_visible_ratio", 0.0)),
            "front_preserved_ratio": float(fixed_metrics.get("front_preserved_ratio", 0.0)),
        }
    )
    return metrics
