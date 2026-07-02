"""Consolidate report-only fisheye intrinsics candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deep_shark_studio.config import PROJECT_ROOT, load_config, load_yaml, save_yaml


CAMERA_KEYS = ("front_left", "front", "front_right")


@dataclass(frozen=True)
class IntrinsicsCandidateResult:
    report: dict[str, Any]
    models: dict[str, "FisheyeCameraModel"]
    source_candidate_dir: Path | None


@dataclass(frozen=True)
class FisheyeCameraModel:
    camera: str
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    source: str
    rms_px: float | None
    accepted_samples: int | None
    warnings: tuple[str, ...] = ()


def find_latest_calibration_candidate(session_dir: str | Path) -> Path | None:
    session_path = Path(session_dir)
    session_id = session_path.name
    root = PROJECT_ROOT / "projects" / "calibration_candidates" / session_id
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "candidate.yaml").exists()
    ]
    return sorted(candidates, key=lambda path: path.name)[-1] if candidates else None


def consolidate_intrinsics_candidates(
    session_dir: str | Path,
    output_dir: str | Path,
    profile_id: str = "triple_front_panorama",
    calibration_candidate_dir: str | Path | None = None,
) -> IntrinsicsCandidateResult:
    """Build human-readable K/D reports without touching formal calibration."""
    formal_config = load_config("calibration.yaml")
    candidate_dir = (
        Path(calibration_candidate_dir)
        if calibration_candidate_dir is not None
        else find_latest_calibration_candidate(session_dir)
    )
    candidate_data: dict[str, Any] = {}
    if candidate_dir is not None:
        candidate_data = load_yaml(candidate_dir / "candidate.yaml")
    report = summarize_intrinsics_candidates(
        formal_config,
        candidate_data,
        Path(session_dir),
        profile_id,
        candidate_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_yaml(output_path / "intrinsics_candidates_report.yaml", report)
    _write_markdown(output_path / "intrinsics_candidates_report.md", report)
    models: dict[str, FisheyeCameraModel] = {}
    for camera, item in report["cameras"].items():
        if item["status"] != "available" or item["model"] != "opencv_fisheye":
            continue
        models[camera] = FisheyeCameraModel(
            camera=camera,
            camera_matrix=np.asarray(item["K"], dtype=np.float64),
            distortion=np.asarray(item["D"], dtype=np.float64).reshape(4, 1),
            image_size=tuple(int(v) for v in item["image_size"]),
            source=str(item["source"]),
            rms_px=item.get("reprojection_error"),
            accepted_samples=item.get("accepted_samples"),
            warnings=tuple(item.get("warning_reasons", [])),
        )
    return IntrinsicsCandidateResult(
        report=report,
        models=models,
        source_candidate_dir=candidate_dir,
    )


def summarize_intrinsics_candidates(
    formal_config: dict[str, Any],
    candidate_data: dict[str, Any],
    session_dir: Path,
    profile_id: str,
    candidate_dir: Path | None,
) -> dict[str, Any]:
    formal_intrinsics = formal_config.get("camera_intrinsics", {})
    session_counts = _session_counts(session_dir)
    cameras: dict[str, Any] = {}
    candidate_intrinsics = candidate_data.get("intrinsics", {})
    for camera in CAMERA_KEYS:
        warnings: list[str] = []
        formal = formal_intrinsics.get(camera, {})
        formal_has_k = bool(formal.get("camera_matrix"))
        formal_has_d = bool(formal.get("distortion") or formal.get("distortion_coefficients"))
        candidate = candidate_intrinsics.get(camera, {})
        if candidate.get("status") == "success" and candidate.get("camera_matrix") and candidate.get("distortion_coefficients"):
            status = "available"
            model = str(candidate.get("model") or "opencv_fisheye")
            source = (
                f"calibration_candidate:{candidate_dir}"
                if candidate_dir is not None
                else "calibration_candidate"
            )
            image_size = list(candidate.get("resolution", candidate_data.get("resolution", [0, 0])))
            k = candidate["camera_matrix"]
            d = candidate["distortion_coefficients"]
            reprojection = candidate.get("rms_px")
            accepted = candidate.get("accepted_input_count")
            if not (formal_has_k and formal_has_d):
                warnings.append("formal calibration.yaml lacks complete K/D for this camera")
            if model != "opencv_fisheye":
                warnings.append("candidate model is not opencv_fisheye")
        elif formal_has_k and formal_has_d:
            status = "ambiguous"
            model = "pinhole_opencv"
            source = "configs/calibration.yaml"
            image_size = list(formal.get("image_size", formal_config.get("source_reference_size", [0, 0])))
            k = formal["camera_matrix"]
            d = formal.get("distortion") or formal.get("distortion_coefficients")
            reprojection = formal.get("rms") or formal.get("mean_reprojection_error")
            accepted = None
            warnings.append("formal K/D exists but is pinhole-style, not a fisheye candidate")
        else:
            status = "missing"
            model = "unknown"
            source = "none"
            image_size = list(candidate_data.get("resolution", [0, 0]))
            k = None
            d = None
            reprojection = None
            accepted = session_counts.get(camera, {}).get("accepted_raw_count")
            warnings.append("no usable K/D candidate found")

        cameras[camera] = {
            "status": status,
            "source": source,
            "model": model,
            "image_size": image_size,
            "K": k,
            "D": d,
            "reprojection_error": reprojection,
            "accepted_samples": accepted,
            "session_accepted_raw": session_counts.get(camera, {}).get("accepted_raw_count", 0),
            "session_rejected_raw": session_counts.get(camera, {}).get("rejected_raw_count", 0),
            "formal_has_k": formal_has_k,
            "formal_has_d": formal_has_d,
            "warning_reasons": warnings,
        }
    all_available = all(
        item["status"] == "available" and item["model"] == "opencv_fisheye"
        for item in cameras.values()
    )
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "session_dir": str(session_dir),
        "source_calibration_candidate": str(candidate_dir) if candidate_dir else None,
        "board": {
            "user_total_squares": [9, 12],
            "config_or_session_inner_corners": [11, 8],
            "square_size_mm": 25.0,
            "note": "9x12 total squares are orientation-equivalent to 12x9; OpenCV uses 11x8 inner corners.",
        },
        "all_required_intrinsics_available": all_available,
        "cameras": cameras,
    }


def _session_counts(session_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for camera in CAMERA_KEYS:
        accepted = list((session_dir / "intrinsics" / camera / "accepted").glob("*_raw.png"))
        rejected = list((session_dir / "intrinsics" / camera / "rejected").glob("*_raw.png"))
        counts[camera] = {
            "accepted_raw_count": len(accepted),
            "rejected_raw_count": len(rejected),
        }
    return counts


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Intrinsics Candidates Report",
        "",
        f"- profile_id: `{report['profile_id']}`",
        f"- all_required_intrinsics_available: `{report['all_required_intrinsics_available']}`",
        f"- session_dir: `{report['session_dir']}`",
        f"- source_calibration_candidate: `{report.get('source_calibration_candidate')}`",
        "",
        "## Cameras",
        "",
    ]
    for camera, item in report["cameras"].items():
        lines.extend(
            [
                f"### {camera}",
                "",
                f"- status: `{item['status']}`",
                f"- model: `{item['model']}`",
                f"- source: `{item['source']}`",
                f"- image_size: `{item['image_size']}`",
                f"- reprojection_error: `{item['reprojection_error']}`",
                f"- accepted_samples: `{item['accepted_samples']}`",
                f"- warnings: `{item['warning_reasons']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
