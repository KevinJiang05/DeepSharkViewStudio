"""Read-only fisheye projection readiness audit."""

from __future__ import annotations

from pathlib import Path
from deep_shark_studio.calibration import spec_from_config
from deep_shark_studio.config import CONFIG_DIR, file_revision, load_config
from deep_shark_studio.topology import active_stitch_profile, selected_topology_name


def write_projection_readiness_audit(
    output_dir: str | Path,
    session_dir: str | Path | None = None,
    formal_calibration_path: str | Path | None = None,
) -> Path:
    formal_path = Path(formal_calibration_path or CONFIG_DIR / "calibration.yaml")
    before = file_revision(formal_path)
    config = load_config("calibration.yaml")
    profile = active_stitch_profile(config)
    topology = selected_topology_name(config)
    spec = spec_from_config(config)
    intrinsics = config.get("camera_intrinsics", {})
    cameras = [str(camera) for camera in profile.get("active_cameras", [])]
    session_summary = _session_intrinsics_summary(session_dir) if session_dir else {}
    lines = [
        "# Fisheye Projection Readiness Audit",
        "",
        f"- topology: `{topology}`",
        f"- active_cameras: `{', '.join(cameras)}`",
        f"- formal canvas: `{profile.get('canvas', {}).get('width')}x{profile.get('canvas', {}).get('height')}`",
        f"- chessboard total squares from config: `{spec.total_columns}x{spec.total_rows}`",
        f"- chessboard inner corners used by OpenCV: `{spec.inner_columns}x{spec.inner_rows}`",
        f"- square size: `{spec.square_size_mm} mm`",
        "- user-supplied board note: `9x12 squares, 25mm x 25mm`; orientation is equivalent to the config's `12x9` total-square board.",
        "",
        "## Formal Intrinsics",
        "",
    ]
    for camera in cameras:
        data = intrinsics.get(camera, {})
        distortion = data.get("distortion_coefficients") or data.get("distortion")
        model = "formal_pinhole_opencv"
        if data.get("model"):
            model = str(data["model"])
        lines.append(
            f"- `{camera}`: K={bool(data.get('camera_matrix'))}, "
            f"D={bool(distortion)}, model_hint=`{model}`"
        )
    if session_summary:
        lines.extend(["", "## Session Intrinsic Samples", ""])
        lines.append(f"- session_dir: `{Path(session_dir)}`")
        for camera, summary in session_summary.items():
            lines.append(
                f"- `{camera}`: accepted_raw={summary['accepted_raw_count']}, "
                f"rejected_raw={summary['rejected_raw_count']}"
            )
    lines.extend(
        [
            "",
            "## Projection Layer Findings",
            "",
            "- Formal runtime uses `SurroundStitcher.warp()` with `cv2.warpPerspective()` and optional `undistort_image()` before that warp.",
            "- Formal `undistort_image()` calls `cv2.undistort()`, which is the pinhole/OpenCV distortion path, not an OpenCV fisheye remap.",
            "- GUI live stitching reads `performance_config.use_intrinsics`; current runtime defaults to `False` when that option is absent.",
            "- `calibration_candidate.py` already contains report-only `opencv_fisheye`, `cv2.fisheye.calibrate`, `cv2.fisheye.stereoCalibrate`, and `equirectangular_rotation_only` remap code.",
            "- B-2 candidate artifacts can include `map_x`, `map_y`, and `valid_mask` caches, but those are candidate artifacts, not formal runtime maps.",
            "- Distortion correction and fisheye/equirectangular projection belong in the warp/projection layer.",
            "- Seam/layout layers should consume `warped_images` and preferably explicit `valid_mask`; they should not run ad-hoc undistort inside seam cost.",
            "",
            "## Readiness Assessment",
            "",
            "- Formal K/D is incomplete for the current three-camera runtime if any active camera lacks `camera_matrix` or distortion.",
            "- Strong fisheye cameras should not be represented as solved merely because pinhole `cv2.undistort()` is callable.",
            "- The session intrinsic chessboard data is useful for a separate fisheye candidate solve, but applying it should remain report-only until validated.",
            "- If fisheye/equirectangular remap changes the warped canvas, the layout sweep parameters must be rerun; current `shift/crop/side/fade` values are tied to the current perspective/template warp.",
            "",
            "## Suggested Separate Projection Upgrade Path",
            "",
            "1. Generate a report-only fisheye candidate from the calibration session.",
            "2. Export per-camera remap previews and valid masks without modifying `configs/calibration.yaml`.",
            "3. Compare current perspective baseline vs equirectangular candidate on the same still frames.",
            "4. Rerun seam/layout/vertical-safety sweeps on the new `warped_images`.",
            "5. Only after manual approval, design a runtime projection switch.",
        ]
    )
    path = Path(output_dir) / "projection_readiness_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = file_revision(formal_path)
    if after != before:
        raise RuntimeError("Formal calibration.yaml changed during projection readiness audit.")
    return path


def _session_intrinsics_summary(session_dir: str | Path | None) -> dict[str, dict[str, int]]:
    if session_dir is None:
        return {}
    root = Path(session_dir)
    summary: dict[str, dict[str, int]] = {}
    for camera in ("front_left", "front", "front_right"):
        accepted = list((root / "intrinsics" / camera / "accepted").glob("*_raw.png"))
        rejected = list((root / "intrinsics" / camera / "rejected").glob("*_raw.png"))
        summary[camera] = {
            "accepted_raw_count": len(accepted),
            "rejected_raw_count": len(rejected),
        }
    return summary
