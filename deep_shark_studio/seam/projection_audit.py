"""Read-only projection-layer audit for seam candidate runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deep_shark_studio.config import CONFIG_DIR, file_revision, load_config
from deep_shark_studio.topology import active_stitch_profile, selected_topology_name


def write_projection_audit(
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    formal_calibration_path: str | Path | None = None,
) -> Path:
    formal_path = Path(formal_calibration_path or CONFIG_DIR / "calibration.yaml")
    before = file_revision(formal_path)
    config = config if config is not None else load_config("calibration.yaml")
    profile = active_stitch_profile(config)
    topology_name = selected_topology_name(config)
    active_cameras = [str(camera) for camera in profile.get("active_cameras", [])]
    intrinsics = config.get("camera_intrinsics", {})
    complete = {
        camera: _intrinsics_complete(intrinsics.get(camera, {}))
        for camera in active_cameras
    }
    lines = [
        "# Projection Audit",
        "",
        f"- topology: `{topology_name}`",
        f"- canvas: `{profile.get('canvas', {}).get('width')}x{profile.get('canvas', {}).get('height')}`",
        f"- active_cameras: `{', '.join(active_cameras)}`",
        "- runtime warp: `SurroundStitcher.warp()` uses `cv2.warpPerspective()` after optional `undistort_image()`.",
        "- formal undistort path: `deep_shark_studio/calibration.py::undistort_image()` calls `cv2.undistort()`.",
        "- live GUI default: `deep_shark_studio/gui/main_window.py` reads `performance_config.use_intrinsics`, defaulting to `False` when absent.",
        "- B-2 candidate path: `deep_shark_studio/calibration_candidate.py` contains report-only `opencv_fisheye` and `equirectangular_rotation_only` remap logic with `map_x/map_y/valid_mask` artifacts.",
        "- distortion correction belongs in the warp/projection layer and should not be stuffed into seam cost.",
        "",
        "## Intrinsics Completeness",
        "",
    ]
    for camera in active_cameras:
        camera_intrinsics = intrinsics.get(camera, {})
        model = str(camera_intrinsics.get("model") or camera_intrinsics.get("distortion_model") or "formal_pinhole_opencv")
        lines.append(
            f"- `{camera}`: complete={complete[camera]}, model_hint=`{model}`, "
            f"has_camera_matrix={bool(camera_intrinsics.get('camera_matrix'))}, "
            f"has_distortion={bool(camera_intrinsics.get('distortion_coefficients') or camera_intrinsics.get('distortion'))}"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- The formal runtime is still perspective/homography based; it is not a full fisheye-to-equirectangular projection pipeline.",
            "- `cv2.undistort()` is the pinhole/OpenCV distortion API. It is not equivalent to `cv2.fisheye` calibration/remap for strong fisheye lenses.",
            "- B-2 candidate code is useful as a future projection reference, but this V2 seam candidate does not apply B-2 maps to formal runtime.",
            "- The seam candidate generator consumes already warped images, so it can continue to work if the projection layer later changes its remap implementation.",
            "- A future projection upgrade would likely touch `calibration.py`, `calibration_candidate.py`, `stitcher.py`, and config-loading/report code, not the GUI state machine.",
        ]
    )
    path = Path(output_dir) / "projection_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = file_revision(formal_path)
    if after != before:
        raise RuntimeError("Formal calibration.yaml changed during projection audit.")
    return path


def _intrinsics_complete(data: dict[str, Any]) -> bool:
    matrix = data.get("camera_matrix")
    distortion = data.get("distortion_coefficients") or data.get("distortion")
    return bool(matrix and distortion)
