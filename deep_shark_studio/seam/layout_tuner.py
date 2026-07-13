"""Report-only front-priority layout tuner helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.config import CONFIG_DIR, PROJECT_ROOT, file_revision, save_yaml
from deep_shark_studio.stitching.camera_layout_adjust import (
    AdjustedWarpResult,
    CameraAdjustParams,
    apply_post_warp_camera_adjustments,
    identity_camera_adjust,
)
from deep_shark_studio.stitcher import save_image

from .layout_preview import (
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
    LayoutPreviewResult,
)
from .vertical_safety import (
    VerticalSafetyParams,
    VerticalSafetyRenderer,
    VerticalSafetyResult,
)


@dataclass(frozen=True)
class LayoutTunerParams:
    side_shift_px: int = 40
    output_width_px: int = 1800
    side_visible_fraction: float = 0.30
    feather_width_px: int = 24
    output_height_px: int = 700
    vertical_safe_ratio: float = 1.00
    side_vertical_fade_px: int = 0

    def to_layout_params(self) -> LayoutPreviewParams:
        return LayoutPreviewParams(
            side_shift_px=int(self.side_shift_px),
            output_width_px=int(self.output_width_px),
            side_visible_fraction=float(self.side_visible_fraction),
            feather_width_px=int(self.feather_width_px),
        )

    def to_vertical_params(self) -> VerticalSafetyParams:
        return VerticalSafetyParams(
            output_height_px=int(self.output_height_px),
            vertical_safe_ratio=float(self.vertical_safe_ratio),
            side_vertical_fade_px=int(self.side_vertical_fade_px),
        )

    def vertical_safety_enabled(self, canvas_height: int | None = None) -> bool:
        height_changed = (
            canvas_height is not None
            and int(self.output_height_px) < int(canvas_height)
        )
        return (
            height_changed
            or float(self.vertical_safe_ratio) < 0.999
            or int(self.side_vertical_fade_px) > 0
        )

    @property
    def layout_id(self) -> str:
        base = self.to_layout_params().layout_id
        if not self.vertical_safety_enabled():
            return base
        return f"{base}_{self.to_vertical_params().layout_id}"

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PairLayoutParams:
    side_shift_px: int = 40
    side_visible_fraction: float = 0.30
    feather_width_px: int = 24

    def to_layout_params(self, output_width_px: int) -> LayoutPreviewParams:
        return LayoutPreviewParams(
            side_shift_px=int(self.side_shift_px),
            output_width_px=int(output_width_px),
            side_visible_fraction=float(self.side_visible_fraction),
            feather_width_px=int(self.feather_width_px),
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LayoutPreviewParamsV2:
    left_pair: PairLayoutParams = field(default_factory=PairLayoutParams)
    right_pair: PairLayoutParams = field(default_factory=PairLayoutParams)
    camera_adjust: dict[str, CameraAdjustParams] | None = None
    output_width_px: int = 1800
    output_height_px: int = 700
    vertical_safe_ratio: float = 1.00
    side_vertical_fade_px: int = 0
    vertical_safety_enabled_override: bool | None = None

    def __post_init__(self) -> None:
        if self.camera_adjust is None:
            object.__setattr__(self, "camera_adjust", identity_camera_adjust())
        if (
            self.vertical_safety_enabled_override is not None
            and not isinstance(self.vertical_safety_enabled_override, bool)
        ):
            raise ValueError("vertical_safety_enabled_override must be a boolean or None.")

    @property
    def layout_id(self) -> str:
        left = self.left_pair
        right = self.right_pair
        return (
            f"L{left.side_shift_px:03d}_{int(round(left.side_visible_fraction * 100)):03d}_{left.feather_width_px:02d}"
            f"_R{right.side_shift_px:03d}_{int(round(right.side_visible_fraction * 100)):03d}_{right.feather_width_px:02d}"
            f"_crop_{int(self.output_width_px)}_h{int(self.output_height_px)}"
        )

    def to_layout_params(self) -> LayoutPreviewParams:
        return LayoutPreviewParams(
            side_shift_px=int(self.left_pair.side_shift_px),
            output_width_px=int(self.output_width_px),
            side_visible_fraction=float(self.left_pair.side_visible_fraction),
            feather_width_px=int(self.left_pair.feather_width_px),
        )

    def to_vertical_params(self) -> VerticalSafetyParams:
        return VerticalSafetyParams(
            output_height_px=int(self.output_height_px),
            vertical_safe_ratio=float(self.vertical_safe_ratio),
            side_vertical_fade_px=int(self.side_vertical_fade_px),
        )

    def vertical_safety_enabled(self, canvas_height: int | None = None) -> bool:
        if self.vertical_safety_enabled_override is not None:
            return self.vertical_safety_enabled_override
        height_changed = (
            canvas_height is not None
            and int(self.output_height_px) < int(canvas_height)
        )
        return (
            height_changed
            or float(self.vertical_safe_ratio) < 0.999
            or int(self.side_vertical_fade_px) > 0
        )

    def pair_layout_params(self) -> dict[str, LayoutPreviewParams]:
        return {
            "left_front": self.left_pair.to_layout_params(self.output_width_px),
            "front_right": self.right_pair.to_layout_params(self.output_width_px),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_pair": self.left_pair.to_dict(),
            "right_pair": self.right_pair.to_dict(),
            "camera_adjust": {
                camera: params.to_dict()
                for camera, params in (self.camera_adjust or {}).items()
            },
            "output_width_px": int(self.output_width_px),
            "output_height_px": int(self.output_height_px),
            "vertical_safe_ratio": float(self.vertical_safe_ratio),
            "side_vertical_fade_px": int(self.side_vertical_fade_px),
        }


@dataclass(frozen=True)
class LayoutTunerPreviewResult:
    image: np.ndarray
    side_visible_mask: np.ndarray
    front_preserved_mask: np.ndarray
    side_suppressed_mask: np.ndarray
    metrics: dict[str, Any]
    crop_x: tuple[int, int]
    crop_y: tuple[int, int]
    layout_id: str
    vertical_safety_enabled: bool
    adjusted_warped_images: dict[str, np.ndarray] | None = None
    adjusted_valid_masks: dict[str, np.ndarray] | None = None
    camera_adjust_metadata: dict[str, Any] | None = None


LAYOUT_TUNER_PRESETS: dict[str, LayoutTunerParams] = {
    "Balanced": LayoutTunerParams(),
    "Conservative": LayoutTunerParams(
        side_shift_px=40,
        output_width_px=1700,
        side_visible_fraction=0.25,
        feather_width_px=16,
        output_height_px=640,
        vertical_safe_ratio=0.86,
        side_vertical_fade_px=48,
    ),
    "Wide": LayoutTunerParams(
        side_shift_px=20,
        output_width_px=2000,
        side_visible_fraction=0.35,
        feather_width_px=24,
        output_height_px=700,
        vertical_safe_ratio=1.00,
        side_vertical_fade_px=0,
    ),
    "Hard Seam": LayoutTunerParams(
        side_shift_px=40,
        output_width_px=1800,
        side_visible_fraction=0.30,
        feather_width_px=0,
        output_height_px=700,
        vertical_safe_ratio=1.00,
        side_vertical_fade_px=0,
    ),
}


LAYOUT_TUNER_PRESETS_V2: dict[str, LayoutPreviewParamsV2] = {
    "Balanced": LayoutPreviewParamsV2(),
    "Conservative": LayoutPreviewParamsV2(
        left_pair=PairLayoutParams(40, 0.25, 16),
        right_pair=PairLayoutParams(40, 0.25, 16),
        output_width_px=1700,
        output_height_px=640,
        vertical_safe_ratio=0.86,
        side_vertical_fade_px=48,
    ),
    "Wide": LayoutPreviewParamsV2(
        left_pair=PairLayoutParams(20, 0.35, 24),
        right_pair=PairLayoutParams(20, 0.35, 24),
        output_width_px=2000,
        output_height_px=700,
    ),
    "Hard Seam": LayoutPreviewParamsV2(
        left_pair=PairLayoutParams(40, 0.30, 0),
        right_pair=PairLayoutParams(40, 0.30, 0),
        output_width_px=1800,
        output_height_px=700,
    ),
    "Front Smaller": LayoutPreviewParamsV2(
        left_pair=PairLayoutParams(40, 0.30, 24),
        right_pair=PairLayoutParams(40, 0.30, 24),
        camera_adjust={
            "front_left": CameraAdjustParams(),
            "front": CameraAdjustParams(scale=0.94),
            "front_right": CameraAdjustParams(),
        },
        output_width_px=1800,
        output_height_px=700,
    ),
}


def layout_tuner_params_v1_to_v2(
    params: LayoutTunerParams | LayoutPreviewParams,
) -> LayoutPreviewParamsV2:
    if isinstance(params, LayoutPreviewParams):
        pair = PairLayoutParams(
            side_shift_px=int(params.side_shift_px),
            side_visible_fraction=float(params.side_visible_fraction),
            feather_width_px=int(params.feather_width_px),
        )
        return LayoutPreviewParamsV2(
            left_pair=pair,
            right_pair=pair,
            output_width_px=int(params.output_width_px),
            output_height_px=700,
        )
    pair = PairLayoutParams(
        side_shift_px=int(params.side_shift_px),
        side_visible_fraction=float(params.side_visible_fraction),
        feather_width_px=int(params.feather_width_px),
    )
    return LayoutPreviewParamsV2(
        left_pair=pair,
        right_pair=pair,
        output_width_px=int(params.output_width_px),
        output_height_px=int(params.output_height_px),
        vertical_safe_ratio=float(params.vertical_safe_ratio),
        side_vertical_fade_px=int(params.side_vertical_fade_px),
    )


def normalize_layout_tuner_params(
    params: LayoutPreviewParamsV2 | LayoutTunerParams | LayoutPreviewParams | None,
) -> LayoutPreviewParamsV2:
    if params is None:
        return LayoutPreviewParamsV2()
    if isinstance(params, LayoutPreviewParamsV2):
        return params
    return layout_tuner_params_v1_to_v2(params)


def layout_tuner_pair_candidates(
    stitch_profile: dict[str, Any],
) -> list[LayoutPairCandidate]:
    """Build front-priority side candidates from current profile seams."""
    stitch_points = stitch_profile.get("stitch_points", {})
    pairs: list[LayoutPairCandidate] = []
    for overlap in stitch_profile.get("overlaps", []):
        cameras = list(overlap.get("cameras", []))
        seam_name = str(overlap.get("seam", ""))
        if "front" not in cameras or not seam_name:
            continue
        side_camera = next((camera for camera in cameras if camera != "front"), "")
        if side_camera == "front_left":
            side_position = "left"
        elif side_camera == "front_right":
            side_position = "right"
        else:
            continue
        height = _int_or_default(
            stitch_profile.get("canvas", {}).get("height"),
            700,
        )
        x_range = overlap.get("x_range", [0, 0])
        seam_x = _overlap_midpoint(x_range)
        seam_points = _valid_seam_points(
            stitch_points.get(seam_name, []),
            fallback_x=seam_x,
            fallback_height=height,
        )
        if not seam_points:
            seam_points = [(seam_x, 0), (seam_x, height)]
        pairs.append(
            LayoutPairCandidate(
                pair_id=str(overlap.get("name") or seam_name),
                side_camera=side_camera,
                side_position=side_position,
                boundary_type="layout_tuner_profile_seam",
                seam_points=seam_points,
            )
        )
    return pairs


def render_front_priority_layout_preview(
    warped_images: dict[str, np.ndarray],
    pair_candidates: list[LayoutPairCandidate],
    params: LayoutPreviewParamsV2 | LayoutTunerParams | LayoutPreviewParams | None = None,
    vertical_params: VerticalSafetyParams | None = None,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> LayoutTunerPreviewResult:
    """Render one tuner preview from already-warped images."""
    tuner_params = normalize_layout_tuner_params(params)
    adjusted = apply_post_warp_camera_adjustments(
        warped_images,
        valid_masks=valid_masks,
        camera_adjust=tuner_params.camera_adjust or identity_camera_adjust(),
    )
    canvas_height = _front_canvas_height(adjusted.warped_images)
    layout_params = tuner_params.to_layout_params()
    pair_params = tuner_params.pair_layout_params()
    safety_params = vertical_params or tuner_params.to_vertical_params()
    use_vertical = (
        vertical_params is not None
        or tuner_params.vertical_safety_enabled(canvas_height)
    )
    if use_vertical:
        result = VerticalSafetyRenderer().render(
            adjusted.warped_images,
            pair_candidates,
            layout_params,
            safety_params,
            pair_params=pair_params,
        )
        return _from_vertical_result(
            result,
            layout_params,
            adjusted,
            tuner_params.layout_id,
        )
    result = FrontPriorityLayoutPreviewRenderer().render(
        adjusted.warped_images,
        pair_candidates,
        layout_params,
        pair_params=pair_params,
    )
    return _from_layout_result(result, adjusted, tuner_params.layout_id)


def save_layout_tuner_candidate(
    output_root: str | Path,
    profile_id: str,
    params: LayoutPreviewParamsV2 | LayoutTunerParams | LayoutPreviewParams,
    preview: LayoutTunerPreviewResult,
    source: dict[str, Any] | None = None,
    projection: dict[str, Any] | None = None,
    formal_calibration_path: str | Path | None = None,
) -> Path:
    """Save one human-readable layout tuner candidate without config writes."""
    tuner_params = normalize_layout_tuner_params(params)
    calibration_path = Path(formal_calibration_path or CONFIG_DIR / "calibration.yaml")
    before = file_revision(calibration_path)
    directory = _create_candidate_directory(Path(output_root))
    masks_dir = directory / "masks"
    debug_dir = directory / "debug"
    masks_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    preview_path = directory / "preview.png"
    overlay_path = directory / "preview_with_overlay.png"
    save_image(preview_path, preview.image)
    save_image(overlay_path, _overlay_preview(preview.image, preview))
    save_image(masks_dir / "side_visible_mask.png", _mask_image(preview.side_visible_mask))
    save_image(masks_dir / "front_preserved_mask.png", _mask_image(preview.front_preserved_mask))
    save_image(masks_dir / "suppressed_side_mask.png", _mask_image(preview.side_suppressed_mask))
    debug_files = _save_adjusted_debug_images(debug_dir, preview)
    schema_version = 3 if projection else 2
    candidate = {
        "schema_version": schema_version,
        "candidate_type": "front_priority_layout",
        "profile_id": profile_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "source": {
                "mode": "snapshot_preview",
                "calibration_hash": before,
                **(source or {}),
        },
        "camera_adjust": {
            camera: value.to_dict()
            for camera, value in (tuner_params.camera_adjust or {}).items()
        },
        "left_pair": tuner_params.left_pair.to_dict(),
        "right_pair": tuner_params.right_pair.to_dict(),
        "output": {
            "width_px": int(tuner_params.output_width_px),
            "height_px": int(tuner_params.output_height_px),
        },
        "layout": {
            "side_shift_px": int(tuner_params.left_pair.side_shift_px),
            "output_width_px": int(tuner_params.output_width_px),
            "side_visible_fraction": float(tuner_params.left_pair.side_visible_fraction),
            "feather_width_px": int(tuner_params.left_pair.feather_width_px),
            "compatibility_note": "V1 compatibility mirror of left_pair; prefer left_pair/right_pair.",
        },
        "vertical_safety": {
            "enabled": bool(preview.vertical_safety_enabled),
            "vertical_safe_ratio": float(tuner_params.vertical_safe_ratio),
            "side_vertical_fade_px": int(tuner_params.side_vertical_fade_px),
        },
        "status": "candidate",
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
        "metrics": preview.metrics,
        "files": {
            "preview": "preview.png",
            "preview_with_overlay": "preview_with_overlay.png",
            "side_visible_mask": "masks/side_visible_mask.png",
            "front_preserved_mask": "masks/front_preserved_mask.png",
            "suppressed_side_mask": "masks/suppressed_side_mask.png",
            "debug": debug_files,
        },
    }
    if projection:
        candidate["projection"] = dict(projection)
    report = {
        "schema_version": schema_version,
        "candidate_type": "front_priority_layout",
        "layout_id": preview.layout_id,
        "profile_id": profile_id,
        "camera_adjust": candidate["camera_adjust"],
        "left_pair": candidate["left_pair"],
        "right_pair": candidate["right_pair"],
        "output": candidate["output"],
        "metrics": preview.metrics,
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
        "calibration_hash_before": before,
        "calibration_hash_after": before,
        "notes": [
            "Report-only layout tuner candidate.",
            "Not applied to formal runtime/profile.",
            (
                "Projection source is recorded in candidate.yaml."
                if projection
                else "Uses existing perspective/template warped images, not projection candidates."
            ),
        ],
    }
    if projection:
        report["projection"] = dict(projection)
    save_yaml(directory / "candidate.yaml", candidate)
    save_yaml(directory / "report.yaml", report)
    after = file_revision(calibration_path)
    if after != before:
        raise RuntimeError("Formal calibration.yaml changed during layout tuner save.")
    report["calibration_hash_after"] = after
    save_yaml(directory / "report.yaml", report)
    return directory


def default_layout_candidate_root() -> Path:
    return PROJECT_ROOT / "projects" / "layout_candidates"


def _valid_seam_points(
    raw_points: Any,
    fallback_x: int,
    fallback_height: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    if not isinstance(raw_points, (list, tuple)):
        return points
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = _float_or_none(point[0])
        y = _float_or_none(point[1])
        if x is None or y is None:
            continue
        points.append((int(round(x)), int(round(y))))
    if len(points) == 1:
        return [(points[0][0], 0), (points[0][0], fallback_height)]
    return points


def _overlap_midpoint(x_range: Any) -> int:
    if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
        start = _float_or_none(x_range[0])
        end = _float_or_none(x_range[1])
        if start is not None and end is not None:
            return int(round((start + end) / 2.0))
    return 0


def _int_or_default(value: Any, default: int) -> int:
    numeric = _float_or_none(value)
    return int(round(numeric)) if numeric is not None else int(default)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _from_layout_result(
    result: LayoutPreviewResult,
    adjusted: AdjustedWarpResult | None = None,
    layout_id: str | None = None,
) -> LayoutTunerPreviewResult:
    return LayoutTunerPreviewResult(
        image=result.image,
        side_visible_mask=result.side_visible_mask,
        front_preserved_mask=result.front_preserved_mask,
        side_suppressed_mask=result.side_suppressed_mask,
        metrics=dict(result.metrics),
        crop_x=result.crop,
        crop_y=(0, result.image.shape[0]),
        layout_id=layout_id or result.layout_id,
        vertical_safety_enabled=False,
        adjusted_warped_images=adjusted.warped_images if adjusted else None,
        adjusted_valid_masks=adjusted.valid_masks if adjusted else None,
        camera_adjust_metadata=adjusted.metadata if adjusted else None,
    )


def _from_vertical_result(
    result: VerticalSafetyResult,
    layout_params: LayoutPreviewParams,
    adjusted: AdjustedWarpResult | None = None,
    layout_id: str | None = None,
) -> LayoutTunerPreviewResult:
    metrics = dict(result.metrics)
    metrics["base_layout_id"] = layout_params.layout_id
    return LayoutTunerPreviewResult(
        image=result.image,
        side_visible_mask=result.side_visible_after,
        front_preserved_mask=result.front_preserved_mask,
        side_suppressed_mask=result.side_suppressed_top_bottom,
        metrics=metrics,
        crop_x=(0, result.image.shape[1]),
        crop_y=result.crop_y,
        layout_id=layout_id or f"{layout_params.layout_id}_{result.layout_id}",
        vertical_safety_enabled=True,
        adjusted_warped_images=adjusted.warped_images if adjusted else None,
        adjusted_valid_masks=adjusted.valid_masks if adjusted else None,
        camera_adjust_metadata=adjusted.metadata if adjusted else None,
    )


def _front_canvas_height(warped_images: dict[str, np.ndarray]) -> int | None:
    front = warped_images.get("front")
    return int(front.shape[0]) if front is not None else None


def _create_candidate_directory(output_root: Path) -> Path:
    created = datetime.now().astimezone()
    session_id = created.strftime("%Y%m%d_%H%M%S")
    run_id = created.strftime("run_%H%M%S_%f")[:-3]
    directory = output_root / session_id / run_id
    suffix = 1
    while directory.exists():
        directory = output_root / session_id / f"{run_id}_{suffix:02d}"
        suffix += 1
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _overlay_preview(image: np.ndarray, preview: LayoutTunerPreviewResult) -> np.ndarray:
    overlay = image.copy()
    if preview.side_visible_mask.shape == image.shape[:2]:
        color = np.zeros_like(image)
        color[:, :, 1] = 190
        overlay[preview.side_visible_mask] = cv2.addWeighted(
            image[preview.side_visible_mask],
            0.55,
            color[preview.side_visible_mask],
            0.45,
            0,
        )
    if preview.side_suppressed_mask.shape == image.shape[:2]:
        color = np.zeros_like(image)
        color[:, :, 2] = 220
        overlay[preview.side_suppressed_mask] = cv2.addWeighted(
            overlay[preview.side_suppressed_mask],
            0.55,
            color[preview.side_suppressed_mask],
            0.45,
            0,
        )
    return overlay


def _mask_image(mask: np.ndarray) -> np.ndarray:
    return np.dstack([mask.astype(np.uint8) * 255] * 3)


def _save_adjusted_debug_images(
    debug_dir: Path,
    preview: LayoutTunerPreviewResult,
) -> dict[str, str]:
    files: dict[str, str] = {}
    adjusted = preview.adjusted_warped_images or {}
    masks = preview.adjusted_valid_masks or {}
    for camera in ("front_left", "front", "front_right"):
        image = adjusted.get(camera)
        if image is not None:
            name = f"adjusted_{camera}.png"
            save_image(debug_dir / name, image)
            files[f"adjusted_{camera}"] = f"debug/{name}"
        mask = masks.get(camera)
        if mask is not None:
            name = f"adjusted_valid_{camera}.png"
            save_image(debug_dir / name, _mask_image(mask))
            files[f"adjusted_valid_{camera}"] = f"debug/{name}"
    overlay = _camera_adjust_overlay(preview)
    if overlay is not None:
        name = "camera_adjust_overlay.png"
        save_image(debug_dir / name, overlay)
        files["camera_adjust_overlay"] = f"debug/{name}"
    return files


def _camera_adjust_overlay(preview: LayoutTunerPreviewResult) -> np.ndarray | None:
    adjusted = preview.adjusted_warped_images or {}
    masks = preview.adjusted_valid_masks or {}
    front = adjusted.get("front")
    if front is None:
        return None
    overlay = front.copy()
    colors = {
        "front_left": (0, 0, 255),
        "front": (0, 255, 0),
        "front_right": (255, 0, 0),
    }
    labels = {
        "front_left": "front_left",
        "front": "front",
        "front_right": "front_right",
    }
    for camera, color in colors.items():
        mask = masks.get(camera)
        if mask is None or not np.any(mask):
            continue
        tint = np.zeros_like(overlay)
        tint[:, :] = color
        overlay[mask] = cv2.addWeighted(overlay[mask], 0.72, tint[mask], 0.28, 0)
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            labels[camera],
            (max(0, x0 + 4), max(18, y0 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay
