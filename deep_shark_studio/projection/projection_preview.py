"""Comparison previews for projection candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.seam.layout_preview import (
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
)
from deep_shark_studio.stitcher import save_image


def render_fixed_layout(
    warped_images: dict[str, np.ndarray],
    pair_candidates: list[LayoutPairCandidate],
    layout_params: LayoutPreviewParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = FrontPriorityLayoutPreviewRenderer().render(
        warped_images,
        pair_candidates,
        layout_params,
    )
    return result.image, result.metrics | {"crop_x0": result.crop[0], "crop_x1": result.crop[1]}


def write_projection_comparison(
    output_dir: str | Path,
    perspective_baseline: np.ndarray,
    projection_fixed: np.ndarray,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    files["perspective_baseline"] = _save(root, "perspective_baseline.png", perspective_baseline)
    files["projection_fixed_layout"] = _save(root, "projection_fixed_layout.png", projection_fixed)
    side_by_side = _side_by_side(perspective_baseline, projection_fixed)
    files["perspective_vs_projection_side_by_side"] = _save(
        root,
        "perspective_vs_projection_side_by_side.png",
        side_by_side,
    )
    contact = _contact_sheet(
        [
            ("perspective baseline", perspective_baseline),
            ("projection fixed layout", projection_fixed),
        ]
    )
    files["perspective_vs_projection_contact_sheet"] = _save(
        root,
        "perspective_vs_projection_contact_sheet.png",
        contact,
    )
    files["top_bottom_detail_compare"] = _save(
        root,
        "top_bottom_detail_compare.png",
        _detail_compare(perspective_baseline, projection_fixed, band="top_bottom"),
    )
    files["center_detail_compare"] = _save(
        root,
        "center_detail_compare.png",
        _detail_compare(perspective_baseline, projection_fixed, band="center"),
    )
    return files


def projection_metrics(image: np.ndarray, valid_masks: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    black = np.all(image == 0, axis=2)
    area = max(1, image.shape[0] * image.shape[1])
    valid_pixel_ratio = 1.0 - float(np.count_nonzero(black) / area)
    top_bottom = np.zeros(image.shape[:2], dtype=bool)
    band = max(1, image.shape[0] // 8)
    top_bottom[:band] = True
    top_bottom[-band:] = True
    top_bottom_valid_ratio = 1.0 - float(np.count_nonzero(black & top_bottom) / max(1, np.count_nonzero(top_bottom)))
    duplicate_proxy = 0.0
    if valid_masks:
        masks = [mask for mask in valid_masks.values() if mask.shape == image.shape[:2]]
        if len(masks) >= 2:
            overlap_count = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask in masks:
                overlap_count += mask.astype(np.uint8)
            duplicate_proxy = float(np.count_nonzero(overlap_count > 1) / area)
    return {
        "black_pixel_ratio": float(np.count_nonzero(black) / area),
        "valid_pixel_ratio": valid_pixel_ratio,
        "top_bottom_valid_ratio": top_bottom_valid_ratio,
        "duplicate_risk_proxy": duplicate_proxy,
    }


def _save(root: Path, name: str, image: np.ndarray) -> str:
    path = root / name
    save_image(path, image)
    return path.relative_to(root).as_posix()


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_resized, right_resized = _same_height(left, right)
    return np.hstack([left_resized, right_resized])


def _same_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(left.shape[0], right.shape[0])
    def resize(image: np.ndarray) -> np.ndarray:
        width = int(round(image.shape[1] * height / image.shape[0]))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resize(left), resize(right)


def _contact_sheet(items: list[tuple[str, np.ndarray]]) -> np.ndarray:
    thumb_h = 260
    label_h = 34
    thumbs = []
    for label, image in items:
        width = int(round(image.shape[1] * thumb_h / image.shape[0]))
        thumb = cv2.resize(image, (width, thumb_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((thumb_h + label_h, width, 3), dtype=np.uint8)
        canvas[:thumb_h] = thumb
        cv2.putText(canvas, label, (8, thumb_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    return np.hstack(thumbs)


def _detail_compare(left: np.ndarray, right: np.ndarray, band: str) -> np.ndarray:
    left_resized, right_resized = _same_height(left, right)
    height = left_resized.shape[0]
    if band == "center":
        y0 = max(0, height // 2 - height // 8)
        y1 = min(height, height // 2 + height // 8)
        return np.hstack([left_resized[y0:y1], right_resized[y0:y1]])
    strip = max(1, height // 6)
    left_detail = np.vstack([left_resized[:strip], left_resized[-strip:]])
    right_detail = np.vstack([right_resized[:strip], right_resized[-strip:]])
    return np.hstack([left_detail, right_detail])
