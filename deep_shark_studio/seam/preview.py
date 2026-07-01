"""Preview rendering for static seam candidates."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from deep_shark_studio.stitcher import save_image

from .cost import SeamCostResult
from .dp import SeamPath
from .mask import SeamMaskBuilder


class SeamPreviewRenderer:
    """Render stable PNG diagnostics for a pairwise seam candidate."""

    def render_pair(
        self,
        output_dir: str | Path,
        image_a: np.ndarray,
        image_b: np.ndarray,
        cost_result: SeamCostResult,
        seam_path: SeamPath,
        feather_widths: list[int],
    ) -> dict[str, str]:
        directory = Path(output_dir)
        previews = directory / "previews"
        previews.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}

        heatmap = self.cost_heatmap(image_a.shape[:2], cost_result)
        files["cost_heatmap"] = self._save(previews, "cost_heatmap.png", heatmap)

        overlay = self.seam_overlay(image_a, image_b, seam_path)
        files["seam_overlay"] = self._save(previews, "seam_overlay.png", overlay)

        for width in feather_widths:
            preview = self.blend_preview(
                image_a,
                image_b,
                seam_path,
                cost_result.x_range,
                int(width),
            )
            filename = (
                "hard_seam_preview.png"
                if int(width) == 0
                else f"feather_{int(width)}_preview.png"
            )
            key = "hard_seam_preview" if int(width) == 0 else f"feather_{int(width)}"
            files[key] = self._save(previews, filename, preview)
        return files

    @staticmethod
    def cost_heatmap(
        canvas_shape: tuple[int, int],
        cost_result: SeamCostResult,
    ) -> np.ndarray:
        height, width = canvas_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        roi_cost = cost_result.cost.copy()
        valid = cost_result.valid_mask & np.isfinite(roi_cost) & (roi_cost < 1.0e8)
        normalized = np.zeros(roi_cost.shape, dtype=np.uint8)
        if np.any(valid):
            scale = float(np.percentile(roi_cost[valid], 95))
            scale = max(scale, 1e-6)
            normalized[valid] = np.clip(roi_cost[valid] / scale * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        heatmap[~cost_result.valid_mask] = (0, 0, 80)
        start, end = cost_result.x_range
        canvas[:, start : end + 1] = heatmap
        cv2.rectangle(canvas, (start, 0), (end, height - 1), (0, 255, 255), 2)
        return canvas

    @staticmethod
    def seam_overlay(
        image_a: np.ndarray,
        image_b: np.ndarray,
        seam_path: SeamPath,
    ) -> np.ndarray:
        valid_a = np.any(image_a != 0, axis=2)
        valid_b = np.any(image_b != 0, axis=2)
        base = np.zeros_like(image_a)
        both = valid_a & valid_b
        base[valid_a & ~valid_b] = image_a[valid_a & ~valid_b]
        base[valid_b & ~valid_a] = image_b[valid_b & ~valid_a]
        base[both] = (
            image_a[both].astype(np.float32) * 0.5
            + image_b[both].astype(np.float32) * 0.5
        ).astype(np.uint8)
        points = np.asarray(seam_path.points_canvas, dtype=np.int32).reshape(-1, 1, 2)
        if len(points) > 1:
            cv2.polylines(base, [points], False, (0, 255, 255), 2, cv2.LINE_AA)
        return base

    @staticmethod
    def blend_preview(
        image_a: np.ndarray,
        image_b: np.ndarray,
        seam_path: SeamPath,
        x_range: tuple[int, int],
        feather_width: int,
    ) -> np.ndarray:
        height, width = image_a.shape[:2]
        masks = SeamMaskBuilder().build(
            seam_path,
            (height, width),
            x_range,
            feather_width,
        )
        valid_a = np.any(image_a != 0, axis=2)
        valid_b = np.any(image_b != 0, axis=2)
        both = valid_a & valid_b
        output = np.zeros_like(image_a)
        output[valid_a & ~valid_b] = image_a[valid_a & ~valid_b]
        output[valid_b & ~valid_a] = image_b[valid_b & ~valid_a]
        alpha_a = masks.alpha_a[:, :, None]
        alpha_b = masks.alpha_b[:, :, None]
        blended = (
            image_a.astype(np.float32) * alpha_a
            + image_b.astype(np.float32) * alpha_b
        )
        output[both] = np.clip(blended[both], 0, 255).astype(np.uint8)
        points = np.asarray(seam_path.points_canvas, dtype=np.int32).reshape(-1, 1, 2)
        if len(points) > 1:
            cv2.polylines(output, [points], False, (0, 255, 255), 1, cv2.LINE_AA)
        return output

    @staticmethod
    def _save(directory: Path, filename: str, image: np.ndarray) -> str:
        path = directory / filename
        save_image(path, image)
        return path.relative_to(directory.parent).as_posix()
