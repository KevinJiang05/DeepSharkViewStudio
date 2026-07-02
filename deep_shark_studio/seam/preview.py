"""Preview rendering for static seam candidates."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from deep_shark_studio.stitcher import save_image

from .cost import SeamCostResult
from .dp import SeamPath
from .mask import SeamMaskBuilder
from .boundary import BoundarySearchResult
from .roles import PairRole


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

    def render_front_priority_pair(
        self,
        output_dir: str | Path,
        warped_images: dict[str, np.ndarray],
        pair_role: PairRole,
        cost_result: SeamCostResult,
        seam_path: SeamPath,
        boundary_result: BoundarySearchResult,
        feather_widths: list[int],
    ) -> dict[str, str]:
        directory = Path(output_dir)
        previews = directory / "previews"
        previews.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        side = warped_images[pair_role.side_camera]
        front = warped_images[pair_role.main_camera]

        files["final_cost"] = self._save(
            previews,
            "final_cost.png",
            self.cost_heatmap(front.shape[:2], cost_result),
        )
        files["front_priority_cost"] = self._save(
            previews,
            "front_priority_cost.png",
            self.component_heatmap(front.shape[:2], cost_result, cost_result.front_priority_cost),
        )
        files["object_proximity_cost"] = self._save(
            previews,
            "object_proximity_cost.png",
            self.component_heatmap(front.shape[:2], cost_result, cost_result.object_proximity_cost),
        )
        files["high_difference_cost"] = self._save(
            previews,
            "high_difference_cost.png",
            self.component_heatmap(front.shape[:2], cost_result, cost_result.high_difference_cost),
        )
        overlay_base = self.seam_overlay(side, front, seam_path)
        files["seam_overlay"] = self._save(previews, "seam_overlay.png", overlay_base)
        files["vertical_boundary_overlay"] = self._save(
            previews,
            "vertical_boundary_overlay.png",
            self.boundary_overlay(front, side, boundary_result),
        )

        hard, side_mask, front_mask = self.front_dominant_preview(
            front,
            side,
            boundary_result,
            pair_role.side_position,
            0,
        )
        files["front_dominant_hard_preview"] = self._save(
            previews,
            "front_dominant_hard_preview.png",
            hard,
        )
        files["side_visible_mask"] = self._save(
            previews,
            "side_visible_mask.png",
            self._mask_image(side_mask),
        )
        files["front_preserved_mask"] = self._save(
            previews,
            "front_preserved_mask.png",
            self._mask_image(front_mask),
        )
        for width in feather_widths:
            if int(width) == 0:
                continue
            preview, _, _ = self.front_dominant_preview(
                front,
                side,
                boundary_result,
                pair_role.side_position,
                int(width),
            )
            key = f"front_dominant_feather_{int(width)}"
            files[key] = self._save(previews, f"{key}.png", preview)
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
    def component_heatmap(
        canvas_shape: tuple[int, int],
        cost_result: SeamCostResult,
        component: np.ndarray | None,
    ) -> np.ndarray:
        height, width = canvas_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        start, end = cost_result.x_range
        if component is None:
            return canvas
        valid = cost_result.valid_mask
        normalized = np.zeros(component.shape, dtype=np.uint8)
        normalized[valid] = np.clip(component[valid] * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        heatmap[~valid] = (0, 0, 80)
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
    def boundary_overlay(
        front: np.ndarray,
        side: np.ndarray,
        boundary_result: BoundarySearchResult,
    ) -> np.ndarray:
        valid_front = np.any(front != 0, axis=2)
        valid_side = np.any(side != 0, axis=2)
        base = np.zeros_like(front)
        both = valid_front & valid_side
        base[valid_front & ~valid_side] = front[valid_front & ~valid_side]
        base[valid_side & ~valid_front] = side[valid_side & ~valid_front]
        base[both] = (
            front[both].astype(np.float32) * 0.65
            + side[both].astype(np.float32) * 0.35
        ).astype(np.uint8)
        vertical = boundary_result.vertical
        if vertical.x_canvas is not None:
            cv2.line(base, (vertical.x_canvas, 0), (vertical.x_canvas, base.shape[0] - 1), (0, 255, 0), 2)
        selected = boundary_result.recommended
        if selected and selected.points_canvas:
            points = np.asarray(selected.points_canvas, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(base, [points], False, (0, 255, 255), 2, cv2.LINE_AA)
        return base

    @staticmethod
    def front_dominant_preview(
        front: np.ndarray,
        side: np.ndarray,
        boundary_result: BoundarySearchResult,
        side_position: str,
        feather_width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = front.shape[:2]
        valid_front = np.any(front != 0, axis=2)
        valid_side = np.any(side != 0, axis=2)
        seam_x = SeamPreviewRenderer._boundary_x_by_row(boundary_result, height, width)
        columns = np.arange(width, dtype=np.float32)[None, :]
        seam = seam_x[:, None].astype(np.float32)
        if side_position == "left":
            side_region = columns < seam
            signed = seam - columns
        else:
            side_region = columns >= seam
            signed = columns - seam

        side_visible = valid_side & (~valid_front | side_region)
        front_preserved = valid_front & ~side_visible
        output = np.zeros_like(front)
        output[front_preserved] = front[front_preserved]
        output[side_visible] = side[side_visible]

        feather_width = max(0, int(feather_width))
        both = valid_front & valid_side
        if feather_width > 0 and np.any(both):
            t = np.clip((signed + feather_width / 2.0) / max(1.0, feather_width), 0.0, 1.0)
            side_alpha = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
            front_alpha = 1.0 - side_alpha
            blended = (
                side.astype(np.float32) * side_alpha[:, :, None]
                + front.astype(np.float32) * front_alpha[:, :, None]
            )
            transition = both & (np.abs(columns - seam) <= feather_width / 2.0)
            output[transition] = np.clip(blended[transition], 0, 255).astype(np.uint8)
        points = np.asarray(
            [(int(seam_x[y]), y) for y in range(height)],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        if len(points) > 1:
            cv2.polylines(output, [points], False, (0, 255, 255), 1, cv2.LINE_AA)
        return output, side_visible, front_preserved

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
    def _boundary_x_by_row(
        boundary_result: BoundarySearchResult,
        height: int,
        width: int,
    ) -> np.ndarray:
        candidate = boundary_result.recommended or boundary_result.vertical
        fallback = candidate.x_canvas if candidate.x_canvas is not None else width // 2
        seam_x = np.full((height,), float(fallback), dtype=np.float32)
        for x, y in candidate.points_canvas:
            if 0 <= y < height:
                seam_x[y] = float(max(0, min(width - 1, int(x))))
        return seam_x

    @staticmethod
    def _mask_image(mask: np.ndarray) -> np.ndarray:
        return np.dstack([mask.astype(np.uint8) * 255] * 3)

    @staticmethod
    def _save(directory: Path, filename: str, image: np.ndarray) -> str:
        path = directory / filename
        save_image(path, image)
        return path.relative_to(directory.parent).as_posix()
