"""Mask generation from a static seam path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dp import SeamPath


@dataclass(frozen=True)
class SeamMaskResult:
    alpha_a: np.ndarray
    alpha_b: np.ndarray
    hard_mask_a: np.ndarray
    hard_mask_b: np.ndarray


class SeamMaskBuilder:
    """Convert a seam path into hard masks and narrow-feather alpha maps."""

    def build(
        self,
        seam_path: SeamPath,
        canvas_shape: tuple[int, int],
        x_range: tuple[int, int],
        feather_width_px: int,
    ) -> SeamMaskResult:
        height, width = int(canvas_shape[0]), int(canvas_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("Canvas shape must be positive.")
        seam_x_by_row = self._seam_x_by_row(seam_path, height, x_range)
        columns = np.arange(width, dtype=np.float32)[None, :]
        seam_x = seam_x_by_row[:, None].astype(np.float32)

        hard_mask_a = columns < seam_x
        hard_mask_b = ~hard_mask_a
        feather_width = max(0, int(feather_width_px))
        if feather_width == 0:
            alpha_a = hard_mask_a.astype(np.float32)
            alpha_b = hard_mask_b.astype(np.float32)
            return SeamMaskResult(alpha_a, alpha_b, hard_mask_a, hard_mask_b)

        half = feather_width / 2.0
        t = np.clip((columns - seam_x + half) / max(1.0, feather_width), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        alpha_b = smooth.astype(np.float32)
        alpha_a = (1.0 - smooth).astype(np.float32)
        return SeamMaskResult(alpha_a, alpha_b, hard_mask_a, hard_mask_b)

    @staticmethod
    def _seam_x_by_row(
        seam_path: SeamPath,
        height: int,
        x_range: tuple[int, int],
    ) -> np.ndarray:
        start, end = sorted((int(x_range[0]), int(x_range[1])))
        seam_x = np.full((height,), (start + end) / 2.0, dtype=np.float32)
        for x, y in seam_path.points_canvas:
            if 0 <= y < height:
                seam_x[y] = float(max(start, min(end, int(x))))
        return seam_x
