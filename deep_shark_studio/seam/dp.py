"""Vertical dynamic-programming seam search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DPSeamParams:
    max_step_px: int = 6
    smoothness_weight: float = 0.8
    downsample: int = 2

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SeamPath:
    points_roi: list[tuple[int, int]]
    points_canvas: list[tuple[int, int]]
    mean_cost: float
    p95_cost: float
    jaggedness: float


class VerticalDPSeamFinder:
    """Find one top-to-bottom seam column per row using local DP."""

    def find(
        self,
        cost: np.ndarray,
        valid_mask: np.ndarray,
        params: DPSeamParams | None = None,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> SeamPath:
        params = params or DPSeamParams()
        if cost.shape != valid_mask.shape:
            raise ValueError("Cost and valid_mask must have matching shapes.")
        if cost.ndim != 2:
            raise ValueError("Cost map must be 2D.")
        height, width = cost.shape
        if height == 0 or width == 0:
            return SeamPath([], [], 0.0, 0.0, 0.0)

        downsample = max(1, int(params.downsample))
        ds_cost, ds_valid = self._downsample_cost(cost, valid_mask, downsample)
        path_ds = self._find_downsampled(ds_cost, ds_valid, params, downsample)
        points_roi = self._expand_path(path_ds, height, width, downsample)
        points_canvas = [
            (int(x + x_offset), int(y + y_offset))
            for x, y in points_roi
        ]
        samples = np.asarray(
            [float(cost[y, x]) for x, y in points_roi],
            dtype=np.float64,
        )
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            mean_cost = 0.0
            p95_cost = 0.0
        else:
            mean_cost = float(np.mean(finite))
            p95_cost = float(np.percentile(finite, 95))
        xs = np.asarray([x for x, _ in points_roi], dtype=np.float64)
        jaggedness = float(np.mean(np.abs(np.diff(xs)))) if len(xs) > 1 else 0.0
        return SeamPath(
            points_roi=points_roi,
            points_canvas=points_canvas,
            mean_cost=mean_cost,
            p95_cost=p95_cost,
            jaggedness=jaggedness,
        )

    @staticmethod
    def _downsample_cost(
        cost: np.ndarray,
        valid_mask: np.ndarray,
        downsample: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if downsample <= 1:
            return cost.astype(np.float64), valid_mask.astype(bool)
        height, width = cost.shape
        ds_height = int(math.ceil(height / downsample))
        ds_width = int(math.ceil(width / downsample))
        invalid_value = float(np.nanmax(cost[np.isfinite(cost)])) if np.any(np.isfinite(cost)) else 1.0
        ds_cost = np.full((ds_height, ds_width), invalid_value, dtype=np.float64)
        ds_valid = np.zeros((ds_height, ds_width), dtype=bool)
        for y in range(ds_height):
            y0 = y * downsample
            y1 = min(height, y0 + downsample)
            for x in range(ds_width):
                x0 = x * downsample
                x1 = min(width, x0 + downsample)
                block_valid = valid_mask[y0:y1, x0:x1]
                if not np.any(block_valid):
                    continue
                block_cost = cost[y0:y1, x0:x1][block_valid]
                ds_valid[y, x] = True
                ds_cost[y, x] = float(np.min(block_cost))
        return ds_cost, ds_valid

    @staticmethod
    def _find_downsampled(
        cost: np.ndarray,
        valid_mask: np.ndarray,
        params: DPSeamParams,
        downsample: int,
    ) -> list[int]:
        height, width = cost.shape
        row_cost = cost.astype(np.float64).copy()
        row_valid = valid_mask.astype(bool).copy()
        fallback_value = float(np.nanmax(row_cost[np.isfinite(row_cost)])) if np.any(np.isfinite(row_cost)) else 1.0
        for y in range(height):
            if np.any(row_valid[y]):
                continue
            row_valid[y, :] = True
            row_cost[y, :] = fallback_value

        max_step = max(0, int(math.ceil(max(1, params.max_step_px) / downsample)))
        smoothness = float(params.smoothness_weight)
        dp = np.full((height, width), np.inf, dtype=np.float64)
        back = np.full((height, width), -1, dtype=np.int32)
        dp[0] = np.where(row_valid[0], row_cost[0], np.inf)

        for y in range(1, height):
            for x in range(width):
                if not row_valid[y, x]:
                    continue
                start = max(0, x - max_step)
                end = min(width, x + max_step + 1)
                previous_x = np.arange(start, end, dtype=np.float64)
                penalty = smoothness * ((previous_x - x) ** 2)
                candidates = dp[y - 1, start:end] + penalty
                best_local = int(np.argmin(candidates))
                best_value = candidates[best_local]
                if np.isfinite(best_value):
                    dp[y, x] = row_cost[y, x] + best_value
                    back[y, x] = start + best_local

        if not np.any(np.isfinite(dp[-1])):
            center = width // 2
            return [center for _ in range(height)]

        path = [int(np.argmin(dp[-1]))]
        for y in range(height - 1, 0, -1):
            previous = int(back[y, path[-1]])
            if previous < 0:
                previous = path[-1]
            path.append(previous)
        path.reverse()
        return path

    @staticmethod
    def _expand_path(
        path_ds: list[int],
        height: int,
        width: int,
        downsample: int,
    ) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for y in range(height):
            ds_y = min(len(path_ds) - 1, y // max(1, downsample))
            x = path_ds[ds_y] * max(1, downsample) + max(1, downsample) // 2
            points.append((int(max(0, min(width - 1, x))), y))
        return points
