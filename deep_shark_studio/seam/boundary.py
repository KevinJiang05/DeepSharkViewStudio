"""Side-boundary candidate search for front-priority stitching experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .cost import SeamCostResult
from .dp import SeamPath


@dataclass(frozen=True)
class BoundarySearchParams:
    vertical_fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    high_edge_threshold: float = 0.75
    high_difference_threshold: float = 0.75
    object_proximity_threshold: float = 0.75
    high_edge_weight: float = 0.30
    high_difference_weight: float = 0.35
    object_proximity_weight: float = 0.25
    dp_score_margin: float = 0.08
    max_dp_jaggedness_px: float = 8.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["vertical_fractions"] = list(self.vertical_fractions)
        return data


@dataclass(frozen=True)
class BoundaryCandidate:
    kind: str
    status: str
    score: float
    x_roi: int | None
    x_canvas: int | None
    points_canvas: list[tuple[int, int]]
    mean_cost: float
    p95_cost: float
    high_edge_ratio: float
    high_difference_ratio: float
    object_proximity_ratio: float
    jaggedness: float
    side_invasion_ratio: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "status": self.status,
            "score": self.score,
            "x_roi": self.x_roi,
            "x_canvas": self.x_canvas,
            "mean_cost": self.mean_cost,
            "p95_cost": self.p95_cost,
            "high_edge_ratio": self.high_edge_ratio,
            "high_difference_ratio": self.high_difference_ratio,
            "object_proximity_ratio": self.object_proximity_ratio,
            "jaggedness": self.jaggedness,
            "side_invasion_ratio": self.side_invasion_ratio,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BoundarySearchResult:
    vertical: BoundaryCandidate
    dp: BoundaryCandidate
    recommended: BoundaryCandidate | None
    recommended_boundary_type: str
    params: BoundarySearchParams

    def to_dict(self) -> dict[str, object]:
        return {
            "vertical": self.vertical.to_dict(),
            "dp": self.dp.to_dict(),
            "recommended_boundary_type": self.recommended_boundary_type,
            "params": self.params.to_dict(),
        }


class SideBoundaryCandidateFinder:
    """Compare stable vertical boundaries with the free DP seam."""

    def search(
        self,
        cost_result: SeamCostResult,
        dp_path: SeamPath,
        side_position: str,
        params: BoundarySearchParams | None = None,
    ) -> BoundarySearchResult:
        params = params or BoundarySearchParams()
        vertical = self._best_vertical(cost_result, side_position, params)
        dp = self._dp_candidate(cost_result, dp_path, side_position, params)

        recommended: BoundaryCandidate | None = vertical
        recommended_type = "vertical"
        if dp.status != "rejected":
            dp_better = dp.score + params.dp_score_margin < vertical.score
            dp_stable = dp.jaggedness <= params.max_dp_jaggedness_px
            if dp_better and dp_stable:
                recommended = dp
                recommended_type = "dp"
        if vertical.status == "rejected" and dp.status == "rejected":
            recommended = None
            recommended_type = "none"
        return BoundarySearchResult(
            vertical=vertical,
            dp=dp,
            recommended=recommended,
            recommended_boundary_type=recommended_type,
            params=params,
        )

    def _best_vertical(
        self,
        cost_result: SeamCostResult,
        side_position: str,
        params: BoundarySearchParams,
    ) -> BoundaryCandidate:
        height, width = cost_result.cost.shape
        best: BoundaryCandidate | None = None
        for fraction in params.vertical_fractions:
            x_roi = int(round(np.clip(fraction, 0.0, 1.0) * (width - 1)))
            candidate = self._vertical_candidate(cost_result, x_roi, side_position, params)
            if best is None or candidate.score < best.score:
                best = candidate
        if best is None:
            start, _ = cost_result.x_range
            return BoundaryCandidate(
                kind="vertical",
                status="rejected",
                score=math.inf,
                x_roi=None,
                x_canvas=None,
                points_canvas=[],
                mean_cost=math.inf,
                p95_cost=math.inf,
                high_edge_ratio=1.0,
                high_difference_ratio=1.0,
                object_proximity_ratio=1.0,
                jaggedness=0.0,
                side_invasion_ratio=0.0,
                reasons=[f"no vertical boundary was searchable from x={start}"],
            )
        return best

    def _vertical_candidate(
        self,
        cost_result: SeamCostResult,
        x_roi: int,
        side_position: str,
        params: BoundarySearchParams,
    ) -> BoundaryCandidate:
        height, width = cost_result.cost.shape
        x_roi = int(max(0, min(width - 1, x_roi)))
        valid = cost_result.valid_mask[:, x_roi]
        values = cost_result.cost[:, x_roi]
        finite_valid = valid & np.isfinite(values) & (values < 1.0e8)
        reasons: list[str] = []
        if not np.any(finite_valid):
            finite_valid = np.isfinite(values)
            reasons.append("boundary column has no valid overlap pixels")
        samples = values[finite_valid]
        if samples.size == 0:
            mean_cost = math.inf
            p95_cost = math.inf
            status = "rejected"
        else:
            mean_cost = float(np.mean(samples))
            p95_cost = float(np.percentile(samples, 95))
            status = "accepted"
        high_edge = self._ratio(cost_result.edge_cost[:, x_roi], finite_valid, params.high_edge_threshold)
        high_diff = self._ratio(
            self._component(cost_result.high_difference_cost, cost_result.cost)[:, x_roi],
            finite_valid,
            params.high_difference_threshold,
        )
        object_proximity = self._ratio(
            self._component(cost_result.object_proximity_cost, cost_result.edge_cost)[:, x_roi],
            finite_valid,
            params.object_proximity_threshold,
        )
        side_invasion = self._side_invasion_ratio(x_roi, width, side_position)
        score = (
            mean_cost
            + p95_cost
            + params.high_edge_weight * high_edge
            + params.high_difference_weight * high_diff
            + params.object_proximity_weight * object_proximity
        )
        x_canvas = int(cost_result.x_range[0] + x_roi)
        points = [(x_canvas, y) for y in range(height)]
        return BoundaryCandidate(
            kind="vertical",
            status=status,
            score=float(score),
            x_roi=x_roi,
            x_canvas=x_canvas,
            points_canvas=points,
            mean_cost=mean_cost,
            p95_cost=p95_cost,
            high_edge_ratio=high_edge,
            high_difference_ratio=high_diff,
            object_proximity_ratio=object_proximity,
            jaggedness=0.0,
            side_invasion_ratio=side_invasion,
            reasons=reasons,
        )

    def _dp_candidate(
        self,
        cost_result: SeamCostResult,
        dp_path: SeamPath,
        side_position: str,
        params: BoundarySearchParams,
    ) -> BoundaryCandidate:
        height, width = cost_result.cost.shape
        points_roi = [
            (int(max(0, min(width - 1, x))), int(y))
            for x, y in dp_path.points_roi
            if 0 <= y < height
        ]
        if not points_roi:
            return BoundaryCandidate(
                kind="dp",
                status="rejected",
                score=math.inf,
                x_roi=None,
                x_canvas=None,
                points_canvas=[],
                mean_cost=math.inf,
                p95_cost=math.inf,
                high_edge_ratio=1.0,
                high_difference_ratio=1.0,
                object_proximity_ratio=1.0,
                jaggedness=math.inf,
                side_invasion_ratio=1.0,
                reasons=["DP returned no seam points"],
            )
        xs = np.asarray([x for x, _ in points_roi], dtype=np.int32)
        ys = np.asarray([y for _, y in points_roi], dtype=np.int32)
        samples = cost_result.cost[ys, xs]
        valid = cost_result.valid_mask[ys, xs] & np.isfinite(samples) & (samples < 1.0e8)
        finite_samples = samples[valid]
        if finite_samples.size == 0:
            mean_cost = math.inf
            p95_cost = math.inf
            status = "rejected"
            reasons = ["DP seam has no valid overlap samples"]
        else:
            mean_cost = float(np.mean(finite_samples))
            p95_cost = float(np.percentile(finite_samples, 95))
            status = "accepted"
            reasons = []
        edge_values = cost_result.edge_cost[ys, xs]
        high_diff_values = self._component(cost_result.high_difference_cost, cost_result.cost)[ys, xs]
        object_values = self._component(cost_result.object_proximity_cost, cost_result.edge_cost)[ys, xs]
        high_edge = self._ratio(edge_values, valid, params.high_edge_threshold)
        high_diff = self._ratio(high_diff_values, valid, params.high_difference_threshold)
        object_proximity = self._ratio(object_values, valid, params.object_proximity_threshold)
        side_invasion = float(
            np.mean([self._side_invasion_ratio(int(x), width, side_position) for x in xs])
        )
        score = (
            mean_cost
            + p95_cost
            + params.high_edge_weight * high_edge
            + params.high_difference_weight * high_diff
            + params.object_proximity_weight * object_proximity
            + min(1.0, dp_path.jaggedness / max(1.0, width))
        )
        x_canvas = int(round(float(np.median([x for x, _ in dp_path.points_canvas]))))
        return BoundaryCandidate(
            kind="dp",
            status=status,
            score=float(score),
            x_roi=int(round(float(np.median(xs)))),
            x_canvas=x_canvas,
            points_canvas=list(dp_path.points_canvas),
            mean_cost=mean_cost,
            p95_cost=p95_cost,
            high_edge_ratio=high_edge,
            high_difference_ratio=high_diff,
            object_proximity_ratio=object_proximity,
            jaggedness=float(dp_path.jaggedness),
            side_invasion_ratio=side_invasion,
            reasons=reasons,
        )

    @staticmethod
    def _ratio(values: np.ndarray, valid: np.ndarray, threshold: float) -> float:
        valid_values = values[np.asarray(valid, dtype=bool)]
        if valid_values.size == 0:
            return 1.0
        return float(np.count_nonzero(valid_values >= threshold) / valid_values.size)

    @staticmethod
    def _component(component: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
        return component if component is not None else fallback

    @staticmethod
    def _side_invasion_ratio(x_roi: int, width: int, side_position: str) -> float:
        if width <= 1:
            return 0.0
        fraction = x_roi / float(width - 1)
        if side_position == "right":
            fraction = 1.0 - fraction
        return float(np.clip(fraction, 0.0, 1.0))
