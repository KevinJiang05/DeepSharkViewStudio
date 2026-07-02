"""Quality gates for front-priority seam candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .boundary import BoundaryCandidate, BoundarySearchResult
from .cost import SeamCostResult


@dataclass(frozen=True)
class QualityGateParams:
    side_invasion_ratio_gt: float = 0.55
    high_edge_ratio_on_boundary_gt: float = 0.30
    high_difference_ratio_on_boundary_gt: float = 0.35
    object_proximity_ratio_on_boundary_gt: float = 0.35
    boundary_jaggedness_gt: float = 0.35
    side_visible_ratio_lt: float = 0.10
    overlap_high_difference_ratio_gt: float = 0.50
    vertical_dp_disagreement_ratio_gt: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class QualityGateResult:
    status: str
    reject_reasons: list[str]
    warning_reasons: list[str]
    front_preserved_ratio: float
    side_visible_ratio: float
    side_invasion_ratio: float
    high_edge_ratio_on_boundary: float
    high_difference_ratio_on_boundary: float
    object_proximity_ratio_on_boundary: float
    boundary_jaggedness: float
    boundary_switch_risk: float
    params: QualityGateParams

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reject_reasons": list(self.reject_reasons),
            "warning_reasons": list(self.warning_reasons),
            "front_preserved_ratio": self.front_preserved_ratio,
            "side_visible_ratio": self.side_visible_ratio,
            "side_invasion_ratio": self.side_invasion_ratio,
            "high_edge_ratio_on_boundary": self.high_edge_ratio_on_boundary,
            "high_difference_ratio_on_boundary": self.high_difference_ratio_on_boundary,
            "object_proximity_ratio_on_boundary": self.object_proximity_ratio_on_boundary,
            "boundary_jaggedness": self.boundary_jaggedness,
            "boundary_switch_risk": self.boundary_switch_risk,
            "params": self.params.to_dict(),
        }


class FrontPriorityQualityGate:
    def evaluate(
        self,
        cost_result: SeamCostResult,
        boundary_result: BoundarySearchResult,
        params: QualityGateParams | None = None,
    ) -> QualityGateResult:
        params = params or QualityGateParams()
        candidate = boundary_result.recommended
        if candidate is None:
            return QualityGateResult(
                status="rejected",
                reject_reasons=["no accepted boundary candidate"],
                warning_reasons=[],
                front_preserved_ratio=0.0,
                side_visible_ratio=0.0,
                side_invasion_ratio=1.0,
                high_edge_ratio_on_boundary=1.0,
                high_difference_ratio_on_boundary=1.0,
                object_proximity_ratio_on_boundary=1.0,
                boundary_jaggedness=1.0,
                boundary_switch_risk=1.0,
                params=params,
            )

        overlap_width = max(1, cost_result.cost.shape[1])
        side_invasion = float(candidate.side_invasion_ratio)
        side_visible = side_invasion
        front_preserved = float(np.clip(1.0 - side_invasion, 0.0, 1.0))
        jaggedness_norm = float(np.clip(candidate.jaggedness / overlap_width, 0.0, 1.0))
        switch_risk = self._switch_risk(boundary_result, overlap_width)
        reject: list[str] = []
        warn: list[str] = []

        self._reject_if(
            reject,
            side_invasion > params.side_invasion_ratio_gt,
            f"side_invasion_ratio {side_invasion:.3f} > {params.side_invasion_ratio_gt:.3f}",
        )
        self._reject_if(
            reject,
            candidate.high_edge_ratio > params.high_edge_ratio_on_boundary_gt,
            f"high_edge_ratio_on_boundary {candidate.high_edge_ratio:.3f} > {params.high_edge_ratio_on_boundary_gt:.3f}",
        )
        self._reject_if(
            reject,
            candidate.high_difference_ratio > params.high_difference_ratio_on_boundary_gt,
            f"high_difference_ratio_on_boundary {candidate.high_difference_ratio:.3f} > {params.high_difference_ratio_on_boundary_gt:.3f}",
        )
        self._reject_if(
            reject,
            candidate.object_proximity_ratio > params.object_proximity_ratio_on_boundary_gt,
            f"object_proximity_ratio_on_boundary {candidate.object_proximity_ratio:.3f} > {params.object_proximity_ratio_on_boundary_gt:.3f}",
        )
        self._reject_if(
            reject,
            jaggedness_norm > params.boundary_jaggedness_gt,
            f"boundary_jaggedness {jaggedness_norm:.3f} > {params.boundary_jaggedness_gt:.3f}",
        )
        if side_visible < params.side_visible_ratio_lt:
            warn.append(
                f"side_visible_ratio {side_visible:.3f} < {params.side_visible_ratio_lt:.3f}"
            )
        if self._overlap_high_difference_ratio(cost_result) > params.overlap_high_difference_ratio_gt:
            warn.append("overlap is dominated by high difference pixels")
        if switch_risk > params.vertical_dp_disagreement_ratio_gt:
            warn.append("vertical and DP candidates disagree strongly")

        status = "rejected" if reject else "warning" if warn else "accepted"
        return QualityGateResult(
            status=status,
            reject_reasons=reject,
            warning_reasons=warn,
            front_preserved_ratio=front_preserved,
            side_visible_ratio=side_visible,
            side_invasion_ratio=side_invasion,
            high_edge_ratio_on_boundary=float(candidate.high_edge_ratio),
            high_difference_ratio_on_boundary=float(candidate.high_difference_ratio),
            object_proximity_ratio_on_boundary=float(candidate.object_proximity_ratio),
            boundary_jaggedness=jaggedness_norm,
            boundary_switch_risk=switch_risk,
            params=params,
        )

    @staticmethod
    def _reject_if(reasons: list[str], condition: bool, reason: str) -> None:
        if condition:
            reasons.append(reason)

    @staticmethod
    def _switch_risk(boundary_result: BoundarySearchResult, overlap_width: int) -> float:
        vertical = boundary_result.vertical
        dp = boundary_result.dp
        if vertical.x_canvas is None or dp.x_canvas is None:
            return 1.0
        return float(min(1.0, abs(vertical.x_canvas - dp.x_canvas) / max(1, overlap_width)))

    @staticmethod
    def _overlap_high_difference_ratio(cost_result: SeamCostResult) -> float:
        component = cost_result.high_difference_cost
        if component is None:
            return 0.0
        valid = cost_result.valid_mask
        values = component[valid]
        if values.size == 0:
            return 1.0
        return float(np.count_nonzero(values >= 0.75) / values.size)
