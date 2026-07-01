"""Cost-map construction for static pairwise seam candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SeamCostParams:
    color_weight: float = 0.45
    gradient_weight: float = 0.25
    edge_weight: float = 0.20
    margin_weight: float = 0.10
    normalize_percentile: float = 95.0
    safe_erode_px: int = 6
    invalid_cost: float = 1.0e9

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SeamCostResult:
    cost: np.ndarray
    valid_mask: np.ndarray
    color_cost: np.ndarray
    gradient_cost: np.ndarray
    edge_cost: np.ndarray
    margin_cost: np.ndarray
    x_range: tuple[int, int]


class SeamCostBuilder:
    """Build normalized overlap costs without touching project files."""

    def build(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        x_range: tuple[int, int],
        params: SeamCostParams | None = None,
    ) -> SeamCostResult:
        params = params or SeamCostParams()
        if image_a.shape != image_b.shape:
            raise ValueError("Pairwise seam inputs must have matching shapes.")
        if image_a.ndim != 3 or image_a.shape[2] != 3:
            raise ValueError("Pairwise seam inputs must be BGR images.")

        height, width = image_a.shape[:2]
        start, end = sorted((int(round(x_range[0])), int(round(x_range[1]))))
        start = max(0, min(width - 1, start))
        end = max(0, min(width - 1, end))
        if end < start:
            raise ValueError("Overlap x_range is outside the image.")
        roi = slice(start, end + 1)

        roi_a = image_a[:, roi]
        roi_b = image_b[:, roi]

        # Temporary compatibility with the current runtime: black pixels are
        # treated as invalid because SurroundStitcher produces black borders.
        valid_a = np.any(roi_a != 0, axis=2)
        valid_b = np.any(roi_b != 0, axis=2)
        valid_mask = valid_a & valid_b
        if params.safe_erode_px > 0 and np.any(valid_mask):
            kernel_size = max(1, int(params.safe_erode_px) * 2 + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size),
            )
            valid_mask = cv2.erode(
                valid_mask.astype(np.uint8),
                kernel,
                iterations=1,
            ).astype(bool)

        color_cost = self._color_cost(roi_a, roi_b)
        gradient_a = self._gradient_magnitude(roi_a)
        gradient_b = self._gradient_magnitude(roi_b)
        gradient_cost = np.abs(gradient_a - gradient_b)
        edge_cost = np.maximum(gradient_a, gradient_b)
        margin_cost = self._margin_cost(valid_mask, params.safe_erode_px)

        color_cost = self._robust_normalize(
            color_cost,
            valid_mask,
            params.normalize_percentile,
        )
        gradient_cost = self._robust_normalize(
            gradient_cost,
            valid_mask,
            params.normalize_percentile,
        )
        edge_cost = self._robust_normalize(
            edge_cost,
            valid_mask,
            params.normalize_percentile,
        )
        margin_cost = self._robust_normalize(
            margin_cost,
            valid_mask,
            params.normalize_percentile,
        )

        cost = (
            params.color_weight * color_cost
            + params.gradient_weight * gradient_cost
            + params.edge_weight * edge_cost
            + params.margin_weight * margin_cost
        ).astype(np.float32)
        cost[~valid_mask] = np.float32(params.invalid_cost)

        return SeamCostResult(
            cost=cost,
            valid_mask=valid_mask,
            color_cost=color_cost,
            gradient_cost=gradient_cost,
            edge_cost=edge_cost,
            margin_cost=margin_cost,
            x_range=(start, end),
        )

    @staticmethod
    def _color_cost(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        lab_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2LAB).astype(np.float32)
        return np.linalg.norm(lab_a - lab_b, axis=2).astype(np.float32)

    @staticmethod
    def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        return cv2.magnitude(grad_x, grad_y).astype(np.float32)

    @staticmethod
    def _margin_cost(valid_mask: np.ndarray, safe_erode_px: int) -> np.ndarray:
        height, width = valid_mask.shape
        if height == 0 or width == 0:
            return np.zeros_like(valid_mask, dtype=np.float32)
        valid_distance = cv2.distanceTransform(
            valid_mask.astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        x = np.arange(width, dtype=np.float32)
        boundary_distance = np.minimum(x + 1.0, width - x)
        boundary_distance = np.broadcast_to(boundary_distance[None, :], (height, width))
        distance = np.minimum(valid_distance, boundary_distance)
        safe_distance = float(max(1, safe_erode_px * 2, width // 20))
        return (1.0 - np.clip(distance / safe_distance, 0.0, 1.0)).astype(np.float32)

    @staticmethod
    def _robust_normalize(
        values: np.ndarray,
        valid_mask: np.ndarray,
        percentile: float,
    ) -> np.ndarray:
        normalized = np.zeros(values.shape, dtype=np.float32)
        valid_values = values[valid_mask]
        if valid_values.size == 0:
            return normalized
        scale = float(np.percentile(valid_values, percentile))
        if scale <= 1e-6:
            max_value = float(np.max(valid_values))
            scale = max(max_value, 1.0)
        normalized[valid_mask] = np.clip(values[valid_mask] / scale, 0.0, 1.0)
        return normalized
