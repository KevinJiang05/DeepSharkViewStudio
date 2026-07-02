"""Top-bottom side suppression for front-priority layout previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .layout_preview import (
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
)


@dataclass(frozen=True)
class VerticalSafetyParams:
    output_height_px: int
    vertical_safe_ratio: float
    side_vertical_fade_px: int

    @property
    def layout_id(self) -> str:
        safe_value = int(round(self.vertical_safe_ratio * 100))
        return (
            f"h{int(self.output_height_px)}"
            f"_safe{safe_value:03d}"
            f"_fade{int(self.side_vertical_fade_px):03d}"
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class VerticalSafetyResult:
    layout_id: str
    image: np.ndarray
    vertical_side_weight_mask: np.ndarray
    side_visible_before: np.ndarray
    side_visible_after: np.ndarray
    side_suppressed_top_bottom: np.ndarray
    front_preserved_mask: np.ndarray
    invalid_or_black_mask: np.ndarray
    crop_y: tuple[int, int]
    metrics: dict[str, float | int | list[str]]


class VerticalSafetyRenderer:
    """Suppress side-camera contribution near top/bottom fisheye-risk bands."""

    def render(
        self,
        warped_images: dict[str, np.ndarray],
        pair_candidates: list[LayoutPairCandidate],
        layout_params: LayoutPreviewParams,
        safety_params: VerticalSafetyParams,
        main_camera: str = "front",
        pair_params: dict[str, LayoutPreviewParams] | None = None,
    ) -> VerticalSafetyResult:
        base = FrontPriorityLayoutPreviewRenderer().render(
            warped_images,
            pair_candidates,
            layout_params,
            main_camera=main_camera,
            pair_params=pair_params,
        )
        if main_camera not in warped_images:
            raise ValueError("warped_images must include the front camera.")
        front = warped_images[main_camera]
        crop_x0, crop_x1 = base.crop
        front_crop = front[:, crop_x0:crop_x1]
        front_valid = np.any(front_crop != 0, axis=2)
        height, width = base.image.shape[:2]
        weight_1d = self.vertical_weight(
            height,
            safety_params.vertical_safe_ratio,
            safety_params.side_vertical_fade_px,
        )
        weight = np.broadcast_to(weight_1d[:, None], (height, width)).astype(np.float32)
        output = base.image.copy()
        side_before = base.side_visible_mask
        side_after = side_before & (weight > 0.05)
        suppressed = side_before & (weight < 0.999)
        if np.any(side_before):
            alpha = weight[:, :, None]
            front_blend = np.zeros_like(output)
            front_blend[front_valid] = front_crop[front_valid]
            faded = output.astype(np.float32) * alpha + front_blend.astype(np.float32) * (1.0 - alpha)
            output[side_before] = np.clip(faded[side_before], 0, 255).astype(np.uint8)

        crop_y0, crop_y1 = self.crop_y_bounds(height, safety_params.output_height_px)
        cropped = output[crop_y0:crop_y1].copy()
        weight_crop = weight[crop_y0:crop_y1]
        before_crop = side_before[crop_y0:crop_y1]
        after_crop = side_after[crop_y0:crop_y1]
        suppressed_crop = suppressed[crop_y0:crop_y1]
        front_valid_crop = front_valid[crop_y0:crop_y1]
        front_preserved = front_valid_crop & ~after_crop
        black = np.all(cropped == 0, axis=2)
        self._draw_label(cropped, layout_params, safety_params)
        metrics = self._metrics(
            front_valid_crop,
            before_crop,
            after_crop,
            suppressed_crop,
            black,
            weight_crop,
            height,
            crop_y0,
            crop_y1,
        )
        return VerticalSafetyResult(
            layout_id=safety_params.layout_id,
            image=cropped,
            vertical_side_weight_mask=weight_crop,
            side_visible_before=before_crop,
            side_visible_after=after_crop,
            side_suppressed_top_bottom=suppressed_crop,
            front_preserved_mask=front_preserved,
            invalid_or_black_mask=black,
            crop_y=(crop_y0, crop_y1),
            metrics=metrics,
        )

    @staticmethod
    def vertical_weight(
        height: int,
        vertical_safe_ratio: float,
        fade_px: int,
    ) -> np.ndarray:
        height = int(max(1, height))
        ratio = float(np.clip(vertical_safe_ratio, 0.0, 1.0))
        safe_height = height * ratio
        safe_y0 = (height - safe_height) / 2.0
        safe_y1 = safe_y0 + safe_height
        y = np.arange(height, dtype=np.float32)
        weight = np.ones((height,), dtype=np.float32)
        if ratio >= 0.999:
            return weight
        top_distance = safe_y0 - y
        bottom_distance = y - (safe_y1 - 1.0)
        risk_distance = np.maximum(top_distance, bottom_distance)
        outside = risk_distance > 0.0
        if int(fade_px) <= 0:
            weight[outside] = 0.0
            return weight
        top_risk = max(0.0, safe_y0)
        bottom_risk = max(0.0, height - safe_y1)
        effective_fade = max(1.0, min(float(fade_px), max(top_risk, bottom_risk, 1.0)))
        t = np.clip(1.0 - risk_distance / effective_fade, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        weight[outside] = smooth[outside]
        return weight.astype(np.float32)

    @staticmethod
    def crop_y_bounds(height: int, output_height: int) -> tuple[int, int]:
        crop_height = int(max(1, min(height, output_height)))
        y0 = int(round((height - crop_height) / 2.0))
        y0 = max(0, min(height - crop_height, y0))
        return y0, y0 + crop_height

    @staticmethod
    def _metrics(
        front_valid: np.ndarray,
        side_before: np.ndarray,
        side_after: np.ndarray,
        suppressed: np.ndarray,
        black: np.ndarray,
        weight: np.ndarray,
        original_height: int,
        crop_y0: int,
        crop_y1: int,
    ) -> dict[str, float | int | list[str]]:
        area = max(1, int(front_valid.size))
        front_count = max(1, int(np.count_nonzero(front_valid)))
        before_count = max(1, int(np.count_nonzero(side_before)))
        top_bottom = weight < 0.999
        top_bottom_area = max(1, int(np.count_nonzero(top_bottom)))
        duplicate_before = side_before & front_valid & top_bottom
        duplicate_after = side_after & front_valid & top_bottom
        metrics = {
            "front_preserved_ratio": float(np.count_nonzero(front_valid & ~side_after) / front_count),
            "side_visible_ratio": float(np.count_nonzero(side_after) / area),
            "side_suppressed_top_bottom_ratio": float(np.count_nonzero(suppressed) / before_count),
            "black_pixel_ratio": float(np.count_nonzero(black) / area),
            "top_bottom_black_pixel_ratio": float(np.count_nonzero(black & top_bottom) / top_bottom_area),
            "duplicate_risk_proxy_before": float(np.count_nonzero(duplicate_before) / area),
            "duplicate_risk_proxy_after": float(np.count_nonzero(duplicate_after) / area),
            "retained_area_ratio": float((crop_y1 - crop_y0) / max(1, original_height)),
            "warning_reasons": [],
        }
        warnings: list[str] = []
        if metrics["black_pixel_ratio"] > 0.35:
            warnings.append("high black_pixel_ratio")
        if metrics["retained_area_ratio"] < 0.82:
            warnings.append("large vertical crop")
        if metrics["side_visible_ratio"] < 0.04:
            warnings.append("very low side_visible_ratio")
        if metrics["duplicate_risk_proxy_after"] > metrics["duplicate_risk_proxy_before"] + 1e-6:
            warnings.append("duplicate risk proxy increased")
        metrics["warning_reasons"] = warnings
        return metrics

    @staticmethod
    def _draw_label(
        image: np.ndarray,
        layout_params: LayoutPreviewParams,
        safety_params: VerticalSafetyParams,
    ) -> None:
        label = (
            f"{safety_params.layout_id} "
            f"base={layout_params.layout_id}"
        )
        cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 760), 24), (0, 0, 0), -1)
        cv2.putText(image, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
