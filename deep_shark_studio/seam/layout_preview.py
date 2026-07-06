"""Preview compositor for front-priority layout sweep candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


def valid_mask_for_image(
    image: np.ndarray,
    mask: np.ndarray | None,
    camera: str,
) -> np.ndarray:
    if mask is None:
        return np.any(image != 0, axis=2)
    mask_array = np.asarray(mask)
    if mask_array.shape != image.shape[:2]:
        raise ValueError(f"valid_mask size mismatch for {camera}")
    return mask_array.astype(bool)


@dataclass(frozen=True)
class LayoutPreviewParams:
    side_shift_px: int = 40
    output_width_px: int = 1800
    side_visible_fraction: float = 0.30
    feather_width_px: int = 24

    @property
    def layout_id(self) -> str:
        side_value = int(round(self.side_visible_fraction * 100))
        return (
            f"shift_{int(self.side_shift_px):03d}"
            f"_crop_{int(self.output_width_px)}"
            f"_side_{side_value:03d}"
            f"_feather_{int(self.feather_width_px):02d}"
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LayoutPairCandidate:
    pair_id: str
    side_camera: str
    side_position: str
    boundary_type: str
    seam_points: list[tuple[int, int]]

    def shifted_points(self, shift_px: int, canvas_width: int) -> list[tuple[int, int]]:
        delta = int(shift_px) if self.side_position == "left" else -int(shift_px)
        return [
            (int(max(0, min(canvas_width - 1, x + delta))), int(y))
            for x, y in self.seam_points
        ]


@dataclass(frozen=True)
class LayoutPreviewResult:
    layout_id: str
    image: np.ndarray
    side_visible_mask: np.ndarray
    front_preserved_mask: np.ndarray
    side_suppressed_mask: np.ndarray
    metrics: dict[str, float | int | list[str]]
    crop: tuple[int, int]


class FrontPriorityLayoutPreviewRenderer:
    """Apply preview-only side shift, clamp, narrow feather, and crop."""

    def render(
        self,
        warped_images: dict[str, np.ndarray],
        pair_candidates: list[LayoutPairCandidate],
        params: LayoutPreviewParams,
        main_camera: str = "front",
        pair_params: dict[str, LayoutPreviewParams] | None = None,
        draw_label: bool = True,
        valid_masks: dict[str, np.ndarray] | None = None,
    ) -> LayoutPreviewResult:
        if main_camera not in warped_images:
            raise ValueError("warped_images must include the front camera.")
        input_masks = valid_masks or {}
        front = warped_images[main_camera]
        height, width = front.shape[:2]
        front_valid = valid_mask_for_image(
            front,
            input_masks.get(main_camera),
            main_camera,
        )
        output = np.zeros_like(front)
        output[front_valid] = front[front_valid]
        side_visible_total = np.zeros((height, width), dtype=bool)
        side_suppressed_total = np.zeros((height, width), dtype=bool)
        boundary_count = 0

        for candidate in pair_candidates:
            if candidate.side_camera not in warped_images:
                continue
            candidate_params = (
                pair_params.get(candidate.pair_id, params)
                if pair_params
                else params
            )
            side = self.shift_side_image(
                warped_images[candidate.side_camera],
                candidate.side_position,
                candidate_params.side_shift_px,
            )
            side_source_valid = valid_mask_for_image(
                warped_images[candidate.side_camera],
                input_masks.get(candidate.side_camera),
                candidate.side_camera,
            )
            side_valid = self.shift_side_image(
                side_source_valid.astype(np.uint8),
                candidate.side_position,
                candidate_params.side_shift_px,
            ).astype(bool)
            seam_x = self._seam_x_by_row(
                candidate.shifted_points(candidate_params.side_shift_px, width),
                height,
                width,
            )
            side_region = self._side_region(width, seam_x, candidate.side_position)
            clamp_region = self._clamp_region(
                height,
                width,
                candidate.side_position,
                candidate_params.side_visible_fraction,
            )
            visible = side_valid & clamp_region & (~front_valid | side_region)
            suppressed = side_valid & ~visible
            output[visible] = side[visible]
            self._apply_feather(
                output,
                front,
                side,
                front_valid,
                side_valid,
                seam_x,
                clamp_region,
                candidate.side_position,
                candidate_params.feather_width_px,
            )
            side_visible_total |= visible
            side_suppressed_total |= suppressed
            boundary_count += 1

        front_preserved = front_valid & ~side_visible_total
        crop_x0, crop_x1 = self._crop_bounds(front_valid, width, params.output_width_px)
        cropped = output[:, crop_x0:crop_x1].copy()
        side_visible_crop = side_visible_total[:, crop_x0:crop_x1]
        front_preserved_crop = front_preserved[:, crop_x0:crop_x1]
        side_suppressed_crop = side_suppressed_total[:, crop_x0:crop_x1]
        metrics = self._metrics(
            cropped,
            front_valid[:, crop_x0:crop_x1],
            side_visible_crop,
            front_preserved_crop,
            side_suppressed_crop,
            width,
            crop_x0,
            crop_x1,
            boundary_count,
        )
        if draw_label:
            self._draw_label(cropped, params, pair_candidates)
        return LayoutPreviewResult(
            layout_id=params.layout_id,
            image=cropped,
            side_visible_mask=side_visible_crop,
            front_preserved_mask=front_preserved_crop,
            side_suppressed_mask=side_suppressed_crop,
            metrics=metrics,
            crop=(crop_x0, crop_x1),
        )

    @staticmethod
    def shift_side_image(
        image: np.ndarray,
        side_position: str,
        side_shift_px: int,
    ) -> np.ndarray:
        shift = max(0, int(side_shift_px))
        if shift == 0:
            return image.copy()
        result = np.zeros_like(image)
        if shift >= image.shape[1]:
            return result
        if side_position == "left":
            result[:, shift:] = image[:, : image.shape[1] - shift]
        elif side_position == "right":
            result[:, : image.shape[1] - shift] = image[:, shift:]
        else:
            raise ValueError("side_position must be 'left' or 'right'.")
        return result

    @staticmethod
    def _seam_x_by_row(
        seam_points: list[tuple[int, int]],
        height: int,
        width: int,
    ) -> np.ndarray:
        fallback = width / 2.0
        if seam_points:
            fallback = float(np.median([x for x, _ in seam_points]))
        seam_x = np.full((height,), fallback, dtype=np.float32)
        for x, y in seam_points:
            if 0 <= y < height:
                seam_x[y] = float(max(0, min(width - 1, int(x))))
        return seam_x

    @staticmethod
    def _side_region(width: int, seam_x: np.ndarray, side_position: str) -> np.ndarray:
        columns = np.arange(width, dtype=np.float32)[None, :]
        if side_position == "left":
            return columns < seam_x[:, None]
        return columns >= seam_x[:, None]

    @staticmethod
    def _clamp_region(
        height: int,
        width: int,
        side_position: str,
        side_visible_fraction: float,
    ) -> np.ndarray:
        side_width = int(round(width * float(side_visible_fraction)))
        side_width = max(0, min(width, side_width))
        mask = np.zeros((height, width), dtype=bool)
        if side_position == "left":
            mask[:, :side_width] = True
        else:
            mask[:, width - side_width :] = True
        return mask

    @staticmethod
    def _apply_feather(
        output: np.ndarray,
        front: np.ndarray,
        side: np.ndarray,
        front_valid: np.ndarray,
        side_valid: np.ndarray,
        seam_x: np.ndarray,
        clamp_region: np.ndarray,
        side_position: str,
        feather_width_px: int,
    ) -> None:
        feather = max(0, int(feather_width_px))
        if feather == 0:
            return
        height, width = front.shape[:2]
        columns = np.arange(width, dtype=np.float32)[None, :]
        side_width_by_row = np.sum(clamp_region, axis=1).astype(np.float32)
        if side_position == "left":
            clamp_edge = np.maximum(0.0, side_width_by_row - 1.0)
            effective = np.minimum(seam_x, clamp_edge)
            signed = effective[:, None] - columns
        else:
            clamp_edge = width - side_width_by_row
            effective = np.maximum(seam_x, clamp_edge)
            signed = columns - effective[:, None]
        t = np.clip((signed + feather / 2.0) / max(1.0, feather), 0.0, 1.0)
        side_alpha = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        front_alpha = 1.0 - side_alpha
        transition = front_valid & side_valid & (np.abs(signed) <= feather / 2.0)
        blended = (
            side.astype(np.float32) * side_alpha[:, :, None]
            + front.astype(np.float32) * front_alpha[:, :, None]
        )
        output[transition] = np.clip(blended[transition], 0, 255).astype(np.uint8)

    @staticmethod
    def _crop_bounds(front_valid: np.ndarray, width: int, output_width: int) -> tuple[int, int]:
        crop_width = int(max(1, min(width, output_width)))
        xs = np.where(front_valid)[1]
        center = int(round(float(np.mean(xs)))) if xs.size else width // 2
        x0 = int(round(center - crop_width / 2.0))
        x0 = max(0, min(width - crop_width, x0))
        return x0, x0 + crop_width

    @staticmethod
    def _metrics(
        image: np.ndarray,
        front_valid: np.ndarray,
        side_visible: np.ndarray,
        front_preserved: np.ndarray,
        side_suppressed: np.ndarray,
        full_width: int,
        crop_x0: int,
        crop_x1: int,
        boundary_count: int,
    ) -> dict[str, float | int | list[str]]:
        area = max(1, int(side_visible.size))
        front_count = max(1, int(np.count_nonzero(front_valid)))
        side_kept = int(np.count_nonzero(side_visible))
        side_suppressed_count = int(np.count_nonzero(side_suppressed))
        both_protected = int(np.count_nonzero(front_preserved & side_suppressed))
        black = np.all(image == 0, axis=2)
        side_total = max(1, side_kept + side_suppressed_count)
        metrics = {
            "front_preserved_ratio": float(np.count_nonzero(front_preserved) / front_count),
            "side_visible_ratio": float(side_kept / area),
            "side_suppressed_ratio": float(side_suppressed_count / side_total),
            "side_invasion_ratio": float(np.count_nonzero(side_visible & front_valid) / front_count),
            "crop_retained_ratio": float((crop_x1 - crop_x0) / max(1, full_width)),
            "black_pixel_ratio": float(np.count_nonzero(black) / area),
            "duplicate_risk_proxy": float(both_protected / area),
            "boundary_count": int(boundary_count),
            "warning_reasons": [],
        }
        warnings: list[str] = []
        if metrics["black_pixel_ratio"] > 0.35:
            warnings.append("high black_pixel_ratio")
        if metrics["side_visible_ratio"] < 0.03:
            warnings.append("very low side_visible_ratio")
        if metrics["side_invasion_ratio"] > 0.30:
            warnings.append("high side_invasion_ratio")
        if metrics["duplicate_risk_proxy"] > 0.20:
            warnings.append("high duplicate_risk_proxy")
        metrics["warning_reasons"] = warnings
        return metrics

    @staticmethod
    def _draw_label(
        image: np.ndarray,
        params: LayoutPreviewParams,
        pair_candidates: list[LayoutPairCandidate],
    ) -> None:
        boundary = " / ".join(
            f"{candidate.pair_id}={candidate.boundary_type}"
            for candidate in pair_candidates
        )
        label = (
            f"shift={params.side_shift_px}px crop={params.output_width_px}px "
            f"side={params.side_visible_fraction:.2f} feather={params.feather_width_px}px"
        )
        cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 760), 42), (0, 0, 0), -1)
        cv2.putText(image, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, boundary, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
