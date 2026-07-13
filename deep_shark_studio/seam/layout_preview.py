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


@dataclass(frozen=True)
class LayoutPairRuntimePlan:
    """Static per-pair geometry for repeated runtime composition."""

    side_camera: str
    source_offset: int
    source_x0: int
    destination_x0: int
    copy_width: int
    visible_u8: np.ndarray
    transition_x0: int
    transition_x1: int
    transition_u8: np.ndarray
    side_alpha: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for value in (
                    self.visible_u8,
                    self.transition_u8,
                    self.side_alpha,
                )
            )
        )


@dataclass(frozen=True)
class FrontPriorityLayoutRuntimePlan:
    """Immutable crop-local masks and alpha tables for a fixed layout."""

    canvas_shape: tuple[int, int]
    crop: tuple[int, int]
    layout_id: str
    front_valid: np.ndarray
    front_valid_u8: np.ndarray
    pair_plans: tuple[LayoutPairRuntimePlan, ...]
    side_visible: np.ndarray
    front_preserved: np.ndarray
    side_suppressed: np.ndarray
    boundary_count: int

    @property
    def memory_bytes(self) -> int:
        return int(
            self.front_valid.nbytes
            + self.front_valid_u8.nbytes
            + self.side_visible.nbytes
            + self.front_preserved.nbytes
            + self.side_suppressed.nbytes
            + sum(pair.memory_bytes for pair in self.pair_plans)
        )


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
        plan: FrontPriorityLayoutRuntimePlan | None = None,
    ) -> LayoutPreviewResult:
        if plan is not None:
            return self._render_with_runtime_plan(
                warped_images,
                params,
                plan,
                main_camera=main_camera,
                draw_label=draw_label,
            )
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

    def build_runtime_plan(
        self,
        warped_images: dict[str, np.ndarray],
        pair_candidates: list[LayoutPairCandidate],
        params: LayoutPreviewParams,
        *,
        main_camera: str = "front",
        pair_params: dict[str, LayoutPreviewParams] | None = None,
        valid_masks: dict[str, np.ndarray] | None = None,
    ) -> FrontPriorityLayoutRuntimePlan:
        """Precompute geometry that depends only on masks and layout params."""
        if main_camera not in warped_images:
            raise ValueError("warped_images must include the front camera.")
        input_masks = valid_masks or {}
        front = warped_images[main_camera]
        if front.dtype != np.uint8 or front.ndim != 3 or front.shape[2] != 3:
            raise ValueError("Runtime layout plans require uint8 BGR warped images.")
        height, width = front.shape[:2]
        front_valid = valid_mask_for_image(
            front,
            input_masks.get(main_camera),
            main_camera,
        )
        crop_x0, crop_x1 = self._crop_bounds(
            front_valid,
            width,
            params.output_width_px,
        )
        crop_width = crop_x1 - crop_x0
        side_visible_total = np.zeros((height, crop_width), dtype=bool)
        side_suppressed_total = np.zeros((height, crop_width), dtype=bool)
        runtime_pairs: list[LayoutPairRuntimePlan] = []

        for candidate in pair_candidates:
            if candidate.side_camera not in warped_images:
                continue
            side_image = warped_images[candidate.side_camera]
            if side_image.shape != front.shape or side_image.dtype != front.dtype:
                raise ValueError(
                    "Runtime layout plans require matching uint8 BGR canvas shapes."
                )
            candidate_params = (
                pair_params.get(candidate.pair_id, params)
                if pair_params
                else params
            )
            shift = max(0, int(candidate_params.side_shift_px))
            side_source_valid = valid_mask_for_image(
                side_image,
                input_masks.get(candidate.side_camera),
                candidate.side_camera,
            )
            side_valid = self.shift_side_image(
                side_source_valid.astype(np.uint8),
                candidate.side_position,
                shift,
            ).astype(bool)
            seam_x = self._seam_x_by_row(
                candidate.shifted_points(shift, width),
                height,
                width,
            )
            side_region = self._side_region(
                width,
                seam_x,
                candidate.side_position,
            )
            clamp_region = self._clamp_region(
                height,
                width,
                candidate.side_position,
                candidate_params.side_visible_fraction,
            )
            visible_full = side_valid & clamp_region & (~front_valid | side_region)
            suppressed_full = side_valid & ~visible_full
            columns = np.arange(width, dtype=np.float32)[None, :]
            side_width_by_row = np.sum(clamp_region, axis=1).astype(np.float32)
            if candidate.side_position == "left":
                clamp_edge = np.maximum(0.0, side_width_by_row - 1.0)
                effective = np.minimum(seam_x, clamp_edge)
                signed = effective[:, None] - columns
            elif candidate.side_position == "right":
                clamp_edge = width - side_width_by_row
                effective = np.maximum(seam_x, clamp_edge)
                signed = columns - effective[:, None]
            else:
                raise ValueError("side_position must be 'left' or 'right'.")
            feather = max(0, int(candidate_params.feather_width_px))
            if feather:
                t = np.clip(
                    (signed + feather / 2.0) / max(1.0, feather),
                    0.0,
                    1.0,
                )
                side_alpha_full = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
                transition_full = (
                    front_valid
                    & side_valid
                    & (np.abs(signed) <= feather / 2.0)
                )
            else:
                side_alpha_full = np.zeros((height, width), dtype=np.float32)
                transition_full = np.zeros((height, width), dtype=bool)

            visible = visible_full[:, crop_x0:crop_x1].copy()
            suppressed = suppressed_full[:, crop_x0:crop_x1].copy()
            transition = transition_full[:, crop_x0:crop_x1].copy()
            side_alpha = side_alpha_full[:, crop_x0:crop_x1].copy()
            full_columns = np.arange(crop_x0, crop_x1, dtype=np.int32)
            source_x = (
                full_columns - shift
                if candidate.side_position == "left"
                else full_columns + shift
            )
            source_columns_valid = (source_x >= 0) & (source_x < width)
            valid_column_indexes = np.flatnonzero(source_columns_valid)
            if valid_column_indexes.size:
                destination_x0 = int(valid_column_indexes[0])
                destination_x1 = int(valid_column_indexes[-1]) + 1
                source_x0 = int(source_x[destination_x0])
                copy_width = destination_x1 - destination_x0
            else:
                source_x0 = 0
                destination_x0 = 0
                copy_width = 0
            transition_columns = np.flatnonzero(np.any(transition, axis=0))
            if transition_columns.size:
                transition_x0 = int(transition_columns[0])
                transition_x1 = int(transition_columns[-1]) + 1
            else:
                transition_x0 = 0
                transition_x1 = 0
            side_visible_total |= visible
            side_suppressed_total |= suppressed
            runtime_pairs.append(
                LayoutPairRuntimePlan(
                    side_camera=candidate.side_camera,
                    source_offset=(
                        -shift if candidate.side_position == "left" else shift
                    ),
                    source_x0=source_x0,
                    destination_x0=destination_x0,
                    copy_width=copy_width,
                    visible_u8=_immutable(visible.astype(np.uint8) * 255),
                    transition_x0=transition_x0,
                    transition_x1=transition_x1,
                    transition_u8=_immutable(
                        transition[:, transition_x0:transition_x1].astype(np.uint8)
                        * 255
                    ),
                    side_alpha=_immutable(
                        side_alpha[:, transition_x0:transition_x1].copy()
                    ),
                )
            )

        front_valid_crop = front_valid[:, crop_x0:crop_x1].copy()
        front_preserved = front_valid_crop & ~side_visible_total
        return FrontPriorityLayoutRuntimePlan(
            canvas_shape=(height, width),
            crop=(crop_x0, crop_x1),
            layout_id=params.layout_id,
            front_valid=_immutable(front_valid_crop),
            front_valid_u8=_immutable(front_valid_crop.astype(np.uint8) * 255),
            pair_plans=tuple(runtime_pairs),
            side_visible=_immutable(side_visible_total),
            front_preserved=_immutable(front_preserved),
            side_suppressed=_immutable(side_suppressed_total),
            boundary_count=len(runtime_pairs),
        )

    def _render_with_runtime_plan(
        self,
        warped_images: dict[str, np.ndarray],
        params: LayoutPreviewParams,
        plan: FrontPriorityLayoutRuntimePlan,
        *,
        main_camera: str,
        draw_label: bool,
    ) -> LayoutPreviewResult:
        if draw_label:
            raise ValueError("Runtime layout plans support label-free rendering only.")
        if main_camera not in warped_images:
            raise ValueError("warped_images must include the front camera.")
        front = warped_images[main_camera]
        height, width = front.shape[:2]
        if (height, width) != plan.canvas_shape or params.layout_id != plan.layout_id:
            raise ValueError("Runtime layout plan does not match canvas or parameters.")
        crop_x0, crop_x1 = plan.crop
        front_crop = front[:, crop_x0:crop_x1]
        output = np.zeros_like(front_crop)
        cv2.copyTo(front_crop, plan.front_valid_u8, output)

        for pair in plan.pair_plans:
            side = warped_images.get(pair.side_camera)
            if side is None or side.shape != front.shape or side.dtype != front.dtype:
                raise ValueError("Runtime layout plan side-camera input changed shape or dtype.")
            if pair.copy_width:
                destination_x1 = pair.destination_x0 + pair.copy_width
                source_x1 = pair.source_x0 + pair.copy_width
                cv2.copyTo(
                    side[:, pair.source_x0:source_x1],
                    pair.visible_u8[:, pair.destination_x0:destination_x1],
                    output[:, pair.destination_x0:destination_x1],
                )
            if pair.transition_x1 > pair.transition_x0:
                transition_x0 = pair.transition_x0
                transition_x1 = pair.transition_x1
                source_x0 = crop_x0 + transition_x0 + pair.source_offset
                source_x1 = source_x0 + transition_x1 - transition_x0
                side_roi = side[:, source_x0:source_x1]
                front_roi = front_crop[:, transition_x0:transition_x1]
                alpha = pair.side_alpha[:, :, None]
                blended = (
                    side_roi.astype(np.float32) * alpha
                    + front_roi.astype(np.float32)
                    * (np.float32(1.0) - alpha)
                )
                cv2.copyTo(
                    np.clip(blended, 0, 255).astype(np.uint8),
                    pair.transition_u8,
                    output[:, transition_x0:transition_x1],
                )

        metrics = self._metrics(
            output,
            plan.front_valid,
            plan.side_visible,
            plan.front_preserved,
            plan.side_suppressed,
            width,
            crop_x0,
            crop_x1,
            plan.boundary_count,
        )
        return LayoutPreviewResult(
            layout_id=params.layout_id,
            image=output,
            side_visible_mask=plan.side_visible,
            front_preserved_mask=plan.front_preserved,
            side_suppressed_mask=plan.side_suppressed,
            metrics=metrics,
            crop=plan.crop,
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


def _immutable(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array
