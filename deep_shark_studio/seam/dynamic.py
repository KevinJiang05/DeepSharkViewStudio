"""Offline dynamic-boundary simulation for front-priority candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.config import save_yaml
from deep_shark_studio.stitcher import save_image

from .boundary import BoundarySearchParams, SideBoundaryCandidateFinder
from .cost import FrontPriorityCostParams, SeamCostBuilder, SeamCostParams
from .dp import DPSeamParams, VerticalDPSeamFinder
from .roles import resolve_front_priority_pair_role


@dataclass(frozen=True)
class DynamicBoundarySimulationResult:
    output_dir: Path
    timeline_path: Path
    boundary_switch_count: int
    average_boundary_shift_px: float
    max_boundary_shift_px_observed: float
    frames: list[dict[str, Any]]


def simulate_front_priority_dynamic_boundaries(
    warped_sequence: list[dict[str, np.ndarray]],
    overlaps: list[dict],
    output_root: Path,
    profile_id: str = "triple_front_panorama",
    update_interval_frames: int = 5,
    hysteresis_score_margin: float = 0.10,
    max_boundary_shift_px: int = 20,
    feather_widths: list[int] | None = None,
) -> DynamicBoundarySimulationResult:
    """Simulate boundary updates from warped frames without runtime integration."""
    _ = feather_widths or [0, 16, 24, 32]
    directory = Path(output_root) / "dynamic_simulation"
    directory.mkdir(parents=True, exist_ok=True)
    update_interval = max(1, int(update_interval_frames))
    max_shift = max(1, int(max_boundary_shift_px))

    cost_params = SeamCostParams(
        high_difference_weight=0.10,
        object_proximity_weight=0.10,
        front_priority=FrontPriorityCostParams(enabled=True)
    )
    dp_params = DPSeamParams()
    boundary_params = BoundarySearchParams()
    cost_builder = SeamCostBuilder()
    seam_finder = VerticalDPSeamFinder()
    boundary_finder = SideBoundaryCandidateFinder()

    current: dict[str, tuple[int | None, float]] = {}
    frames: list[dict[str, Any]] = []
    switch_count = 0
    shifts: list[float] = []

    for frame_index, warped_images in enumerate(warped_sequence):
        frame_record: dict[str, Any] = {"frame_index": frame_index, "pairs": {}}
        recompute = frame_index % update_interval == 0
        for overlap in overlaps:
            cameras = [str(camera) for camera in overlap.get("cameras", [])]
            pair_id = str(overlap.get("name") or "_".join(cameras))
            role = resolve_front_priority_pair_role(pair_id, cameras)
            if role is None:
                continue
            if role.side_camera not in warped_images or role.main_camera not in warped_images:
                continue
            if recompute or pair_id not in current:
                x_values = overlap.get("x_range", [])
                cost_result = cost_builder.build(
                    warped_images[role.side_camera if role.side_position == "left" else role.main_camera],
                    warped_images[role.main_camera if role.side_position == "left" else role.side_camera],
                    (int(round(float(x_values[0]))), int(round(float(x_values[1])))),
                    cost_params,
                    side_position=role.side_position,
                )
                seam_path = seam_finder.find(
                    cost_result.cost,
                    cost_result.valid_mask,
                    dp_params,
                    x_offset=cost_result.x_range[0],
                )
                result = boundary_finder.search(
                    cost_result,
                    seam_path,
                    role.side_position,
                    boundary_params,
                )
                proposed = result.recommended
                proposed_x = proposed.x_canvas if proposed else None
                proposed_score = proposed.score if proposed else float("inf")
                old_x, old_score = current.get(pair_id, (None, float("inf")))
                accepted_x = proposed_x
                accepted_score = proposed_score
                if old_x is not None and proposed_x is not None:
                    if proposed_score >= old_score - hysteresis_score_margin:
                        accepted_x = old_x
                        accepted_score = old_score
                    else:
                        shift = int(np.clip(proposed_x - old_x, -max_shift, max_shift))
                        accepted_x = old_x + shift
                        accepted_score = proposed_score
                    if accepted_x != old_x:
                        switch_count += 1
                        shifts.append(abs(float(accepted_x - old_x)))
                current[pair_id] = (accepted_x, accepted_score)
                frame_record["pairs"][pair_id] = {
                    "boundary_x": accepted_x,
                    "score": accepted_score,
                    "recomputed": True,
                    "proposed_x": proposed_x,
                    "proposed_score": proposed_score,
                }
            else:
                x, score = current.get(pair_id, (None, float("inf")))
                frame_record["pairs"][pair_id] = {
                    "boundary_x": x,
                    "score": score,
                    "recomputed": False,
                }
        frames.append(frame_record)

    timeline = {
        "schema_version": 1,
        "profile_id": profile_id,
        "update_interval_frames": update_interval,
        "hysteresis_score_margin": float(hysteresis_score_margin),
        "max_boundary_shift_px": max_shift,
        "boundary_switch_count": switch_count,
        "average_boundary_shift_px": float(np.mean(shifts)) if shifts else 0.0,
        "max_boundary_shift_px_observed": float(max(shifts)) if shifts else 0.0,
        "frames": frames,
    }
    timeline_path = directory / "timeline.yaml"
    save_yaml(timeline_path, timeline)
    _write_timeline_image(directory / "boundary_timeline.png", frames)
    return DynamicBoundarySimulationResult(
        output_dir=directory,
        timeline_path=timeline_path,
        boundary_switch_count=switch_count,
        average_boundary_shift_px=timeline["average_boundary_shift_px"],
        max_boundary_shift_px_observed=timeline["max_boundary_shift_px_observed"],
        frames=frames,
    )


def _write_timeline_image(path: Path, frames: list[dict[str, Any]]) -> None:
    width = max(120, len(frames) * 8)
    height = 80
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if not frames:
        save_image(path, image)
        return
    pair_ids = sorted({pair for frame in frames for pair in frame["pairs"]})
    colors = [(0, 255, 255), (0, 180, 255), (0, 255, 0)]
    for pair_index, pair_id in enumerate(pair_ids):
        xs = [
            frame["pairs"].get(pair_id, {}).get("boundary_x")
            for frame in frames
        ]
        valid = [x for x in xs if x is not None]
        if not valid:
            continue
        lo, hi = min(valid), max(valid)
        span = max(1, hi - lo)
        points = []
        for frame_index, value in enumerate(xs):
            if value is None:
                continue
            x = int(round(frame_index * (width - 1) / max(1, len(frames) - 1)))
            y = int(round(10 + pair_index * 24 + (1.0 - (value - lo) / span) * 20))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, colors[pair_index % len(colors)], 1)
    save_image(path, image)
