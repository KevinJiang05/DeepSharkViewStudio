"""Callable runner for static pairwise seam candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .candidate import SavedSeamCandidate, SeamCandidateStore
from .cost import SeamCostBuilder, SeamCostParams
from .dp import DPSeamParams, VerticalDPSeamFinder
from .preview import SeamPreviewRenderer


DEFAULT_FEATHER_WIDTHS = [0, 16, 24, 32]


@dataclass(frozen=True)
class SeamCandidateRunResult:
    output_dir: Path
    pair_results: list[SavedSeamCandidate]
    skipped_pairs: list[dict[str, Any]]


def generate_pairwise_seam_candidates(
    warped_images: dict[str, np.ndarray],
    overlaps: list[dict],
    output_root: Path,
    profile_id: str = "triple_front_panorama",
    feather_widths: list[int] | None = None,
) -> SeamCandidateRunResult:
    """Generate report-only seam candidates from already-warped images."""
    feather_widths = list(feather_widths or DEFAULT_FEATHER_WIDTHS)
    if not feather_widths:
        feather_widths = list(DEFAULT_FEATHER_WIDTHS)
    if 0 not in feather_widths:
        feather_widths = [0, *feather_widths]
    recommended = 24 if 24 in feather_widths else int(feather_widths[-1])

    cost_params = SeamCostParams()
    dp_params = DPSeamParams()
    cost_builder = SeamCostBuilder()
    seam_finder = VerticalDPSeamFinder()
    renderer = SeamPreviewRenderer()
    store = SeamCandidateStore()
    run_directory = store.create_run_directory(output_root)

    pair_results: list[SavedSeamCandidate] = []
    skipped_pairs: list[dict[str, Any]] = []
    for overlap in overlaps:
        cameras = [str(camera) for camera in overlap.get("cameras", [])]
        pair_id = str(overlap.get("name") or "_".join(cameras))
        if len(cameras) != 2:
            skipped_pairs.append({
                "pair_id": pair_id,
                "reason": "overlap must declare exactly two cameras",
            })
            continue
        left_camera, right_camera = cameras
        image_a = warped_images.get(left_camera)
        image_b = warped_images.get(right_camera)
        if image_a is None or image_b is None:
            skipped_pairs.append({
                "pair_id": pair_id,
                "reason": "missing warped image",
                "cameras": cameras,
            })
            continue
        x_values = overlap.get("x_range", [])
        if len(x_values) != 2:
            skipped_pairs.append({
                "pair_id": pair_id,
                "reason": "missing x_range",
            })
            continue

        cost_result = cost_builder.build(
            image_a,
            image_b,
            (int(round(float(x_values[0]))), int(round(float(x_values[1])))),
            cost_params,
        )
        seam_path = seam_finder.find(
            cost_result.cost,
            cost_result.valid_mask,
            dp_params,
            x_offset=cost_result.x_range[0],
        )
        preview_files = renderer.render_pair(
            run_directory / pair_id,
            image_a,
            image_b,
            cost_result,
            seam_path,
            feather_widths,
        )
        saved = store.save_pair_candidate(
            run_directory=run_directory,
            pair_id=pair_id,
            profile_id=profile_id,
            x_range=cost_result.x_range,
            seam_path=seam_path,
            feather_widths=feather_widths,
            recommended_feather_width=recommended,
            cost_params=cost_params,
            dp_params=dp_params,
            preview_files=preview_files,
            extra_report={
                "cameras": cameras,
                "valid_pixel_count": int(np.count_nonzero(cost_result.valid_mask)),
                "overlap": overlap,
            },
        )
        pair_results.append(saved)

    return SeamCandidateRunResult(
        output_dir=run_directory,
        pair_results=pair_results,
        skipped_pairs=skipped_pairs,
    )
