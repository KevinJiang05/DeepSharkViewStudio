"""Batch vertical safe-band sweep for front-priority layout candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deep_shark_studio.config import CONFIG_DIR, file_revision, save_yaml
from deep_shark_studio.stitcher import save_image

from .layout_preview import LayoutPairCandidate, LayoutPreviewParams
from .layout_sweep import load_layout_candidates_from_run
from .projection_readiness_audit import write_projection_readiness_audit
from .vertical_safety import VerticalSafetyParams, VerticalSafetyRenderer, VerticalSafetyResult


@dataclass(frozen=True)
class VerticalSafetySweepResult:
    output_dir: Path
    report_path: Path
    candidate_count: int
    contact_sheets: dict[str, Path]
    recommended_for_manual_review: list[dict[str, Any]]
    projection_audit_path: Path


def run_vertical_safety_sweep_from_warped_images(
    warped_images: dict[str, np.ndarray],
    pair_candidates: list[LayoutPairCandidate],
    layout_params: LayoutPreviewParams,
    output_root: Path,
    profile_id: str = "triple_front_panorama",
    output_heights_px: list[int] | None = None,
    vertical_safe_ratios: list[float] | None = None,
    side_vertical_fade_px: list[int] | None = None,
    session_dir: Path | None = None,
    source_note: str | None = None,
) -> VerticalSafetySweepResult:
    before = file_revision(CONFIG_DIR / "calibration.yaml")
    output_heights_px = list(output_heights_px or [700, 660, 640, 600, 560])
    vertical_safe_ratios = list(vertical_safe_ratios or [1.00, 0.92, 0.86, 0.80])
    side_vertical_fade_px = list(side_vertical_fade_px or [0, 24, 48, 72])
    directory = _create_vertical_directory(output_root)
    candidates_dir = directory / "candidates"
    masks_dir = directory / "masks"
    contacts_dir = directory / "contact_sheets"
    for path in (candidates_dir, masks_dir, contacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    projection_audit_path = write_projection_readiness_audit(
        directory,
        session_dir=session_dir,
    )
    renderer = VerticalSafetyRenderer()
    records: list[dict[str, Any]] = []
    for height in output_heights_px:
        for safe_ratio in vertical_safe_ratios:
            for fade in side_vertical_fade_px:
                params = VerticalSafetyParams(
                    output_height_px=int(height),
                    vertical_safe_ratio=float(safe_ratio),
                    side_vertical_fade_px=int(fade),
                )
                result = renderer.render(
                    warped_images,
                    pair_candidates,
                    layout_params,
                    params,
                )
                preview_path = candidates_dir / f"{result.layout_id}_preview.png"
                save_image(preview_path, result.image)
                mask_files = _save_masks(masks_dir, result)
                records.append(
                    {
                        "layout_id": result.layout_id,
                        "preview_file": preview_path.relative_to(directory).as_posix(),
                        "mask_files": mask_files,
                        "params": {
                            "base_layout_id": layout_params.layout_id,
                            "output_width_px": int(layout_params.output_width_px),
                            "output_height_px": int(height),
                            "vertical_safe_ratio": float(safe_ratio),
                            "side_vertical_fade_px": int(fade),
                        },
                        "crop": {
                            "crop_y0": int(result.crop_y[0]),
                            "crop_y1": int(result.crop_y[1]),
                        },
                        "metrics": result.metrics,
                        "score_for_sorting_only": _score(result.metrics),
                    }
                )

    recommended = _recommended(records)
    contact_sheets = {
        "vertical_height_comparison_contact_sheet": contacts_dir / "vertical_height_comparison_contact_sheet.png",
        "vertical_fade_comparison_contact_sheet": contacts_dir / "vertical_fade_comparison_contact_sheet.png",
        "best_vertical_safety_contact_sheet": contacts_dir / "best_vertical_safety_contact_sheet.png",
    }
    _write_contact_sheet(
        contact_sheets["vertical_height_comparison_contact_sheet"],
        _height_representatives(records),
        directory,
    )
    _write_contact_sheet(
        contact_sheets["vertical_fade_comparison_contact_sheet"],
        _fade_representatives(records),
        directory,
    )
    _write_contact_sheet(
        contact_sheets["best_vertical_safety_contact_sheet"],
        sorted(records, key=lambda item: item["score_for_sorting_only"], reverse=True)[:24],
        directory,
    )

    report = {
        "schema_version": 1,
        "profile_id": profile_id,
        "mode": "vertical_safety_sweep",
        "source_note": source_note or "",
        "base_layout": layout_params.to_dict() | {"layout_id": layout_params.layout_id},
        "input_candidates": [
            {
                "pair_id": candidate.pair_id,
                "side_camera": candidate.side_camera,
                "side_position": candidate.side_position,
                "boundary_type": candidate.boundary_type,
                "seam_point_count": len(candidate.seam_points),
            }
            for candidate in pair_candidates
        ],
        "sweep_params": {
            "output_heights_px": [int(value) for value in output_heights_px],
            "vertical_safe_ratios": [float(value) for value in vertical_safe_ratios],
            "side_vertical_fade_px": [int(value) for value in side_vertical_fade_px],
        },
        "candidate_count": len(records),
        "recommended_for_manual_review": recommended,
        "contact_sheets": {
            key: path.relative_to(directory).as_posix()
            for key, path in contact_sheets.items()
        },
        "projection_readiness_audit": projection_audit_path.relative_to(directory).as_posix(),
        "projection_candidate_preview": {
            "generated": False,
            "reason": (
                "Not generated in this sweep because formal fisheye K/D is not complete for all active cameras "
                "and B-2 remap artifacts are report-only candidates, not a validated runtime projection."
            ),
        },
        "candidates": records,
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
    }
    report_path = directory / "vertical_safety_report.yaml"
    save_yaml(report_path, report)
    after = file_revision(CONFIG_DIR / "calibration.yaml")
    if after != before:
        raise RuntimeError("Formal calibration.yaml changed during vertical safety sweep.")
    return VerticalSafetySweepResult(
        output_dir=directory,
        report_path=report_path,
        candidate_count=len(records),
        contact_sheets=contact_sheets,
        recommended_for_manual_review=recommended,
        projection_audit_path=projection_audit_path,
    )


def run_vertical_safety_sweep(
    candidate_run_dir: Path,
    warped_images: dict[str, np.ndarray],
    baseline_layout_id: str,
    output_root: Path,
    profile_id: str = "triple_front_panorama",
    output_heights_px: list[int] | None = None,
    vertical_safe_ratios: list[float] | None = None,
    side_vertical_fade_px: list[int] | None = None,
    session_dir: Path | None = None,
) -> VerticalSafetySweepResult:
    pair_candidates = load_layout_candidates_from_run(candidate_run_dir)
    layout_params = _layout_params_from_id(baseline_layout_id)
    return run_vertical_safety_sweep_from_warped_images(
        warped_images=warped_images,
        pair_candidates=pair_candidates,
        layout_params=layout_params,
        output_root=output_root,
        profile_id=profile_id,
        output_heights_px=output_heights_px,
        vertical_safe_ratios=vertical_safe_ratios,
        side_vertical_fade_px=side_vertical_fade_px,
        session_dir=session_dir,
        source_note=f"Loaded candidate boundaries from {candidate_run_dir}",
    )


def _layout_params_from_id(layout_id: str) -> LayoutPreviewParams:
    parts = layout_id.split("_")
    try:
        shift = int(parts[1])
        crop = int(parts[3])
        side = int(parts[5]) / 100.0
        feather = int(parts[7])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid layout id: {layout_id}") from exc
    return LayoutPreviewParams(shift, crop, side, feather)


def _create_vertical_directory(output_root: Path) -> Path:
    root = Path(output_root)
    directory = root if root.name.startswith("vertical_safety") else root / "vertical_safety"
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        return directory
    suffix = 1
    while True:
        candidate = directory.with_name(f"{directory.name}_{suffix:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def _save_masks(masks_dir: Path, result: VerticalSafetyResult) -> dict[str, str]:
    files = {
        "vertical_side_weight_mask": masks_dir / f"{result.layout_id}_vertical_side_weight_mask.png",
        "side_visible_before_vertical_safety": masks_dir / f"{result.layout_id}_side_visible_before.png",
        "side_visible_after_vertical_safety": masks_dir / f"{result.layout_id}_side_visible_after.png",
        "side_suppressed_top_bottom_mask": masks_dir / f"{result.layout_id}_suppressed_top_bottom.png",
        "front_preserved_mask": masks_dir / f"{result.layout_id}_front_preserved_mask.png",
        "invalid_or_black_mask": masks_dir / f"{result.layout_id}_invalid_or_black_mask.png",
    }
    save_image(files["vertical_side_weight_mask"], _float_mask(result.vertical_side_weight_mask))
    save_image(files["side_visible_before_vertical_safety"], _bool_mask(result.side_visible_before))
    save_image(files["side_visible_after_vertical_safety"], _bool_mask(result.side_visible_after))
    save_image(files["side_suppressed_top_bottom_mask"], _bool_mask(result.side_suppressed_top_bottom))
    save_image(files["front_preserved_mask"], _bool_mask(result.front_preserved_mask))
    save_image(files["invalid_or_black_mask"], _bool_mask(result.invalid_or_black_mask))
    return {key: path.relative_to(masks_dir.parent).as_posix() for key, path in files.items()}


def _float_mask(mask: np.ndarray) -> np.ndarray:
    value = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([value] * 3)


def _bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.dstack([mask.astype(np.uint8) * 255] * 3)


def _score(metrics: dict[str, Any]) -> float:
    return float(
        2.0 * metrics["front_preserved_ratio"]
        - 1.2 * metrics["black_pixel_ratio"]
        - 1.5 * metrics["duplicate_risk_proxy_after"]
        + 0.8 * max(0.0, metrics["duplicate_risk_proxy_before"] - metrics["duplicate_risk_proxy_after"])
        - 0.6 * max(0.0, 0.86 - metrics["retained_area_ratio"])
    )


def _recommended(records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    selected = sorted(records, key=lambda item: item["score_for_sorting_only"], reverse=True)[:limit]
    output: list[dict[str, Any]] = []
    for record in selected:
        metrics = record["metrics"]
        output.append(
            {
                "layout_id": record["layout_id"],
                "preview_file": record["preview_file"],
                "score_for_sorting_only": record["score_for_sorting_only"],
                "reason": (
                    f"retained={metrics['retained_area_ratio']:.3f}, "
                    f"black={metrics['black_pixel_ratio']:.3f}, "
                    f"dup_before={metrics['duplicate_risk_proxy_before']:.3f}, "
                    f"dup_after={metrics['duplicate_risk_proxy_after']:.3f}"
                ),
            }
        )
    return output


def _height_representatives(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "h700_safe100_fade000",
        "h660_safe086_fade024",
        "h640_safe086_fade024",
        "h640_safe086_fade048",
        "h600_safe080_fade048",
        "h560_safe080_fade072",
    }
    return [record for record in records if record["layout_id"] in wanted]


def _fade_representatives(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["params"]["output_height_px"] == 640
        and abs(record["params"]["vertical_safe_ratio"] - 0.86) < 1e-6
    ]


def _write_contact_sheet(path: Path, records: list[dict[str, Any]], root: Path) -> None:
    records = records[:32]
    thumb_w, thumb_h = 360, 140
    label_h = 42
    columns = 3 if len(records) <= 12 else 4
    rows = max(1, int(np.ceil(max(1, len(records)) / columns)))
    sheet = np.zeros((rows * (thumb_h + label_h), columns * thumb_w, 3), dtype=np.uint8)
    for index, record in enumerate(records):
        image = cv2.imread(str(root / record["preview_file"]))
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        y0 = row * (thumb_h + label_h)
        x0 = col * thumb_w
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
        metrics = record["metrics"]
        cv2.putText(sheet, record["layout_id"], (x0 + 4, y0 + thumb_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            sheet,
            f"blk={metrics['black_pixel_ratio']:.2f} dup={metrics['duplicate_risk_proxy_after']:.2f} keep={metrics['retained_area_ratio']:.2f}",
            (x0 + 4, y0 + thumb_h + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    save_image(path, sheet)
