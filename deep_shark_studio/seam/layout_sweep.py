"""Batch front-priority layout preview sweep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision, save_yaml
from deep_shark_studio.stitcher import save_image

from .layout_preview import (
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
    LayoutPreviewResult,
)


@dataclass(frozen=True)
class LayoutSweepResult:
    output_dir: Path
    report_path: Path
    candidate_count: int
    contact_sheets: dict[str, Path]
    recommended_for_manual_review: list[dict[str, Any]]


def load_layout_candidates_from_run(candidate_run_dir: str | Path) -> list[LayoutPairCandidate]:
    """Load V2 front-priority candidate paths from a candidate run directory."""
    run_dir = Path(candidate_run_dir)
    candidates: list[LayoutPairCandidate] = []
    for candidate_path in sorted(run_dir.glob("*/candidate.yaml")):
        data = yaml.safe_load(candidate_path.read_text(encoding="utf-8")) or {}
        pair_role = data.get("pair_role", {})
        side_camera = str(pair_role.get("side_camera", ""))
        side_position = str(pair_role.get("side_position", ""))
        if side_camera not in {"front_left", "front_right"}:
            continue
        if side_position not in {"left", "right"}:
            continue
        seam_points = [
            (int(point[0]), int(point[1]))
            for point in data.get("seam_points", [])
            if isinstance(point, list) and len(point) >= 2
        ]
        boundary_type = str(data.get("recommended_boundary_type") or "candidate")
        candidates.append(
            LayoutPairCandidate(
                pair_id=str(data.get("pair_id") or candidate_path.parent.name),
                side_camera=side_camera,
                side_position=side_position,
                boundary_type=boundary_type,
                seam_points=seam_points,
            )
        )
    return candidates


def run_front_priority_layout_sweep(
    warped_images: dict[str, np.ndarray],
    candidates: list[LayoutPairCandidate],
    output_root: Path,
    profile_id: str = "triple_front_panorama",
    side_shifts_px: list[int] | None = None,
    output_widths_px: list[int] | None = None,
    side_visible_fractions: list[float] | None = None,
    feather_widths: list[int] | None = None,
    source_note: str | None = None,
) -> LayoutSweepResult:
    """Generate report-only layout candidates from already-warped images."""
    before = file_revision(CONFIG_DIR / "calibration.yaml")
    side_shifts_px = list(side_shifts_px or [0, 40, 80, 120])
    output_widths_px = list(output_widths_px or [2200, 2000, 1900, 1800])
    side_visible_fractions = list(side_visible_fractions or [0.20, 0.25, 0.30])
    feather_widths = [int(value) for value in (feather_widths or [0, 16, 24]) if int(value) <= 24]
    if 0 not in feather_widths:
        feather_widths = [0, *feather_widths]
    directory = _create_layout_directory(output_root)
    candidates_dir = directory / "candidates"
    masks_dir = directory / "masks"
    contact_dir = directory / "contact_sheets"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    renderer = FrontPriorityLayoutPreviewRenderer()
    records: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    mask_keys_written: set[str] = set()
    for shift in side_shifts_px:
        for crop_width in output_widths_px:
            for side_fraction in side_visible_fractions:
                for feather in feather_widths:
                    params = LayoutPreviewParams(
                        side_shift_px=int(shift),
                        output_width_px=int(crop_width),
                        side_visible_fraction=float(side_fraction),
                        feather_width_px=int(feather),
                    )
                    result = renderer.render(warped_images, candidates, params)
                    filename = f"{result.layout_id}.png"
                    preview_path = candidates_dir / filename
                    save_image(preview_path, result.image)
                    preview_paths.append(preview_path)
                    mask_key = _mask_key(params)
                    mask_files: dict[str, str] = {}
                    if mask_key not in mask_keys_written:
                        mask_files = _save_masks(masks_dir, mask_key, result)
                        mask_keys_written.add(mask_key)
                    records.append(
                        {
                            "layout_id": result.layout_id,
                            "preview_file": preview_path.relative_to(directory).as_posix(),
                            "mask_files": mask_files,
                            "params": params.to_dict(),
                            "crop": {
                                "crop_x0": int(result.crop[0]),
                                "crop_x1": int(result.crop[1]),
                            },
                            "metrics": result.metrics,
                            "score_for_sorting_only": _score(result.metrics),
                        }
                    )

    recommended = _recommended(records)
    contact_sheets = {
        "all_candidates_contact_sheet": contact_dir / "all_candidates_contact_sheet.png",
        "best_by_width_contact_sheet": contact_dir / "best_by_width_contact_sheet.png",
        "shift_comparison_contact_sheet": contact_dir / "shift_comparison_contact_sheet.png",
        "crop_comparison_contact_sheet": contact_dir / "crop_comparison_contact_sheet.png",
    }
    _write_contact_sheet(contact_sheets["all_candidates_contact_sheet"], records, directory, max_items=144)
    _write_contact_sheet(contact_sheets["best_by_width_contact_sheet"], _best_by_width(records), directory, max_items=24)
    _write_contact_sheet(
        contact_sheets["shift_comparison_contact_sheet"],
        _filter_records(records, output_width_px=1900, side_visible_fraction=0.25, feather_width_px=24),
        directory,
        max_items=24,
    )
    _write_contact_sheet(
        contact_sheets["crop_comparison_contact_sheet"],
        _filter_records(records, side_shift_px=80, side_visible_fraction=0.25, feather_width_px=24),
        directory,
        max_items=24,
    )

    report = {
        "schema_version": 1,
        "profile_id": profile_id,
        "mode": "front_priority_layout_sweep",
        "source_note": source_note or "",
        "candidate_count": len(records),
        "input_candidates": [
            {
                "pair_id": candidate.pair_id,
                "side_camera": candidate.side_camera,
                "side_position": candidate.side_position,
                "boundary_type": candidate.boundary_type,
                "seam_point_count": len(candidate.seam_points),
            }
            for candidate in candidates
        ],
        "sweep_params": {
            "side_shifts_px": [int(value) for value in side_shifts_px],
            "output_widths_px": [int(value) for value in output_widths_px],
            "side_visible_fractions": [float(value) for value in side_visible_fractions],
            "feather_widths": [int(value) for value in feather_widths],
        },
        "recommended_for_manual_review": recommended,
        "contact_sheets": {
            key: path.relative_to(directory).as_posix()
            for key, path in contact_sheets.items()
        },
        "candidates": records,
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
    }
    report_path = directory / "layout_sweep_report.yaml"
    save_yaml(report_path, report)
    after = file_revision(CONFIG_DIR / "calibration.yaml")
    if after != before:
        raise RuntimeError("Formal calibration.yaml changed during layout sweep.")
    return LayoutSweepResult(
        output_dir=directory,
        report_path=report_path,
        candidate_count=len(records),
        contact_sheets=contact_sheets,
        recommended_for_manual_review=recommended,
    )


def _create_layout_directory(output_root: Path) -> Path:
    root = Path(output_root)
    directory = root if "layout_sweep" in root.name else root / "layout_sweep"
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


def _save_masks(
    masks_dir: Path,
    mask_key: str,
    result: LayoutPreviewResult,
) -> dict[str, str]:
    files = {
        "side_visible_clamped_mask": masks_dir / f"{mask_key}_side_visible_clamped_mask.png",
        "front_preserved_mask": masks_dir / f"{mask_key}_front_preserved_mask.png",
        "side_suppressed_by_clamp_mask": masks_dir / f"{mask_key}_side_suppressed_by_clamp_mask.png",
    }
    save_image(files["side_visible_clamped_mask"], _mask_image(result.side_visible_mask))
    save_image(files["front_preserved_mask"], _mask_image(result.front_preserved_mask))
    save_image(files["side_suppressed_by_clamp_mask"], _mask_image(result.side_suppressed_mask))
    return {key: path.relative_to(masks_dir.parent).as_posix() for key, path in files.items()}


def _mask_key(params: LayoutPreviewParams) -> str:
    side_value = int(round(params.side_visible_fraction * 100))
    return (
        f"shift_{params.side_shift_px:03d}"
        f"_crop_{params.output_width_px}"
        f"_side_{side_value:03d}"
    )


def _mask_image(mask: np.ndarray) -> np.ndarray:
    return np.dstack([mask.astype(np.uint8) * 255] * 3)


def _score(metrics: dict[str, Any]) -> float:
    side_visible = float(metrics.get("side_visible_ratio", 0.0))
    side_balance_bonus = max(0.0, 0.12 - abs(side_visible - 0.12))
    return float(
        2.0 * float(metrics.get("front_preserved_ratio", 0.0))
        - 2.0 * float(metrics.get("side_invasion_ratio", 0.0))
        - 1.5 * float(metrics.get("black_pixel_ratio", 0.0))
        - 1.0 * float(metrics.get("duplicate_risk_proxy", 0.0))
        + side_balance_bonus
    )


def _recommended(records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    sorted_records = sorted(records, key=lambda item: float(item["score_for_sorting_only"]), reverse=True)
    recommended: list[dict[str, Any]] = []
    for record in sorted_records[:limit]:
        metrics = record["metrics"]
        recommended.append(
            {
                "layout_id": record["layout_id"],
                "preview_file": record["preview_file"],
                "score_for_sorting_only": record["score_for_sorting_only"],
                "reason": (
                    f"front_preserved={metrics['front_preserved_ratio']:.3f}, "
                    f"side_invasion={metrics['side_invasion_ratio']:.3f}, "
                    f"black={metrics['black_pixel_ratio']:.3f}, "
                    f"duplicate_proxy={metrics['duplicate_risk_proxy']:.3f}"
                ),
            }
        )
    return recommended


def _best_by_width(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["params"]["output_width_px"]), []).append(record)
    selected: list[dict[str, Any]] = []
    for width in sorted(grouped):
        selected.extend(
            sorted(grouped[width], key=lambda item: float(item["score_for_sorting_only"]), reverse=True)[:4]
        )
    return selected


def _filter_records(records: list[dict[str, Any]], **criteria: Any) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        params = record["params"]
        ok = True
        for key, value in criteria.items():
            current = params.get(key)
            if isinstance(value, float):
                ok = ok and abs(float(current) - value) < 1e-6
            else:
                ok = ok and current == value
        if ok:
            selected.append(record)
    return sorted(selected, key=lambda item: item["layout_id"])


def _write_contact_sheet(
    path: Path,
    records: list[dict[str, Any]],
    root: Path,
    max_items: int,
) -> None:
    records = records[:max_items]
    thumb_w, thumb_h = 320, 116
    label_h = 42
    columns = 4 if len(records) <= 24 else 6
    rows = max(1, int(np.ceil(max(1, len(records)) / columns)))
    sheet = np.zeros((rows * (thumb_h + label_h), columns * thumb_w, 3), dtype=np.uint8)
    for index, record in enumerate(records):
        image_path = root / record["preview_file"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        y0 = row * (thumb_h + label_h)
        x0 = col * thumb_w
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
        label = record["layout_id"]
        metrics = record["metrics"]
        metric_label = (
            f"fp={metrics['front_preserved_ratio']:.2f} "
            f"si={metrics['side_invasion_ratio']:.2f} "
            f"blk={metrics['black_pixel_ratio']:.2f}"
        )
        cv2.putText(sheet, label[:44], (x0 + 4, y0 + thumb_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, metric_label, (x0 + 4, y0 + thumb_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
    save_image(path, sheet)
