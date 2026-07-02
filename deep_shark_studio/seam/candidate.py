"""Human-readable storage for static seam candidates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from deep_shark_studio.config import CONFIG_DIR, file_revision, save_yaml

from .cost import SeamCostParams
from .dp import DPSeamParams, SeamPath


@dataclass(frozen=True)
class SavedSeamCandidate:
    pair_id: str
    directory: Path
    candidate_path: Path
    report_path: Path
    preview_files: dict[str, str]
    mean_cost: float
    p95_cost: float
    jaggedness: float


class SeamCandidateStore:
    """Save seam candidates without writing the formal calibration profile."""

    def __init__(self, formal_calibration_path: str | Path | None = None):
        self.formal_calibration_path = Path(
            formal_calibration_path or CONFIG_DIR / "calibration.yaml"
        )

    def create_run_directory(
        self,
        output_root: str | Path,
        created_at: datetime | None = None,
    ) -> Path:
        created_at = created_at or datetime.now().astimezone()
        session_id = created_at.strftime("%Y%m%d_%H%M%S")
        run_id = created_at.strftime("run_%H%M%S_%f")[:-3]
        root = Path(output_root)
        directory = root / session_id / run_id
        suffix = 1
        while directory.exists():
            directory = root / session_id / f"{run_id}_{suffix:02d}"
            suffix += 1
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def save_pair_candidate(
        self,
        run_directory: str | Path,
        pair_id: str,
        profile_id: str,
        x_range: tuple[int, int],
        seam_path: SeamPath,
        feather_widths: list[int],
        recommended_feather_width: int,
        cost_params: SeamCostParams,
        dp_params: DPSeamParams,
        preview_files: dict[str, str],
        created_at: datetime | None = None,
        extra_report: dict[str, Any] | None = None,
        mode: str = "pairwise",
        pair_role: dict[str, Any] | None = None,
        boundary_candidates: dict[str, Any] | None = None,
        quality: dict[str, Any] | None = None,
    ) -> SavedSeamCandidate:
        before = file_revision(self.formal_calibration_path)
        created_at = created_at or datetime.now().astimezone()
        pair_directory = Path(run_directory) / pair_id
        if (
            pair_directory.exists()
            and (
                (pair_directory / "candidate.yaml").exists()
                or (pair_directory / "report.yaml").exists()
            )
        ):
            suffix = 1
            base = pair_directory
            while pair_directory.exists():
                pair_directory = Path(f"{base}_{suffix:02d}")
                suffix += 1
        pair_directory.mkdir(parents=True, exist_ok=True)

        seam_points = [[int(x), int(y)] for x, y in seam_path.points_canvas]
        candidate = {
            "schema_version": 1,
            "profile_id": profile_id,
            "pair_id": pair_id,
            "mode": mode,
            "created_at": created_at.isoformat(timespec="milliseconds"),
            "x_range": [int(x_range[0]), int(x_range[1])],
            "coordinate_space": "panorama_canvas",
            "feather_width_candidates": [int(value) for value in feather_widths],
            "recommended_feather_width": int(recommended_feather_width),
            "seam_points": seam_points,
            "metrics": {
                "mean_cost": seam_path.mean_cost,
                "p95_cost": seam_path.p95_cost,
                "jaggedness": seam_path.jaggedness,
            },
            "cost_params": cost_params.to_dict(),
            "dp_params": dp_params.to_dict(),
            "preview_files": preview_files,
            "formal_profile_modified": False,
            "writes_calibration_yaml": False,
        }
        if pair_role:
            candidate["pair_role"] = pair_role
        if boundary_candidates:
            candidate["boundary_candidates"] = boundary_candidates
            candidate["recommended_boundary_type"] = boundary_candidates.get(
                "recommended_boundary_type"
            )
        if quality:
            candidate["quality"] = quality
        report = {
            "schema_version": 1,
            "profile_id": profile_id,
            "pair_id": pair_id,
            "mode": mode,
            "created_at": candidate["created_at"],
            "summary": {
                "seam_point_count": len(seam_points),
                "recommended_feather_width": int(recommended_feather_width),
                "mean_cost": seam_path.mean_cost,
                "p95_cost": seam_path.p95_cost,
                "jaggedness": seam_path.jaggedness,
            },
            "limitations": [
                "Static candidate only; not applied to live runtime.",
                "No depth model, optical flow, graph cut, CUDA, or deep learning is used.",
                "The black-pixel valid mask follows the current warp runtime and is a temporary compatibility heuristic.",
            ],
            "formal_profile_modified": False,
            "writes_calibration_yaml": False,
        }
        if pair_role:
            report["pair_role"] = pair_role
        if boundary_candidates:
            report["boundary_candidates"] = boundary_candidates
            report["recommended_boundary_type"] = boundary_candidates.get(
                "recommended_boundary_type"
            )
        if quality:
            report["quality"] = quality
        if extra_report:
            report["details"] = extra_report

        candidate_path = pair_directory / "candidate.yaml"
        report_path = pair_directory / "report.yaml"
        save_yaml(candidate_path, candidate)
        save_yaml(report_path, report)
        after = file_revision(self.formal_calibration_path)
        if after != before:
            raise RuntimeError("Formal calibration.yaml changed during seam candidate save.")
        return SavedSeamCandidate(
            pair_id=pair_id,
            directory=pair_directory,
            candidate_path=candidate_path,
            report_path=report_path,
            preview_files=preview_files,
            mean_cost=seam_path.mean_cost,
            p95_cost=seam_path.p95_cost,
            jaggedness=seam_path.jaggedness,
        )
