"""Read-only pairwise planar-alignment candidates for calibration snapshots."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .calibration import find_chessboard, spec_from_config
from .config import load_yaml
from .geometry_diagnostics import ALPHA_50, NO_FEATHER, compose_pair_only
from .stitcher import CameraCalibration, save_image
from .topology import (
    active_stitch_profile,
    seam_x,
    selected_topology_name,
    source_coordinate_diagnostics,
)


PAIRWISE_FORMAT = "DeepSharkPairwiseDiagnostic"
PAIRWISE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairDefinition:
    name: str
    left_camera: str
    right_camera: str
    overlap: dict[str, Any]
    seam_position: float


@dataclass(frozen=True)
class PairwiseCandidateResult:
    pair: PairDefinition
    snapshot_dir: Path
    method: str
    point_count: int
    inlier_count: int
    reprojection_error: float | None
    homography: np.ndarray | None
    left_points: np.ndarray
    right_points: np.ndarray
    left_visualization: np.ndarray
    right_visualization: np.ndarray
    alpha_overlay: np.ndarray | None
    hard_cut_overlay: np.ndarray | None
    warnings: tuple[str, ...]
    error: str

    def metadata(self) -> dict[str, Any]:
        return {
            "pair": self.pair.name,
            "cameras": [
                self.pair.left_camera,
                self.pair.right_camera,
            ],
            "snapshot": self.snapshot_dir.name,
            "method": self.method,
            "point_count": self.point_count,
            "inlier_count": self.inlier_count,
            "reprojection_error_px": self.reprojection_error,
            "homography_left_to_right": (
                self.homography.tolist()
                if self.homography is not None
                else None
            ),
            "left_points": self.left_points.tolist(),
            "right_points": self.right_points.tolist(),
            "overlap_x_range": deepcopy(
                self.pair.overlap.get("x_range", [])
            ),
            "seam": self.pair.overlap.get("seam", ""),
            "seam_x": self.pair.seam_position,
            "warnings": list(self.warnings),
            "error": self.error,
            "writes_formal_profile": False,
        }


def pair_definitions(calibration_config: dict[str, Any]) -> list[PairDefinition]:
    profile = active_stitch_profile(calibration_config)
    definitions: list[PairDefinition] = []
    for overlap in profile.get("overlaps", []):
        cameras = [str(key) for key in overlap.get("cameras", [])]
        if len(cameras) != 2:
            continue
        seam_name = str(overlap.get("seam", ""))
        definitions.append(
            PairDefinition(
                name=str(
                    overlap.get("name")
                    or f"{cameras[0]}_{cameras[1]}"
                ),
                left_camera=cameras[0],
                right_camera=cameras[1],
                overlap=deepcopy(overlap),
                seam_position=seam_x(
                    profile.get("stitch_points", {}),
                    seam_name,
                ),
            )
        )
    return definitions


def latest_snapshot_directory(
    snapshots_root: str | Path,
    topology_name: str,
) -> Path:
    root = Path(snapshots_root)
    if not root.exists():
        raise FileNotFoundError(f"Snapshot directory does not exist: {root}")
    for directory in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        reverse=True,
    ):
        metadata_path = directory / "metadata.yaml"
        if not metadata_path.exists():
            continue
        metadata = load_yaml(metadata_path)
        if str(metadata.get("topology", "")) == topology_name:
            return directory
    raise FileNotFoundError(
        f"No calibration snapshot found for topology '{topology_name}'."
    )


def load_snapshot_frames(snapshot_dir: str | Path) -> dict[str, np.ndarray]:
    directory = Path(snapshot_dir)
    metadata = load_yaml(directory / "metadata.yaml")
    frames: dict[str, np.ndarray] = {}
    raw_files = metadata.get("files", {}).get("raw", {})
    camera_order = metadata.get("camera_order", [])
    for key in camera_order:
        filename = str(raw_files.get(key) or f"raw_{key}.png")
        path = directory / Path(filename).name
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is not None:
            frames[str(key)] = image
    return frames


def _aruco_correspondences(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    if not hasattr(cv2, "aruco"):
        return "", np.empty((0, 2)), np.empty((0, 2))
    dictionary_names = (
        "DICT_4X4_50",
        "DICT_5X5_100",
        "DICT_6X6_250",
        "DICT_ARUCO_ORIGINAL",
    )
    best: tuple[str, np.ndarray, np.ndarray] | None = None
    for name in dictionary_names:
        dictionary_id = getattr(cv2.aruco, name, None)
        if dictionary_id is None:
            continue
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary)
            left_corners, left_ids, _ = detector.detectMarkers(left)
            right_corners, right_ids, _ = detector.detectMarkers(right)
        else:
            left_corners, left_ids, _ = cv2.aruco.detectMarkers(
                left,
                dictionary,
            )
            right_corners, right_ids, _ = cv2.aruco.detectMarkers(
                right,
                dictionary,
            )
        if left_ids is None or right_ids is None:
            continue
        left_by_id = {
            int(marker_id): corners.reshape(4, 2)
            for marker_id, corners in zip(left_ids.reshape(-1), left_corners)
        }
        right_by_id = {
            int(marker_id): corners.reshape(4, 2)
            for marker_id, corners in zip(right_ids.reshape(-1), right_corners)
        }
        common_ids = sorted(set(left_by_id) & set(right_by_id))
        if not common_ids:
            continue
        left_points = np.concatenate(
            [left_by_id[marker_id] for marker_id in common_ids]
        ).astype(np.float32)
        right_points = np.concatenate(
            [right_by_id[marker_id] for marker_id in common_ids]
        ).astype(np.float32)
        candidate = (
            f"aruco_or_charuco_markers:{name}",
            left_points,
            right_points,
        )
        if best is None or len(left_points) > len(best[1]):
            best = candidate
    if best is None:
        return "", np.empty((0, 2)), np.empty((0, 2))
    return best


def _homography_metrics(
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, float | None]:
    if len(left_points) < 4 or len(left_points) != len(right_points):
        return None, np.zeros((len(left_points),), dtype=bool), None
    homography, mask = cv2.findHomography(
        left_points.astype(np.float32),
        right_points.astype(np.float32),
        cv2.RANSAC,
        3.0,
    )
    if homography is None:
        return None, np.zeros((len(left_points),), dtype=bool), None
    inliers = (
        mask.reshape(-1).astype(bool)
        if mask is not None
        else np.ones((len(left_points),), dtype=bool)
    )
    projected = cv2.perspectiveTransform(
        left_points.reshape(1, -1, 2).astype(np.float32),
        homography,
    )[0]
    errors = np.linalg.norm(projected - right_points, axis=1)
    selected = errors[inliers] if np.any(inliers) else errors
    return homography, inliers, float(np.mean(selected))


def _chessboard_correspondences(
    left: np.ndarray,
    right: np.ndarray,
    calibration_config: dict[str, Any],
) -> tuple[str, np.ndarray, np.ndarray]:
    spec = spec_from_config(calibration_config)
    left_ok, left_corners = find_chessboard(left, spec)
    right_ok, right_corners = find_chessboard(right, spec)
    if not left_ok or not right_ok or left_corners is None or right_corners is None:
        return "", np.empty((0, 2)), np.empty((0, 2))
    left_points = left_corners.reshape(
        spec.inner_rows,
        spec.inner_columns,
        2,
    )
    right_grid = right_corners.reshape(
        spec.inner_rows,
        spec.inner_columns,
        2,
    )
    variants = {
        "direct": right_grid,
        "reverse_rows": right_grid[::-1, :, :],
        "reverse_columns": right_grid[:, ::-1, :],
        "reverse_both": right_grid[::-1, ::-1, :],
    }
    best: tuple[float, str, np.ndarray] | None = None
    flat_left = left_points.reshape(-1, 2).astype(np.float32)
    for orientation, grid in variants.items():
        flat_right = grid.reshape(-1, 2).astype(np.float32)
        homography, inliers, error = _homography_metrics(
            flat_left,
            flat_right,
        )
        if homography is None or error is None:
            continue
        score = error + (len(inliers) - int(np.count_nonzero(inliers))) * 0.1
        if best is None or score < best[0]:
            best = (score, orientation, flat_right)
    if best is None:
        return "", np.empty((0, 2)), np.empty((0, 2))
    return f"chessboard:{best[1]}", flat_left, best[2]


def _annotate_points(
    image: np.ndarray,
    points: np.ndarray,
    title: str,
) -> np.ndarray:
    annotated = image.copy()
    cv2.putText(
        annotated,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, (x, y) in enumerate(points):
        center = (int(round(float(x))), int(round(float(y))))
        cv2.circle(annotated, center, 6, (0, 255, 255), 2)
        if len(points) <= 24 or index % 8 == 0:
            cv2.putText(
                annotated,
                str(index + 1),
                (center[0] + 7, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return annotated


def _annotate_candidate_canvas(
    image: np.ndarray,
    pair: PairDefinition,
    title: str,
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    x_range = pair.overlap.get("x_range", [])
    if len(x_range) == 2:
        start, end = sorted(int(round(float(value))) for value in x_range)
        cv2.rectangle(
            annotated,
            (max(0, start), 0),
            (min(width - 1, end), height - 1),
            (0, 165, 255),
            2,
        )
    seam = int(round(pair.seam_position))
    if 0 <= seam < width:
        cv2.line(
            annotated,
            (seam, 0),
            (seam, height - 1),
            (0, 255, 255),
            2,
        )
    cv2.putText(
        annotated,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _candidate_overlays(
    left: np.ndarray,
    right: np.ndarray,
    homography: np.ndarray,
    profile: dict[str, Any],
    pair: PairDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    canvas = profile.get("canvas", {})
    width = int(canvas.get("width", 0))
    height = int(canvas.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Pairwise candidate requires a valid canvas size.")
    right_config = profile.get("cameras", {}).get(pair.right_camera)
    if not isinstance(right_config, dict):
        raise ValueError(
            f"Missing camera geometry for {pair.right_camera}."
        )
    right_matrix = CameraCalibration.from_config(
        pair.right_camera,
        right_config,
    ).matrix
    candidate_left_matrix = right_matrix @ homography
    left_warped = cv2.warpPerspective(
        left,
        candidate_left_matrix,
        (width, height),
    )
    right_warped = cv2.warpPerspective(
        right,
        right_matrix,
        (width, height),
    )
    alpha = compose_pair_only(
        left_warped,
        right_warped,
        ALPHA_50,
        pair.seam_position,
    )
    hard = compose_pair_only(
        left_warped,
        right_warped,
        NO_FEATHER,
        pair.seam_position,
    )
    camera_text = f"{pair.left_camera} -> {pair.right_camera}"
    return (
        _annotate_candidate_canvas(
            alpha,
            pair,
            f"Candidate 50% alpha: {camera_text}",
        ),
        _annotate_candidate_canvas(
            hard,
            pair,
            f"Candidate hard cut: {camera_text}",
        ),
    )


def estimate_pairwise_candidate(
    snapshot_dir: str | Path,
    calibration_config: dict[str, Any],
    pair: PairDefinition,
    manual_left_points: list[list[float]] | np.ndarray | None = None,
    manual_right_points: list[list[float]] | np.ndarray | None = None,
) -> PairwiseCandidateResult:
    directory = Path(snapshot_dir)
    frames = load_snapshot_frames(directory)
    left = frames.get(pair.left_camera)
    right = frames.get(pair.right_camera)
    empty = np.empty((0, 2), dtype=np.float32)
    if left is None or right is None:
        missing = [
            key
            for key, frame in (
                (pair.left_camera, left),
                (pair.right_camera, right),
            )
            if frame is None
        ]
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        return PairwiseCandidateResult(
            pair=pair,
            snapshot_dir=directory,
            method="unavailable",
            point_count=0,
            inlier_count=0,
            reprojection_error=None,
            homography=None,
            left_points=empty,
            right_points=empty,
            left_visualization=blank.copy(),
            right_visualization=blank.copy(),
            alpha_overlay=None,
            hard_cut_overlay=None,
            warnings=(),
            error=f"Snapshot is missing raw frames: {', '.join(missing)}",
        )

    if manual_left_points is not None or manual_right_points is not None:
        if manual_left_points is None or manual_right_points is None:
            raise ValueError(
                "Manual diagnostics require both left and right point sets."
            )
        left_points = np.asarray(manual_left_points, dtype=np.float32).reshape(-1, 2)
        right_points = np.asarray(manual_right_points, dtype=np.float32).reshape(-1, 2)
        method = "manual_correspondences"
    else:
        method, left_points, right_points = _aruco_correspondences(left, right)
        if len(left_points) < 4:
            method, left_points, right_points = _chessboard_correspondences(
                left,
                right,
                calibration_config,
            )

    profile = active_stitch_profile(calibration_config)
    warnings: list[str] = []
    for key, frame in (
        (pair.left_camera, left),
        (pair.right_camera, right),
    ):
        height, width = frame.shape[:2]
        warnings.extend(
            source_coordinate_diagnostics(
                profile,
                key,
                (width, height),
            )["warnings"]
        )

    error = ""
    homography: np.ndarray | None = None
    inliers = np.zeros((len(left_points),), dtype=bool)
    reprojection_error: float | None = None
    alpha: np.ndarray | None = None
    hard: np.ndarray | None = None
    if len(left_points) < 4 or len(left_points) != len(right_points):
        error = (
            "No common automatic feature set with at least four points was "
            "found. Use manual correspondences for this pair."
        )
    else:
        homography, inliers, reprojection_error = _homography_metrics(
            left_points,
            right_points,
        )
        if homography is None:
            error = "The selected correspondences do not define a homography."
        else:
            try:
                alpha, hard = _candidate_overlays(
                    left,
                    right,
                    homography,
                    profile,
                    pair,
                )
            except Exception as exc:
                error = f"Candidate overlay failed: {exc}"

    return PairwiseCandidateResult(
        pair=pair,
        snapshot_dir=directory,
        method=method or "automatic_failed",
        point_count=int(len(left_points)),
        inlier_count=int(np.count_nonzero(inliers)),
        reprojection_error=reprojection_error,
        homography=homography,
        left_points=left_points,
        right_points=right_points,
        left_visualization=_annotate_points(
            left,
            left_points,
            f"{pair.left_camera}: {method or 'no common features'}",
        ),
        right_visualization=_annotate_points(
            right,
            right_points,
            f"{pair.right_camera}: {method or 'no common features'}",
        ),
        alpha_overlay=alpha,
        hard_cut_overlay=hard,
        warnings=tuple(dict.fromkeys(warnings)),
        error=error,
    )


def run_automatic_pairwise_candidates(
    snapshots_root: str | Path,
    calibration_config: dict[str, Any],
) -> list[PairwiseCandidateResult]:
    snapshot_dir = latest_snapshot_directory(
        snapshots_root,
        selected_topology_name(calibration_config),
    )
    return [
        estimate_pairwise_candidate(
            snapshot_dir,
            calibration_config,
            pair,
        )
        for pair in pair_definitions(calibration_config)
    ]


def save_pairwise_results(
    output_root: str | Path,
    results: list[PairwiseCandidateResult],
    captured_at: datetime | None = None,
) -> Path:
    if not results:
        raise ValueError("No pairwise diagnostic results to save.")
    captured_at = captured_at or datetime.now().astimezone()
    name = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = Path(output_root) / name
    suffix = 1
    while output_dir.exists():
        output_dir = Path(output_root) / f"{name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "format": PAIRWISE_FORMAT,
        "schema_version": PAIRWISE_SCHEMA_VERSION,
        "created_at": captured_at.isoformat(timespec="milliseconds"),
        "snapshot": results[0].snapshot_dir.name,
        "writes_formal_profile": False,
        "results": [],
    }
    for result in results:
        pair_dir = output_dir / result.pair.name
        pair_dir.mkdir(parents=True, exist_ok=False)
        save_image(pair_dir / "left_points.png", result.left_visualization)
        save_image(pair_dir / "right_points.png", result.right_visualization)
        if result.alpha_overlay is not None:
            save_image(pair_dir / "candidate_alpha_50.png", result.alpha_overlay)
        if result.hard_cut_overlay is not None:
            save_image(pair_dir / "candidate_hard_cut.png", result.hard_cut_overlay)
        item = result.metadata()
        item["files"] = {
            "left_points": f"{result.pair.name}/left_points.png",
            "right_points": f"{result.pair.name}/right_points.png",
            "alpha_overlay": (
                f"{result.pair.name}/candidate_alpha_50.png"
                if result.alpha_overlay is not None
                else None
            ),
            "hard_cut_overlay": (
                f"{result.pair.name}/candidate_hard_cut.png"
                if result.hard_cut_overlay is not None
                else None
            ),
        }
        metadata["results"].append(item)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir
