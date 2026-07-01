"""Chessboard calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ChessboardSpec:
    """Physical chessboard description.

    OpenCV detects inner corners. If the lab board has 12 x 9 total squares,
    the default inner-corner pattern is 11 x 8.
    """

    total_columns: int = 12
    total_rows: int = 9
    square_size_mm: float = 25.0

    @property
    def inner_columns(self) -> int:
        return max(2, self.total_columns - 1)

    @property
    def inner_rows(self) -> int:
        return max(2, self.total_rows - 1)

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.inner_columns, self.inner_rows

    def object_points(self) -> np.ndarray:
        points = np.zeros((self.inner_columns * self.inner_rows, 3), np.float32)
        grid = np.mgrid[0 : self.inner_columns, 0 : self.inner_rows].T.reshape(-1, 2)
        points[:, :2] = grid * self.square_size_mm
        return points


def spec_from_config(config: dict) -> ChessboardSpec:
    board = config.get("calibration_board", {})
    return ChessboardSpec(
        total_columns=int(board.get("total_columns", 12)),
        total_rows=int(board.get("total_rows", 9)),
        square_size_mm=float(board.get("square_size_mm", 25.0)),
    )


def find_chessboard(image: np.ndarray, spec: ChessboardSpec) -> tuple[bool, np.ndarray | None]:
    """Find and refine chessboard inner corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, spec.pattern_size, flags)
    if not ok or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def draw_chessboard_detection(image: np.ndarray, spec: ChessboardSpec) -> tuple[np.ndarray, bool, int]:
    ok, corners = find_chessboard(image, spec)
    annotated = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(annotated, spec.pattern_size, corners, ok)
    return annotated, ok, 0 if corners is None else len(corners)


def calibrate_camera_from_images(images: list[np.ndarray], spec: ChessboardSpec) -> dict:
    """Estimate camera intrinsics from a list of chessboard images."""
    object_points = []
    image_points = []
    image_size: tuple[int, int] | None = None
    detections = []

    for index, image in enumerate(images):
        ok, corners = find_chessboard(image, spec)
        detections.append({
            "index": index,
            "detected": bool(ok),
            "corner_count": 0 if corners is None else int(len(corners)),
        })
        if not ok or corners is None:
            continue
        image_size = (image.shape[1], image.shape[0])
        object_points.append(spec.object_points())
        image_points.append(corners)

    if not object_points or image_size is None:
        raise ValueError("No chessboard corners were detected in the provided images.")

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    per_view_errors = reprojection_errors(object_points, image_points, rvecs, tvecs, camera_matrix, distortion)
    return {
        "rms": float(rms),
        "image_count": len(object_points),
        "input_image_count": len(images),
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "mean_reprojection_error": float(np.mean(per_view_errors)) if per_view_errors else 0.0,
        "per_view_reprojection_errors": per_view_errors,
        "detections": detections,
        "rotation_vectors": [r.reshape(-1).tolist() for r in rvecs],
        "translation_vectors": [t.reshape(-1).tolist() for t in tvecs],
    }


def reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    tvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]:
    """Compute per-view mean reprojection error in pixels."""
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        error = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
    return errors


def undistort_image(image: np.ndarray, intrinsics: dict) -> np.ndarray:
    """Undistort an image using saved camera intrinsics."""
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(intrinsics["distortion"], dtype=np.float64)
    height, width = image.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (width, height), 1, (width, height))
    return cv2.undistort(image, camera_matrix, distortion, None, new_matrix)


def write_calibration_report(path: str | Path, camera_key: str, result: dict, spec: ChessboardSpec) -> None:
    """Write a compact Markdown quality report for a camera calibration."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Calibration Report: {camera_key}",
        "",
        f"- Board total squares: {spec.total_columns} x {spec.total_rows}",
        f"- Board inner corners: {spec.inner_columns} x {spec.inner_rows}",
        f"- Square size: {spec.square_size_mm} mm",
        f"- Input images: {result.get('input_image_count', result.get('image_count', 0))}",
        f"- Usable images: {result.get('image_count', 0)}",
        f"- RMS: {result.get('rms', 0):.6f}",
        f"- Mean reprojection error: {result.get('mean_reprojection_error', 0):.6f} px",
        "",
        "## Per-View Reprojection Errors",
        "",
    ]
    errors = result.get("per_view_reprojection_errors", [])
    if errors:
        for index, error in enumerate(errors, start=1):
            lines.append(f"- View {index}: {error:.6f} px")
    else:
        lines.append("- No per-view errors available.")
    lines.extend(["", "## Detection Summary", ""])
    for item in result.get("detections", []):
        status = "OK" if item.get("detected") else "FAILED"
        lines.append(f"- Image {item.get('index')}: {status}, corners={item.get('corner_count', 0)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
