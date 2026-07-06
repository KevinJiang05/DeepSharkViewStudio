from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.stitch_runtime_modes import ProjectionSource
from deep_shark_studio.seam.layout_candidate_runtime import (
    LayoutCandidateRuntimeError,
    load_layout_candidate_for_runtime,
)
from deep_shark_studio.seam.near_field_compositor import render_near_field_from_warped


def _candidate_data(calibration_hash: str | None = None) -> dict:
    return {
        "schema_version": 2,
        "candidate_type": "front_priority_layout",
        "profile_id": "triple_front_panorama",
        "source": {
            "mode": "unit_test",
            "calibration_hash": calibration_hash
            or file_revision(CONFIG_DIR / "calibration.yaml"),
        },
        "camera_adjust": {
            "front_left": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
            "front": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
            "front_right": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
        },
        "left_pair": {
            "side_shift_px": 40,
            "side_visible_fraction": 0.30,
            "feather_width_px": 24,
        },
        "right_pair": {
            "side_shift_px": 40,
            "side_visible_fraction": 0.30,
            "feather_width_px": 24,
        },
        "output": {"width_px": 1800, "height_px": 700},
        "vertical_safety": {
            "enabled": False,
            "vertical_safe_ratio": 1.0,
            "side_vertical_fade_px": 0,
        },
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
    }


def _write_candidate(directory: Path, data: dict) -> Path:
    path = directory / "candidate.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_fisheye_source(directory: Path) -> Path:
    source_dir = directory / "fisheye_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = {}
    for camera in ("front_left", "front", "front_right"):
        intrinsics[camera] = {
            "status": "success",
            "model": "opencv_fisheye",
            "resolution": [32, 24],
            "camera_matrix": [
                [24.0, 0.0, 16.0],
                [0.0, 24.0, 12.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
            "rms_px": 0.1,
            "accepted_input_count": 8,
        }
    path = source_dir / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "topology": "triple_front_panorama",
                "resolution": [32, 24],
                "experimental": True,
                "apply_allowed": False,
                "intrinsics": intrinsics,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _warped_images(width: int = 2200, height: int = 700) -> dict[str, np.ndarray]:
    front = np.zeros((height, width, 3), dtype=np.uint8)
    front[:, 650:1550] = (0, 160, 0)
    left = np.zeros_like(front)
    left[:, :850] = (0, 0, 200)
    right = np.zeros_like(front)
    right[:, 1350:] = (200, 0, 0)
    return {"front_left": left, "front": front, "front_right": right}


class LayoutCandidateRuntimeTests(unittest.TestCase):
    def test_loader_reads_schema_v2_candidate_without_calibration_write(self) -> None:
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = _write_candidate(Path(temp), _candidate_data())
            candidate = load_layout_candidate_for_runtime(path)

            self.assertEqual(2, candidate.schema_version)
            self.assertEqual("triple_front_panorama", candidate.profile_id)
            self.assertEqual(40, candidate.left_pair.side_shift_px)
            self.assertEqual(40, candidate.right_pair.side_shift_px)
            self.assertIn("front", candidate.camera_adjust)
            self.assertTrue(candidate.pair_candidates)
            self.assertEqual(ProjectionSource.CURRENT_PERSPECTIVE, candidate.projection.source)
            self.assertEqual((), candidate.warnings)
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)

    def test_loader_rejects_v1_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = _candidate_data()
            data["schema_version"] = 1
            path = _write_candidate(Path(temp), data)

            with self.assertRaisesRegex(LayoutCandidateRuntimeError, "V1"):
                load_layout_candidate_for_runtime(path)

    def test_loader_rejects_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = _candidate_data()
            data.pop("left_pair")
            path = _write_candidate(Path(temp), data)

            with self.assertRaisesRegex(LayoutCandidateRuntimeError, "left_pair"):
                load_layout_candidate_for_runtime(path)

    def test_loader_reports_calibration_hash_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = _write_candidate(Path(temp), _candidate_data("not-current"))
            candidate = load_layout_candidate_for_runtime(path)

            self.assertTrue(candidate.warnings)

    def test_loader_reads_schema_v3_current_perspective_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = _candidate_data()
            data["schema_version"] = 3
            data["projection"] = {"source": "current_perspective"}
            candidate = load_layout_candidate_for_runtime(
                _write_candidate(Path(temp), data)
            )

            self.assertEqual(3, candidate.schema_version)
            self.assertEqual(ProjectionSource.CURRENT_PERSPECTIVE, candidate.projection.source)

    def test_loader_reads_schema_v3_fisheye_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fisheye_source = _write_fisheye_source(root)
            data = _candidate_data()
            data["schema_version"] = 3
            data["projection"] = {
                "source": "fisheye_rectilinear",
                "intrinsics_source_path": str(fisheye_source),
                "balance": 0.6,
                "fov_scale": 1.0,
                "projection_pipeline": "fisheye_rectilinear_then_template_perspective_warp",
            }
            candidate = load_layout_candidate_for_runtime(_write_candidate(root, data))

            self.assertEqual(
                ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                candidate.projection.source,
            )
            self.assertEqual(fisheye_source, candidate.projection.intrinsics_source_path)

    def test_loader_rejects_schema_v3_fisheye_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            data = _candidate_data()
            data["schema_version"] = 3
            data["projection"] = {"source": "fisheye_rectilinear"}
            path = _write_candidate(Path(temp), data)

            with self.assertRaisesRegex(LayoutCandidateRuntimeError, "intrinsics_source_path"):
                load_layout_candidate_for_runtime(path)

    def test_near_field_compositor_renders_canvas_from_mocked_warped_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = load_layout_candidate_for_runtime(
                _write_candidate(Path(temp), _candidate_data())
            )
            result = render_near_field_from_warped(_warped_images(), candidate)

            self.assertEqual((700, 1800), result.canvas.shape[:2])
            self.assertEqual("near_field", result.metrics["runtime_mode"])
            self.assertIn("crop_x", result.debug)


if __name__ == "__main__":
    unittest.main()
