from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.projection.fisheye_projection import (
    EquirectangularParams,
    build_equirectangular_projection,
    build_rectilinear_projection,
)
from deep_shark_studio.projection.intrinsics_candidate import (
    FisheyeCameraModel,
    summarize_intrinsics_candidates,
)
from deep_shark_studio.projection.projection_preview import (
    render_fixed_layout,
    write_projection_comparison,
)
from deep_shark_studio.seam.layout_preview import LayoutPairCandidate, LayoutPreviewParams


def _model(camera: str, width: int = 64, height: int = 48) -> FisheyeCameraModel:
    return FisheyeCameraModel(
        camera=camera,
        camera_matrix=np.asarray(
            [[40.0, 0.0, width / 2.0], [0.0, 40.0, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion=np.zeros((4, 1), dtype=np.float64),
        image_size=(width, height),
        source="test",
        rms_px=0.1,
        accepted_samples=10,
    )


class ProjectionCandidateTests(unittest.TestCase):
    def test_intrinsics_summary_marks_available_and_missing(self) -> None:
        candidate = {
            "resolution": [64, 48],
            "intrinsics": {
                "front_left": {
                    "status": "success",
                    "model": "opencv_fisheye",
                    "resolution": [64, 48],
                    "camera_matrix": _model("front_left").camera_matrix.tolist(),
                    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
                    "rms_px": 0.2,
                    "accepted_input_count": 12,
                }
            },
        }
        report = summarize_intrinsics_candidates({}, candidate, Path("missing_session"), "triple_front_panorama", None)

        self.assertEqual("available", report["cameras"]["front_left"]["status"])
        self.assertEqual("missing", report["cameras"]["front"]["status"])
        self.assertFalse(report["all_required_intrinsics_available"])

    def test_rectilinear_map_and_valid_mask_sizes(self) -> None:
        frame = np.full((48, 64, 3), (20, 80, 180), dtype=np.uint8)
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as directory:
            report = build_rectilinear_projection(
                {"front": frame},
                {"front": _model("front")},
                Path(directory),
                balance_values=[0.0],
            )

            item = report["cameras"]["front"]
            self.assertEqual("generated", item["status"])
            self.assertEqual([48, 64], item["map_shape"])
            self.assertTrue((Path(directory) / item["valid_masks"]["balance_00"]).exists())
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)

    def test_missing_kd_does_not_fake_rectilinear_preview(self) -> None:
        frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            report = build_rectilinear_projection(
                {"front": frame},
                {},
                Path(directory),
                balance_values=[0.0],
            )

            self.assertEqual("skipped", report["cameras"]["front"]["status"])

    def test_equirectangular_projection_outputs_expected_keys(self) -> None:
        frames = {
            camera: np.full((48, 64, 3), (20, 80, 180), dtype=np.uint8)
            for camera in ("front_left", "front", "front_right")
        }
        models = {camera: _model(camera) for camera in frames}
        transforms = {
            camera: {"rotation_camera_to_front": np.eye(3).tolist()}
            for camera in frames
        }
        with tempfile.TemporaryDirectory() as directory:
            result = build_equirectangular_projection(
                frames,
                models,
                transforms,
                Path(directory),
                EquirectangularParams(canvas_width=80, canvas_height=40),
            )

            self.assertEqual({"front_left", "front", "front_right"}, set(result.warped_images))
            self.assertEqual((40, 80), result.valid_masks["front"].shape)
            self.assertIn("source", result.metadata["cameras"]["front"])

    def test_layout_consumes_projection_warped_images_and_comparison_writes(self) -> None:
        image = np.full((40, 80, 3), (20, 120, 20), dtype=np.uint8)
        side = np.full((40, 80, 3), (20, 20, 180), dtype=np.uint8)
        candidates = [
            LayoutPairCandidate(
                pair_id="left_front",
                side_camera="front_left",
                side_position="left",
                boundary_type="projection",
                seam_points=[(30, y) for y in range(40)],
            )
        ]

        layout, metrics = render_fixed_layout(
            {"front": image, "front_left": side},
            candidates,
            LayoutPreviewParams(0, 70, 0.30, 16),
        )

        with tempfile.TemporaryDirectory() as directory:
            files = write_projection_comparison(Path(directory), layout, layout)
            self.assertTrue((Path(directory) / files["perspective_vs_projection_side_by_side"]).exists())
        self.assertIn("front_preserved_ratio", metrics)


if __name__ == "__main__":
    unittest.main()
