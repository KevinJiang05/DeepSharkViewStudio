from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.projection.runtime_providers import (
    B2_FAR_FIELD_PROJECTION_SOURCE,
    CurrentPerspectiveProjectionProvider,
)
from deep_shark_studio.seam.far_field_custom_compositor import (
    render_far_field_custom_from_projection,
)
from deep_shark_studio.seam.far_field_layout_candidate_runtime import (
    FarFieldLayoutCandidateError,
    FarFieldLayoutRuntimeCandidate,
    load_far_field_layout_candidate,
    save_far_field_layout_candidate,
)
from deep_shark_studio.seam.layout_candidate_runtime import (
    LayoutCandidateRuntimeError,
    load_layout_candidate_for_runtime,
)
from deep_shark_studio.stitch_runtime_controller import RuntimeStitchController
from deep_shark_studio.stitch_runtime_modes import RuntimeStitchConfig, StitchRuntimeMode
from deep_shark_studio.stitching.camera_layout_adjust import (
    CameraAdjustParams,
    apply_post_warp_camera_adjustments,
)


def _warped_images(width: int = 100, height: int = 40) -> dict[str, np.ndarray]:
    front_left = np.zeros((height, width, 3), dtype=np.uint8)
    front_left[:, :45] = (0, 0, 220)
    front = np.zeros_like(front_left)
    front[:, 25:75] = (0, 180, 0)
    front_right = np.zeros_like(front_left)
    front_right[:, 55:] = (220, 0, 0)
    return {
        "front_left": front_left,
        "front": front,
        "front_right": front_right,
    }


def _candidate_data() -> dict:
    return {
        "schema_version": 1,
        "candidate_type": "far_field_layout",
        "profile_id": "triple_front_panorama",
        "source": {"calibration_hash": file_revision(CONFIG_DIR / "calibration.yaml")},
        "projection": {
            "source": "current_perspective",
            "note": "unit test",
        },
        "camera_adjust": {
            "front_left": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
            "front": {"x_offset_px": 8, "y_offset_px": 0, "scale": 1.0},
            "front_right": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
        },
        "output": {"width_px": 80, "height_px": 30},
        "far_field_blend": {
            "mode": "current_horizontal_feather",
            "use_profile_overlaps": True,
            "feather_width_override_px": None,
        },
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
    }


def _candidate_data_b2(candidate_directory: Path) -> dict:
    data = _candidate_data()
    data["projection"] = {
        "source": B2_FAR_FIELD_PROJECTION_SOURCE,
        "candidate_directory": str(candidate_directory),
        "note": "unit test B-2 source",
    }
    return data


def _write_candidate(directory: Path, data: dict | None = None) -> Path:
    path = directory / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(data or _candidate_data(), sort_keys=False),
        encoding="utf-8",
    )
    return path


class FakeStitcher:
    def __init__(self):
        self.output_width = 100
        self.output_height = 40
        self.composition = {"mode": "horizontal_feather", "feather_width": 10}
        self.stitch_points = {
            "left_front": [(35, 0), (35, 40)],
            "front_right": [(65, 0), (65, 40)],
        }
        self.overlaps = [
            {
                "name": "left_front",
                "cameras": ["front_left", "front"],
                "x_range": [25, 45],
                "seam": "left_front",
            },
            {
                "name": "front_right",
                "cameras": ["front", "front_right"],
                "x_range": [55, 75],
                "seam": "front_right",
            },
        ]
        self.use_intrinsics = False
        self.process_calls = 0
        self.warp_all_calls = 0
        self.warped = _warped_images()

    def warp_all(self, frames):
        self.warp_all_calls += 1
        return {camera: image.copy() for camera, image in self.warped.items()}

    def process(self, frames):
        self.process_calls += 1
        return self.warp_all(frames), np.full((40, 100, 3), 9, dtype=np.uint8)


class FakeB2CandidateProcessor:
    def __init__(self, candidate_directory: Path):
        self.directory = Path(candidate_directory)
        self.output_width = 100
        self.output_height = 40
        self.topology_name = "triple_front_panorama"
        self._masks = {
            camera: np.any(image != 0, axis=2)
            for camera, image in _warped_images().items()
        }
        self._weights = {
            "front_left": self._masks["front_left"].astype(np.float32) * 0.2,
            "front": self._masks["front"].astype(np.float32) * 0.8,
            "front_right": self._masks["front_right"].astype(np.float32) * 0.4,
        }
        self.process_calls = 0

    def process(self, frames):
        self.process_calls += 1
        return _warped_images(), np.full((40, 100, 3), 7, dtype=np.uint8)


class FarFieldCustomLayoutTests(unittest.TestCase):
    def test_shared_camera_adjust_can_shift_far_field_warp(self) -> None:
        warped = _warped_images()
        result = apply_post_warp_camera_adjustments(
            warped,
            valid_masks={camera: np.any(image != 0, axis=2) for camera, image in warped.items()},
            camera_adjust={"front": CameraAdjustParams(x_offset_px=10)},
        )

        self.assertEqual(10, result.metadata["cameras"]["front"]["x_offset_px"])
        self.assertEqual(warped["front"].shape, result.warped_images["front"].shape)

    def test_far_field_candidate_loader_rejects_near_field_candidate(self) -> None:
        near_data = {
            "schema_version": 2,
            "candidate_type": "front_priority_layout",
            "profile_id": "triple_front_panorama",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = _write_candidate(Path(temp), near_data)

            with self.assertRaises(FarFieldLayoutCandidateError):
                load_far_field_layout_candidate(path)

    def test_near_field_candidate_loader_rejects_far_field_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = _write_candidate(Path(temp))

            with self.assertRaises(LayoutCandidateRuntimeError):
                load_layout_candidate_for_runtime(path)

    def test_far_field_custom_compositor_outputs_candidate_size(self) -> None:
        stitcher = FakeStitcher()
        projection = CurrentPerspectiveProjectionProvider(stitcher).project({})
        with tempfile.TemporaryDirectory() as temp:
            candidate = load_far_field_layout_candidate(_write_candidate(Path(temp)))
            result = render_far_field_custom_from_projection(
                projection,
                candidate,
                stitcher,
            )

        self.assertEqual((30, 80), result.image.shape[:2])
        self.assertEqual("far_field_custom", result.metrics["runtime_mode"])
        self.assertIn("far_field_composition_ms", result.metrics["timing"])

    def test_far_field_default_still_calls_process(self) -> None:
        stitcher = FakeStitcher()
        result = RuntimeStitchController(
            stitcher,
            RuntimeStitchConfig(mode=StitchRuntimeMode.FAR_FIELD),
        ).process({})

        self.assertEqual(1, stitcher.process_calls)
        self.assertEqual(StitchRuntimeMode.FAR_FIELD, result.mode)
        self.assertEqual("far_field", result.status)

    def test_far_field_custom_uses_provider_not_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate_path = _write_candidate(Path(temp))
            stitcher = FakeStitcher()
            result = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.FAR_FIELD,
                    use_far_field_custom_layout=True,
                    far_field_layout_candidate_path=candidate_path,
                ),
            ).process({})

        self.assertEqual(0, stitcher.process_calls)
        self.assertEqual(1, stitcher.warp_all_calls)
        self.assertEqual("far_field_custom", result.status)
        self.assertEqual((30, 80), result.canvas.shape[:2])

    def test_far_field_custom_b2_candidate_uses_b2_provider_not_template_warp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            b2_dir = root / "b2"
            b2_dir.mkdir()
            candidate_path = _write_candidate(root, _candidate_data_b2(b2_dir))
            stitcher = FakeStitcher()
            with patch(
                "deep_shark_studio.projection.runtime_providers.CandidatePanoramaProcessor",
                FakeB2CandidateProcessor,
            ):
                result = RuntimeStitchController(
                    stitcher,
                    RuntimeStitchConfig(
                        mode=StitchRuntimeMode.FAR_FIELD,
                        use_far_field_custom_layout=True,
                        far_field_layout_candidate_path=candidate_path,
                    ),
                ).process({})

        self.assertEqual(0, stitcher.process_calls)
        self.assertEqual(0, stitcher.warp_all_calls)
        self.assertEqual("far_field_custom", result.status)
        self.assertEqual(
            B2_FAR_FIELD_PROJECTION_SOURCE,
            result.metrics["projection"]["projection_source"],
        )
        self.assertEqual("b2_weight_selection", result.metrics["composition_mode"])

    def test_save_far_field_candidate_does_not_modify_calibration(self) -> None:
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        stitcher = FakeStitcher()
        projection = CurrentPerspectiveProjectionProvider(stitcher).project({})
        candidate = FarFieldLayoutRuntimeCandidate(
            path=Path("preview"),
            schema_version=1,
            profile_id="triple_front_panorama",
            camera_adjust={
                "front_left": CameraAdjustParams(),
                "front": CameraAdjustParams(x_offset_px=4),
                "front_right": CameraAdjustParams(),
            },
            output_width_px=80,
            output_height_px=30,
            feather_width_override_px=None,
            calibration_hash=None,
            current_calibration_hash=None,
            warnings=(),
            raw={},
        )
        preview = render_far_field_custom_from_projection(projection, candidate, stitcher)
        with tempfile.TemporaryDirectory() as temp:
            directory = save_far_field_layout_candidate(
                Path(temp),
                "triple_front_panorama",
                candidate.camera_adjust,
                candidate.output_width_px,
                candidate.output_height_px,
                preview,
            )

            saved = yaml.safe_load((directory / "candidate.yaml").read_text(encoding="utf-8"))

        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)
        self.assertEqual("far_field_layout", saved["candidate_type"])
        self.assertNotIn("left_pair", saved)
        self.assertFalse(saved["writes_calibration_yaml"])

    def test_save_far_field_candidate_records_b2_projection_source(self) -> None:
        stitcher = FakeStitcher()
        projection = CurrentPerspectiveProjectionProvider(stitcher).project({})
        candidate = FarFieldLayoutRuntimeCandidate(
            path=Path("preview"),
            schema_version=1,
            profile_id="triple_front_panorama",
            camera_adjust={
                "front_left": CameraAdjustParams(),
                "front": CameraAdjustParams(),
                "front_right": CameraAdjustParams(),
            },
            output_width_px=80,
            output_height_px=30,
            feather_width_override_px=None,
            calibration_hash=None,
            current_calibration_hash=None,
            warnings=(),
            raw={},
        )
        preview = render_far_field_custom_from_projection(projection, candidate, stitcher)
        with tempfile.TemporaryDirectory() as temp:
            b2_dir = Path(temp) / "b2"
            b2_dir.mkdir()
            directory = save_far_field_layout_candidate(
                Path(temp),
                "triple_front_panorama",
                candidate.camera_adjust,
                candidate.output_width_px,
                candidate.output_height_px,
                preview,
                source={
                    "projection": {
                        "projection_source": B2_FAR_FIELD_PROJECTION_SOURCE,
                        "candidate_directory": str(b2_dir),
                    }
                },
            )
            saved = yaml.safe_load((directory / "candidate.yaml").read_text(encoding="utf-8"))
            loaded = load_far_field_layout_candidate(directory)

        self.assertEqual(B2_FAR_FIELD_PROJECTION_SOURCE, saved["projection"]["source"])
        self.assertEqual(B2_FAR_FIELD_PROJECTION_SOURCE, loaded.projection_source)
        self.assertEqual(b2_dir, loaded.b2_candidate_directory)


if __name__ == "__main__":
    unittest.main()
