from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from deep_shark_studio.projection.intrinsics_runtime_loader import (
    FisheyeIntrinsicsRuntimeError,
    load_fisheye_intrinsics_source,
)
from deep_shark_studio.projection.runtime_providers import (
    B2CandidateProjectionProvider,
    B2_FAR_FIELD_PROJECTION_SOURCE,
    CurrentPerspectiveProjectionProvider,
    FisheyeRectilinearParams,
    FisheyeRectilinearProjectionProvider,
    ProjectionResult,
    derive_nonzero_valid_masks,
)


class FakePerspectiveStitcher:
    def __init__(self, use_intrinsics: bool = False):
        self.use_intrinsics = use_intrinsics
        self.output_width = 12
        self.output_height = 6
        self.process_calls = 0
        self.warp_all_calls = 0
        self.warped = {
            "front_left": self._image(0, 4, (0, 0, 200)),
            "front": self._image(4, 8, (0, 160, 0)),
            "front_right": self._image(8, 12, (200, 0, 0)),
        }

    def _image(self, x0: int, x1: int, color: tuple[int, int, int]) -> np.ndarray:
        image = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        image[:, x0:x1] = color
        return image

    def process(self, frames):
        self.process_calls += 1
        raise AssertionError("CurrentPerspectiveProjectionProvider must not call process().")

    def warp_all(self, frames):
        self.warp_all_calls += 1
        return self.warped


class FakeTemplateCamera:
    @property
    def matrix(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64)


class FakeTemplateStitcher:
    def __init__(self):
        self.output_width = 32
        self.output_height = 24
        self.max_input_width = None
        self.use_intrinsics = False
        self.cameras = {
            "front_left": FakeTemplateCamera(),
            "front": FakeTemplateCamera(),
            "front_right": FakeTemplateCamera(),
        }
        self.process_calls = 0
        self.warp_all_calls = 0

    def process(self, frames):
        self.process_calls += 1
        raise AssertionError("FisheyeRectilinearProjectionProvider must not call process().")

    def warp_all(self, frames):
        self.warp_all_calls += 1
        raise AssertionError("FisheyeRectilinearProjectionProvider must not call warp_all().")


def _fisheye_candidate_data(include_front: bool = True) -> dict:
    intrinsics = {}
    for camera in ("front_left", "front", "front_right"):
        if camera == "front" and not include_front:
            continue
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
    return {
        "format": "DeepSharkFisheyeCalibrationCandidate",
        "topology": "triple_front_panorama",
        "resolution": [32, 24],
        "experimental": True,
        "apply_allowed": False,
        "intrinsics": intrinsics,
    }


def _write_fisheye_candidate(directory: Path, include_front: bool = True) -> Path:
    path = directory / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(_fisheye_candidate_data(include_front), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _raw_frames() -> dict[str, np.ndarray]:
    frames = {}
    colors = {
        "front_left": (0, 0, 200),
        "front": (0, 160, 0),
        "front_right": (200, 0, 0),
    }
    for camera, color in colors.items():
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[4:20, 4:28] = color
        frames[camera] = image
    return frames


class FakeB2CandidateProcessor:
    def __init__(self, candidate_directory: Path):
        self.directory = Path(candidate_directory)
        self.output_width = 12
        self.output_height = 6
        self.topology_name = "triple_front_panorama"
        self._masks = {
            "front_left": np.ones((6, 12), dtype=bool),
            "front": np.ones((6, 12), dtype=bool),
            "front_right": np.ones((6, 12), dtype=bool),
        }
        self.process_calls = 0

    def process(self, frames):
        self.process_calls += 1
        warped = {
            "front_left": np.full((6, 12, 3), 11, dtype=np.uint8),
            "front": np.full((6, 12, 3), 22, dtype=np.uint8),
            "front_right": np.full((6, 12, 3), 33, dtype=np.uint8),
        }
        canvas = np.full((6, 12, 3), 44, dtype=np.uint8)
        return warped, canvas


class CurrentPerspectiveProjectionProviderTests(unittest.TestCase):
    def test_provider_wraps_warp_all_without_stitch_process(self) -> None:
        stitcher = FakePerspectiveStitcher()
        result = CurrentPerspectiveProjectionProvider(stitcher).project({})

        self.assertIsInstance(result, ProjectionResult)
        self.assertEqual(1, stitcher.warp_all_calls)
        self.assertEqual(0, stitcher.process_calls)
        self.assertEqual(
            {"front_left", "front", "front_right"},
            set(result.warped_images),
        )
        self.assertEqual(
            {"front_left", "front", "front_right"},
            set(result.valid_masks),
        )
        for camera, mask in result.valid_masks.items():
            self.assertEqual(result.warped_images[camera].shape[:2], mask.shape)
            self.assertEqual(bool, mask.dtype)

    def test_provider_metadata_and_timings_describe_current_perspective(self) -> None:
        result = CurrentPerspectiveProjectionProvider(
            FakePerspectiveStitcher(use_intrinsics=True)
        ).project({})

        self.assertEqual("current_perspective", result.metadata["projection_source"])
        self.assertEqual(
            "CurrentPerspectiveProjectionProvider",
            result.metadata["provider_name"],
        )
        self.assertEqual([12, 6], result.metadata["canvas_size"])
        self.assertTrue(result.metadata["uses_intrinsics"])
        self.assertFalse(result.metadata["uses_fisheye"])
        self.assertFalse(result.metadata["uses_remap"])
        self.assertEqual(
            "nonzero_pixels_runtime_compatible",
            result.metadata["valid_mask_source"],
        )
        self.assertTrue(result.metadata["warning_reasons"])
        self.assertIn("projection_total_ms", result.timings)
        self.assertIn("warp_all_ms", result.timings)

    def test_derive_nonzero_valid_masks_matches_warped_canvas_size(self) -> None:
        warped = FakePerspectiveStitcher().warped
        masks = derive_nonzero_valid_masks(warped)

        for camera, image in warped.items():
            self.assertEqual(image.shape[:2], masks[camera].shape)
            self.assertGreater(int(np.count_nonzero(masks[camera])), 0)


class B2CandidateProjectionProviderTests(unittest.TestCase):
    def test_provider_wraps_b2_candidate_per_camera_warped_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "deep_shark_studio.projection.runtime_providers.CandidatePanoramaProcessor",
                FakeB2CandidateProcessor,
            ):
                result = B2CandidateProjectionProvider(Path(temp)).project({})

        self.assertIsInstance(result, ProjectionResult)
        self.assertEqual(B2_FAR_FIELD_PROJECTION_SOURCE, result.metadata["projection_source"])
        self.assertEqual("B2CandidateProjectionProvider", result.metadata["provider_name"])
        self.assertTrue(result.metadata["uses_fisheye"])
        self.assertTrue(result.metadata["uses_remap"])
        self.assertEqual("b2_candidate_remap_valid_mask", result.metadata["valid_mask_source"])
        self.assertEqual({"front_left", "front", "front_right"}, set(result.warped_images))
        self.assertEqual({"front_left", "front", "front_right"}, set(result.valid_masks))
        self.assertIn("b2_candidate_process_ms", result.timings)


class FisheyeRuntimeProviderTests(unittest.TestCase):
    def test_intrinsics_loader_rejects_missing_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = _write_fisheye_candidate(Path(temp), include_front=False)

            with self.assertRaisesRegex(FisheyeIntrinsicsRuntimeError, "front"):
                load_fisheye_intrinsics_source(path)

    def test_fisheye_provider_outputs_projection_result_and_geometry_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = load_fisheye_intrinsics_source(_write_fisheye_candidate(Path(temp)))
            provider = FisheyeRectilinearProjectionProvider(
                FakeTemplateStitcher(),
                source,
                FisheyeRectilinearParams(balance=0.6, fov_scale=1.0),
            )
            result = provider.project(_raw_frames())

            self.assertIsInstance(result, ProjectionResult)
            self.assertEqual({"front_left", "front", "front_right"}, set(result.warped_images))
            self.assertEqual({"front_left", "front", "front_right"}, set(result.valid_masks))
            self.assertTrue(result.metadata["uses_fisheye"])
            self.assertTrue(result.metadata["uses_remap"])
            self.assertFalse(result.metadata["uses_equirectangular"])
            self.assertEqual(
                "fisheye_remap_geometry_mask_then_template_warp",
                result.metadata["valid_mask_source"],
            )
            self.assertIn("fisheye_remap_ms", result.timings)
            self.assertIn("template_warp_ms", result.timings)

    def test_fisheye_provider_reuses_maps_across_project_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = load_fisheye_intrinsics_source(_write_fisheye_candidate(Path(temp)))
            provider = FisheyeRectilinearProjectionProvider(FakeTemplateStitcher(), source)

            self.assertEqual(3, provider.map_build_count)
            provider.project(_raw_frames())
            provider.project(_raw_frames())

            self.assertEqual(3, provider.map_build_count)

    def test_fisheye_provider_rejects_frame_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = load_fisheye_intrinsics_source(_write_fisheye_candidate(Path(temp)))
            provider = FisheyeRectilinearProjectionProvider(FakeTemplateStitcher(), source)
            frames = _raw_frames()
            frames["front"] = np.zeros((12, 16, 3), dtype=np.uint8)

            with self.assertRaisesRegex(ValueError, "does not match"):
                provider.project(frames)


if __name__ == "__main__":
    unittest.main()
