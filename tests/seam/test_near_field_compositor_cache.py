from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from deep_shark_studio.seam.front_priority_compositor_core import (
    render_front_priority_layout_core,
)
from deep_shark_studio.seam.layout_candidate_runtime import (
    LayoutRuntimeCandidate,
    LayoutRuntimeProjection,
)
from deep_shark_studio.seam.layout_preview import LayoutPairCandidate
from deep_shark_studio.seam.layout_tuner import PairLayoutParams
from deep_shark_studio.seam.near_field_compositor import (
    clear_near_field_plan_cache,
    near_field_plan_cache_info,
    render_near_field_from_warped,
)
from deep_shark_studio.stitch_runtime_modes import ProjectionSource
from deep_shark_studio.stitching.camera_layout_adjust import CameraAdjustParams


CAMERAS = ("front_left", "front", "front_right")


def _candidate(*, vertical: bool = False) -> LayoutRuntimeCandidate:
    height = 48
    return LayoutRuntimeCandidate(
        path=Path("unit-near-candidate.yaml"),
        schema_version=2,
        profile_id="triple_front_panorama",
        camera_adjust={
            "front_left": CameraAdjustParams(),
            "front": CameraAdjustParams(y_offset_px=-1, scale=0.91),
            "front_right": CameraAdjustParams(),
        },
        left_pair=PairLayoutParams(
            side_shift_px=7,
            side_visible_fraction=0.45,
            feather_width_px=8,
        ),
        right_pair=PairLayoutParams(
            side_shift_px=9,
            side_visible_fraction=0.42,
            feather_width_px=10,
        ),
        output_width_px=80,
        output_height_px=40 if vertical else height,
        vertical_safety_enabled=vertical,
        vertical_safe_ratio=0.75 if vertical else 1.0,
        side_vertical_fade_px=6 if vertical else 0,
        pair_candidates=[
            LayoutPairCandidate(
                pair_id="left_front",
                side_camera="front_left",
                side_position="left",
                boundary_type="unit",
                seam_points=[(40, 0), (42, 24), (44, 47)],
            ),
            LayoutPairCandidate(
                pair_id="front_right",
                side_camera="front_right",
                side_position="right",
                boundary_type="unit",
                seam_points=[(54, 0), (56, 24), (58, 47)],
            ),
        ],
        calibration_hash=None,
        current_calibration_hash=None,
        projection=LayoutRuntimeProjection(
            source=ProjectionSource.CURRENT_PERSPECTIVE,
        ),
        warnings=(),
        raw={},
    )


def _frames(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        camera: rng.integers(0, 256, (48, 96, 3), dtype=np.uint8)
        for camera in CAMERAS
    }


def _masks() -> dict[str, np.ndarray]:
    masks = {camera: np.zeros((48, 96), dtype=bool) for camera in CAMERAS}
    masks["front_left"][2:47, :58] = True
    masks["front"][3:46, 14:82] = True
    masks["front_right"][1:45, 38:] = True
    masks["front"][18:22, 46:50] = False
    masks["front_left"][8:11, 20:24] = False
    return masks


class NearFieldRuntimePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_near_field_plan_cache()

    def tearDown(self) -> None:
        clear_near_field_plan_cache()

    def assert_plan_matches_legacy(self, *, vertical: bool) -> None:
        candidate = _candidate(vertical=vertical)
        masks = _masks()
        render_near_field_from_warped(
            _frames(11),
            candidate,
            valid_masks=masks,
        )
        changed_frames = _frames(29)
        planned = render_near_field_from_warped(
            changed_frames,
            candidate,
            valid_masks=masks,
        )
        legacy = render_front_priority_layout_core(
            changed_frames,
            candidate.pair_candidates,
            candidate.to_layout_params(),
            valid_masks=masks,
        )

        np.testing.assert_array_equal(legacy.image, planned.canvas)
        for key, expected in legacy.metrics.items():
            if key != "timing":
                self.assertEqual(expected, planned.metrics[key], key)
        self.assertTrue(planned.metrics["timing"]["runtime_plan_cache_hit"])
        self.assertTrue(planned.metrics["timing"]["runtime_plan_used"])
        self.assertEqual(legacy.crop_x, tuple(planned.debug["crop_x"]))
        self.assertEqual(legacy.crop_y, tuple(planned.debug["crop_y"]))

    def test_cached_nonvertical_plan_is_pixel_exact_for_a_new_frame(self) -> None:
        self.assert_plan_matches_legacy(vertical=False)

    def test_cached_vertical_plan_is_pixel_exact_for_a_new_frame(self) -> None:
        self.assert_plan_matches_legacy(vertical=True)

    def test_candidate_parameter_change_invalidates_plan(self) -> None:
        frames = _frames(3)
        masks = _masks()
        candidate = _candidate()
        render_near_field_from_warped(frames, candidate, valid_masks=masks)
        changed = replace(
            candidate,
            left_pair=replace(
                candidate.left_pair,
                side_shift_px=candidate.left_pair.side_shift_px + 1,
            ),
        )

        result = render_near_field_from_warped(
            frames,
            changed,
            valid_masks=masks,
        )

        info = near_field_plan_cache_info()
        self.assertFalse(result.metrics["timing"]["runtime_plan_cache_hit"])
        self.assertEqual(2, info["misses"])
        self.assertEqual(0, info["hits"])
        self.assertEqual(2, info["entries"])

    def test_valid_mask_content_change_invalidates_plan(self) -> None:
        frames = _frames(5)
        masks = _masks()
        candidate = _candidate()
        render_near_field_from_warped(frames, candidate, valid_masks=masks)
        masks["front"] = masks["front"].copy()
        masks["front"][12, 24] = ~masks["front"][12, 24]

        result = render_near_field_from_warped(
            frames,
            candidate,
            valid_masks=masks,
        )

        info = near_field_plan_cache_info()
        self.assertFalse(result.metrics["timing"]["runtime_plan_cache_hit"])
        self.assertEqual(2, info["misses"])
        self.assertEqual(2, info["entries"])

    def test_all_runtime_geometry_dimensions_participate_in_key(self) -> None:
        frames = _frames(6)
        masks = _masks()
        candidate = _candidate()
        changed_pairs = list(candidate.pair_candidates)
        changed_pairs[0] = replace(
            changed_pairs[0],
            seam_points=[(39, 0), (42, 24), (44, 47)],
        )
        changes = {
            "right_pair": replace(
                candidate,
                right_pair=replace(candidate.right_pair, side_shift_px=10),
            ),
            "camera_adjust": replace(
                candidate,
                camera_adjust={
                    **candidate.camera_adjust,
                    "front": CameraAdjustParams(y_offset_px=-1, scale=0.92),
                },
            ),
            "vertical": replace(
                candidate,
                output_height_px=40,
                vertical_safety_enabled=True,
                vertical_safe_ratio=0.75,
                side_vertical_fade_px=6,
            ),
            "seam_geometry": replace(
                candidate,
                pair_candidates=changed_pairs,
            ),
        }
        for name, changed in changes.items():
            with self.subTest(name=name):
                clear_near_field_plan_cache()
                render_near_field_from_warped(
                    frames,
                    candidate,
                    valid_masks=masks,
                )

                result = render_near_field_from_warped(
                    frames,
                    changed,
                    valid_masks=masks,
                )

                info = near_field_plan_cache_info()
                self.assertFalse(
                    result.metrics["timing"]["runtime_plan_cache_hit"]
                )
                self.assertEqual(2, info["misses"])
                self.assertEqual(0, info["hits"])

    def test_plan_cache_is_bounded_and_lru_evicted(self) -> None:
        frames = _frames(7)
        masks = _masks()
        candidate = _candidate()
        for delta in range(3):
            changed = replace(
                candidate,
                left_pair=replace(
                    candidate.left_pair,
                    side_shift_px=candidate.left_pair.side_shift_px + delta,
                ),
            )
            render_near_field_from_warped(
                frames,
                changed,
                valid_masks=masks,
            )

        info = near_field_plan_cache_info()
        self.assertEqual(info["max_entries"], info["entries"])
        self.assertEqual(2, info["max_entries"])
        self.assertEqual(1, info["evictions"])
        self.assertGreater(info["memory_bytes"], 0)
        self.assertLessEqual(info["memory_bytes"], info["max_memory_bytes"])

    def test_missing_explicit_masks_bypasses_geometry_cache(self) -> None:
        result = render_near_field_from_warped(_frames(13), _candidate())

        self.assertEqual((48, 80), result.canvas.shape[:2])
        info = near_field_plan_cache_info()
        self.assertEqual(0, info["entries"])
        self.assertEqual(1, info["bypasses"])

    def test_plan_larger_than_memory_budget_is_not_retained(self) -> None:
        with patch(
            "deep_shark_studio.seam.near_field_compositor."
            "_PLAN_CACHE_MAX_MEMORY_BYTES",
            1,
        ):
            render_near_field_from_warped(
                _frames(17),
                _candidate(),
                valid_masks=_masks(),
            )

            info = near_field_plan_cache_info()

        self.assertEqual(0, info["entries"])
        self.assertEqual(1, info["oversize_rejections"])

    def test_fragmented_vertical_plan_falls_back_without_losing_frame(self) -> None:
        with patch(
            "deep_shark_studio.seam.vertical_safety."
            "_MAX_VERTICAL_FADE_REGIONS",
            1,
        ):
            result = render_near_field_from_warped(
                _frames(19),
                _candidate(vertical=True),
                valid_masks=_masks(),
            )

        self.assertEqual((40, 80), result.canvas.shape[:2])
        self.assertTrue(result.metrics["timing"]["runtime_plan_build_failed"])
        self.assertEqual(0, near_field_plan_cache_info()["entries"])


if __name__ == "__main__":
    unittest.main()
