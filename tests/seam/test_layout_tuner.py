from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.seam.layout_preview import LayoutPreviewParams
from deep_shark_studio.seam.layout_tuner import (
    LAYOUT_TUNER_PRESETS,
    LAYOUT_TUNER_PRESETS_V2,
    CameraAdjustParams,
    LayoutPreviewParamsV2,
    PairLayoutParams,
    LayoutTunerParams,
    apply_post_warp_camera_adjustments,
    layout_tuner_params_v1_to_v2,
    layout_tuner_pair_candidates,
    render_front_priority_layout_preview,
    save_layout_tuner_candidate,
)


def _warped_images(width: int = 2200, height: int = 700) -> dict[str, np.ndarray]:
    front = np.zeros((height, width, 3), dtype=np.uint8)
    front[:, 650:1550] = (0, 180, 0)
    left = np.zeros_like(front)
    left[:, :850] = (0, 0, 220)
    right = np.zeros_like(front)
    right[:, 1350:] = (220, 0, 0)
    return {
        "front_left": left,
        "front": front,
        "front_right": right,
    }


def _profile(height: int = 700) -> dict:
    return {
        "canvas": {"width": 2200, "height": height},
        "overlaps": [
            {
                "name": "left_front",
                "cameras": ["front_left", "front"],
                "seam": "left_front",
                "x_range": [650, 850],
            },
            {
                "name": "front_right",
                "cameras": ["front", "front_right"],
                "seam": "front_right",
                "x_range": [1350, 1550],
            },
        ],
        "stitch_points": {
            "left_front": [[750, 0], [750, height]],
            "front_right": [[1450, 0], [1450, height]],
        },
    }


class LayoutTunerTests(unittest.TestCase):
    def test_layout_preview_params_defaults_are_baseline(self) -> None:
        params = LayoutPreviewParams()

        self.assertEqual(40, params.side_shift_px)
        self.assertEqual(1800, params.output_width_px)
        self.assertAlmostEqual(0.30, params.side_visible_fraction)
        self.assertEqual(24, params.feather_width_px)

    def test_presets_apply_expected_values(self) -> None:
        conservative = LAYOUT_TUNER_PRESETS["Conservative"]
        wide = LAYOUT_TUNER_PRESETS["Wide"]
        hard = LAYOUT_TUNER_PRESETS["Hard Seam"]
        front_smaller = LAYOUT_TUNER_PRESETS_V2["Front Smaller"]

        self.assertEqual(1700, conservative.output_width_px)
        self.assertAlmostEqual(0.86, conservative.vertical_safe_ratio)
        self.assertEqual(2000, wide.output_width_px)
        self.assertEqual(0, hard.feather_width_px)
        self.assertAlmostEqual(0.94, front_smaller.camera_adjust["front"].scale)

    def test_v1_params_convert_to_v2_with_equal_pairs(self) -> None:
        params = layout_tuner_params_v1_to_v2(
            LayoutTunerParams(
                side_shift_px=35,
                output_width_px=1700,
                side_visible_fraction=0.25,
                feather_width_px=16,
                output_height_px=640,
                vertical_safe_ratio=0.86,
                side_vertical_fade_px=48,
            )
        )

        self.assertEqual(params.left_pair, params.right_pair)
        self.assertEqual(35, params.left_pair.side_shift_px)
        self.assertEqual(1700, params.output_width_px)
        self.assertEqual(640, params.output_height_px)
        self.assertAlmostEqual(0.86, params.vertical_safe_ratio)

    def test_v2_default_left_right_pairs_are_equal(self) -> None:
        params = LayoutPreviewParamsV2()

        self.assertEqual(params.left_pair, params.right_pair)

    def test_side_shift_changes_output(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        first = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(side_shift_px=0),
        )
        shifted = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(side_shift_px=80),
        )

        self.assertFalse(np.array_equal(first.image, shifted.image))

    def test_unlocked_pair_params_can_differ_and_affect_respective_pair(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        images = _warped_images()
        gradient = (np.arange(images["front"].shape[1], dtype=np.uint16) % 255).astype(np.uint8)
        images["front_left"][:, :850, 2] = gradient[:850]
        images["front_right"][:, 1350:, 0] = gradient[1350:]
        base = render_front_priority_layout_preview(
            images,
            candidates,
            LayoutPreviewParamsV2(
                left_pair=PairLayoutParams(side_shift_px=0),
                right_pair=PairLayoutParams(side_shift_px=0),
            ),
        )
        left_shifted = render_front_priority_layout_preview(
            images,
            candidates,
            LayoutPreviewParamsV2(
                left_pair=PairLayoutParams(side_shift_px=80),
                right_pair=PairLayoutParams(side_shift_px=0),
            ),
        )
        right_shifted = render_front_priority_layout_preview(
            images,
            candidates,
            LayoutPreviewParamsV2(
                left_pair=PairLayoutParams(side_shift_px=0),
                right_pair=PairLayoutParams(side_shift_px=80),
            ),
        )

        mid = base.image.shape[1] // 2
        rows = slice(80, None)
        self.assertFalse(np.array_equal(base.image[rows, :mid], left_shifted.image[rows, :mid]))
        self.assertTrue(np.array_equal(base.image[rows, mid:], left_shifted.image[rows, mid:]))
        self.assertTrue(np.array_equal(base.image[rows, :mid], right_shifted.image[rows, :mid]))
        self.assertFalse(np.array_equal(base.image[rows, mid:], right_shifted.image[rows, mid:]))

    def test_invalid_or_none_profile_seam_points_fall_back_to_overlap_center(self) -> None:
        profile = _profile()
        profile["stitch_points"]["left_front"] = [[None, None], [None, 700]]
        candidates = layout_tuner_pair_candidates(profile)

        left = next(candidate for candidate in candidates if candidate.pair_id == "left_front")
        self.assertEqual([(750, 0), (750, 700)], left.seam_points)

    def test_output_width_and_height_crop_sizes(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        width_only = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(output_width_px=1700),
        )
        vertical = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(
                output_width_px=1700,
                output_height_px=640,
                vertical_safe_ratio=0.86,
                side_vertical_fade_px=48,
            ),
        )

        self.assertEqual((700, 1700), width_only.image.shape[:2])
        self.assertEqual((640, 1700), vertical.image.shape[:2])
        self.assertTrue(vertical.vertical_safety_enabled)

    def test_side_visible_fraction_limits_side_area(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        narrow = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(side_visible_fraction=0.20),
        )
        wide = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(side_visible_fraction=0.40),
        )

        self.assertLess(
            narrow.metrics["side_visible_ratio"],
            wide.metrics["side_visible_ratio"],
        )

    def test_feather_zero_hard_preview_renders(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        result = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(feather_width_px=0),
        )

        self.assertIn("L040_030_00_R040_030_00", result.layout_id)
        self.assertEqual((700, 1800), result.image.shape[:2])

    def test_post_warp_camera_adjust_moves_image_and_mask_together(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[30:50, 40:60] = (10, 120, 240)
        mask = np.zeros((80, 120), dtype=bool)
        mask[30:50, 40:60] = True

        result = apply_post_warp_camera_adjustments(
            {"front": image},
            {"front": mask},
            {"front": CameraAdjustParams(x_offset_px=10, y_offset_px=-6, scale=1.0)},
        )
        adjusted_mask = result.valid_masks["front"]
        ys, xs = np.where(adjusted_mask)

        self.assertEqual(mask.sum(), adjusted_mask.sum())
        self.assertAlmostEqual(float(xs.mean()), 59.5, delta=0.5)
        self.assertAlmostEqual(float(ys.mean()), 33.5, delta=0.5)
        self.assertTrue(np.all(result.warped_images["front"][adjusted_mask] != 0))

    def test_post_warp_front_scale_changes_preview_and_mask_transform(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        base = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutPreviewParamsV2(),
        )
        scaled = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutPreviewParamsV2(
                camera_adjust={
                    "front_left": CameraAdjustParams(),
                    "front": CameraAdjustParams(scale=0.94),
                    "front_right": CameraAdjustParams(),
                }
            ),
        )

        self.assertFalse(np.array_equal(base.image, scaled.image))
        self.assertIn("front", scaled.adjusted_valid_masks)
        self.assertLess(
            int(np.count_nonzero(scaled.adjusted_valid_masks["front"])),
            int(np.count_nonzero(base.adjusted_valid_masks["front"])),
        )

    def test_candidate_save_does_not_modify_calibration_yaml(self) -> None:
        candidates = layout_tuner_pair_candidates(_profile())
        preview = render_front_priority_layout_preview(
            _warped_images(),
            candidates,
            LayoutTunerParams(),
        )
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = save_layout_tuner_candidate(
                Path(directory),
                "triple_front_panorama",
                LayoutTunerParams(),
                preview,
                source={"mode": "unit_test"},
            )
            data = yaml.safe_load((output / "candidate.yaml").read_text(encoding="utf-8"))

            self.assertTrue((output / "preview.png").exists())
            self.assertTrue((output / "masks" / "side_visible_mask.png").exists())
            self.assertTrue((output / "debug" / "adjusted_front.png").exists())
            self.assertTrue((output / "debug" / "adjusted_valid_front.png").exists())
            self.assertTrue((output / "debug" / "camera_adjust_overlay.png").exists())
            self.assertEqual(2, data["schema_version"])
            self.assertEqual("front_priority_layout", data["candidate_type"])
            self.assertEqual(40, data["layout"]["side_shift_px"])
            self.assertIn("left_pair", data)
            self.assertIn("right_pair", data)
            self.assertIn("camera_adjust", data)
            self.assertIn("output", data)
            self.assertEqual(1800, data["output"]["width_px"])
            self.assertFalse(data["writes_calibration_yaml"])
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
