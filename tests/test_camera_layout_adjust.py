from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from deep_shark_studio.stitching.camera_layout_adjust import (
    CameraAdjustParams,
    apply_post_warp_camera_adjustments,
)


class CameraLayoutAdjustTests(unittest.TestCase):
    def test_explicit_mask_identity_uses_equivalent_fast_path(self) -> None:
        image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:5, 2:7] = True
        expected_image = image.copy()
        expected_image[~mask] = 0

        with (
            patch(
                "deep_shark_studio.stitching.camera_layout_adjust.cv2.warpAffine"
            ) as warp_affine,
            patch(
                "deep_shark_studio.stitching.camera_layout_adjust._valid_bbox_center"
            ) as valid_bbox_center,
        ):
            result = apply_post_warp_camera_adjustments(
                {"front": image},
                {"front": mask},
                {"front": CameraAdjustParams()},
            )

        warp_affine.assert_not_called()
        valid_bbox_center.assert_not_called()
        np.testing.assert_array_equal(expected_image, result.warped_images["front"])
        np.testing.assert_array_equal(mask, result.valid_masks["front"])
        np.testing.assert_array_equal(
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            result.transforms["front"],
        )
        self.assertFalse(np.shares_memory(image, result.warped_images["front"]))
        self.assertFalse(np.shares_memory(mask, result.valid_masks["front"]))
        camera_metadata = result.metadata["cameras"]["front"]
        self.assertTrue(camera_metadata["identity_fast_path"])
        self.assertEqual([3.5, 2.5], camera_metadata["scale_center"])
        self.assertAlmostEqual(20 / 48, camera_metadata["valid_ratio_before"])
        self.assertEqual(
            camera_metadata["valid_ratio_before"],
            camera_metadata["valid_ratio_after"],
        )
        self.assertEqual("provider_valid_masks", result.metadata["valid_mask_source"])

    def test_fast_path_is_limited_to_identity_with_explicit_mask(self) -> None:
        image = np.full((6, 8, 3), 17, dtype=np.uint8)
        mask = np.ones((6, 8), dtype=bool)

        with patch(
            "deep_shark_studio.stitching.camera_layout_adjust.cv2.warpAffine",
            wraps=cv2.warpAffine,
        ) as warp_affine:
            adjusted = apply_post_warp_camera_adjustments(
                {"front": image},
                {"front": mask},
                {"front": CameraAdjustParams(x_offset_px=1)},
            )
        self.assertEqual(2, warp_affine.call_count)
        self.assertFalse(adjusted.metadata["cameras"]["front"]["identity_fast_path"])

        with patch(
            "deep_shark_studio.stitching.camera_layout_adjust.cv2.warpAffine",
            wraps=cv2.warpAffine,
        ) as warp_affine:
            fallback = apply_post_warp_camera_adjustments(
                {"front": image},
                None,
                {"front": CameraAdjustParams()},
            )
        self.assertEqual(2, warp_affine.call_count)
        self.assertFalse(fallback.metadata["cameras"]["front"]["identity_fast_path"])


if __name__ == "__main__":
    unittest.main()
