from __future__ import annotations

import unittest

import numpy as np

from deep_shark_studio.topology import compose_horizontal_feather, seam_x


def _reference_compose_horizontal_feather(
    warped_images: dict[str, np.ndarray],
    output_width: int,
    output_height: int,
    stitch_points: dict,
    overlaps: list[dict],
    default_feather_width: int,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Pre-optimization implementation retained as a pixel oracle."""
    if not warped_images:
        return np.zeros((output_height, output_width, 3), dtype=np.uint8)

    weights: dict[str, np.ndarray] = {}
    masks = valid_masks or {}
    for camera_name, image in warped_images.items():
        if image.shape[:2] != (output_height, output_width):
            raise ValueError(f"Warped image size mismatch for {camera_name}")
        mask = masks.get(camera_name)
        if mask is None:
            mask = np.any(image != 0, axis=2)
        else:
            mask = np.asarray(mask)
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Valid mask size mismatch for {camera_name}")
        weights[camera_name] = mask.astype(np.float32, copy=True)

    x_coordinates = np.arange(output_width, dtype=np.float32)
    for overlap in overlaps:
        cameras = overlap.get("cameras", [])
        if len(cameras) != 2:
            raise ValueError("Each horizontal overlap must declare two cameras")
        left_camera, right_camera = str(cameras[0]), str(cameras[1])
        if left_camera not in weights or right_camera not in weights:
            continue

        x_range = overlap.get("x_range", [])
        if len(x_range) != 2:
            raise ValueError("Each horizontal overlap must declare x_range")
        overlap_start, overlap_end = sorted(
            (float(x_range[0]), float(x_range[1]))
        )
        seam_position = seam_x(stitch_points, str(overlap.get("seam", "")))
        if not overlap_start <= seam_position <= overlap_end:
            raise ValueError(
                f"Seam for {left_camera}/{right_camera} is outside its overlap"
            )

        feather_width = max(
            1.0,
            float(overlap.get("feather_width", default_feather_width)),
        )
        transition_start = max(
            overlap_start,
            seam_position - feather_width / 2.0,
        )
        transition_end = min(
            overlap_end,
            seam_position + feather_width / 2.0,
        )
        if transition_end <= transition_start:
            transition_start, transition_end = overlap_start, overlap_end

        in_overlap = (x_coordinates >= overlap_start) & (
            x_coordinates <= overlap_end
        )
        right_factor = np.clip(
            (x_coordinates - transition_start)
            / max(1.0, transition_end - transition_start),
            0.0,
            1.0,
        )
        left_factor = 1.0 - right_factor
        weights[left_camera] *= np.where(
            in_overlap,
            left_factor,
            1.0,
        )[None, :]
        weights[right_camera] *= np.where(
            in_overlap,
            right_factor,
            1.0,
        )[None, :]

    accumulator = np.zeros(
        (output_height, output_width, 3),
        dtype=np.float32,
    )
    total_weight = np.zeros(
        (output_height, output_width),
        dtype=np.float32,
    )
    for camera_name, image in warped_images.items():
        weight = weights[camera_name]
        accumulator += image.astype(np.float32) * weight[:, :, None]
        total_weight += weight

    canvas = np.zeros_like(accumulator)
    valid = total_weight > 1e-6
    canvas[valid] = accumulator[valid] / total_weight[valid][:, None]
    return np.clip(canvas, 0, 255).astype(np.uint8)


class HorizontalFeatherComposeTests(unittest.TestCase):
    def test_zero_sized_dimensions_keep_reference_contract(self) -> None:
        for height, width in ((0, 4), (3, 0), (0, 0)):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            mask = np.zeros((height, width), dtype=bool)
            expected = _reference_compose_horizontal_feather(
                {"front": image},
                width,
                height,
                {},
                [],
                1,
                valid_masks={"front": mask},
            )
            actual = compose_horizontal_feather(
                {"front": image},
                width,
                height,
                {},
                [],
                1,
                valid_masks={"front": mask},
            )
            with self.subTest(size=(width, height)):
                np.testing.assert_array_equal(expected, actual)

    def test_random_geometry_masks_match_reference_pixel_for_pixel(self) -> None:
        rng = np.random.default_rng(9371)
        for case_index in range(40):
            height = int(rng.integers(1, 25))
            width = int(rng.integers(1, 65))
            camera_count = int(rng.integers(1, 5))
            camera_names = tuple(f"camera_{index}" for index in range(camera_count))
            images = {
                name: rng.integers(
                    0,
                    256,
                    (height, width, 3),
                    dtype=np.uint8,
                )
                for name in camera_names
            }
            masks = {
                name: rng.random((height, width)) > 0.35
                for name in camera_names
            }
            # Explicitly exercise valid black content.
            black_selection = masks[camera_names[0]] & (
                rng.random((height, width)) > 0.7
            )
            images[camera_names[0]][black_selection] = 0

            stitch_points = {}
            overlaps = []
            for index in range(max(0, camera_count - 1)):
                overlap_start = float(rng.uniform(-width * 0.25, width * 0.75))
                overlap_end = float(
                    rng.uniform(
                        max(overlap_start + 0.01, width * 0.25),
                        width * 1.25,
                    )
                )
                seam_position = float(rng.uniform(overlap_start, overlap_end))
                seam_name = f"seam_{index}"
                stitch_points[seam_name] = [
                    [seam_position, -2.0],
                    [seam_position, height + 2.0],
                ]
                overlaps.append(
                    {
                        "cameras": [camera_names[index], camera_names[index + 1]],
                        "x_range": [overlap_start, overlap_end],
                        "seam": seam_name,
                        "feather_width": float(rng.uniform(0.1, width * 1.5 + 1)),
                    }
                )

            expected = _reference_compose_horizontal_feather(
                images,
                width,
                height,
                stitch_points,
                overlaps,
                7,
                valid_masks={name: mask.copy() for name, mask in masks.items()},
            )
            actual = compose_horizontal_feather(
                images,
                width,
                height,
                stitch_points,
                overlaps,
                7,
                valid_masks={name: mask.copy() for name, mask in masks.items()},
            )
            with self.subTest(case_index=case_index, size=(width, height)):
                np.testing.assert_array_equal(expected, actual)

    def test_cache_key_tracks_mutated_seam_and_overlap(self) -> None:
        width, height = 23, 7
        left = np.zeros((height, width, 3), dtype=np.uint8)
        left[:, :, 2] = 230
        right = np.zeros_like(left)
        right[:, :, 1] = 210
        images = {"left": left, "right": right}
        masks = {
            "left": np.ones((height, width), dtype=bool),
            "right": np.ones((height, width), dtype=bool),
        }
        stitch_points = {"pair": [[8.0, 0.0], [8.0, float(height)]]}
        overlaps = [
            {
                "cameras": ["left", "right"],
                "x_range": [3.0, 18.0],
                "seam": "pair",
                "feather_width": 5.0,
            }
        ]
        before = compose_horizontal_feather(
            images,
            width,
            height,
            stitch_points,
            overlaps,
            5,
            valid_masks=masks,
        )

        stitch_points["pair"] = [[14.0, 0.0], [14.0, float(height)]]
        overlaps[0]["feather_width"] = 11.0
        expected = _reference_compose_horizontal_feather(
            images,
            width,
            height,
            stitch_points,
            overlaps,
            5,
            valid_masks=masks,
        )
        actual = compose_horizontal_feather(
            images,
            width,
            height,
            stitch_points,
            overlaps,
            5,
            valid_masks=masks,
        )

        self.assertFalse(np.array_equal(before, actual))
        np.testing.assert_array_equal(expected, actual)

    def test_numeric_mask_fallback_matches_reference_without_mutation(self) -> None:
        rng = np.random.default_rng(731)
        width, height = 29, 11
        images = {
            "left": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            "right": rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
        }
        masks = {
            "left": rng.random((height, width), dtype=np.float32),
            "right": rng.random((height, width), dtype=np.float32),
        }
        original_masks = {name: mask.copy() for name, mask in masks.items()}
        stitch_points = {"pair": [[13.5, 0.0], [13.5, float(height)]]}
        overlaps = [
            {
                "cameras": ["left", "right"],
                "x_range": [-2.0, 31.0],
                "seam": "pair",
                "feather_width": 9.3,
            }
        ]

        expected = _reference_compose_horizontal_feather(
            images,
            width,
            height,
            stitch_points,
            overlaps,
            7,
            valid_masks={name: mask.copy() for name, mask in masks.items()},
        )
        actual = compose_horizontal_feather(
            images,
            width,
            height,
            stitch_points,
            overlaps,
            7,
            valid_masks=masks,
        )

        np.testing.assert_array_equal(expected, actual)
        for name, mask in masks.items():
            np.testing.assert_array_equal(original_masks[name], mask)


if __name__ == "__main__":
    unittest.main()
