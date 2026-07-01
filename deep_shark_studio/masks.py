"""Mask generation utilities used by the surround-view stitcher."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


MaskSide = Literal[
    "left",
    "right",
    "front",
    "behind",
    "front_other_left",
    "front_other_right",
    "behind_other",
    "left_other",
    "right_other",
]


def _line_y(point_a: tuple[int, int], point_b: tuple[int, int], x: int) -> int:
    x1, y1 = point_a
    x2, y2 = point_b
    if x2 == x1:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return int(slope * x + intercept)


def crop_image_by_line(
    image: np.ndarray,
    point1: tuple[int, int],
    point2: tuple[int, int],
    point3: tuple[int, int],
    point4: tuple[int, int],
    side: MaskSide,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an image using the line-seam rules from the legacy AVM prototype."""
    height, width = image.shape[:2]
    mask = np.zeros_like(image)

    x1, _ = point1
    x2, _ = point2
    x3, _ = point3
    x4, _ = point4
    half_h = height // 2

    for x in range(width):
        y1_val = _line_y(point1, point2, x)
        y2_val = _line_y(point3, point4, x)
        y3_val = _line_y(point4, point2, x)

        if side == "left":
            if x <= x1:
                mask[:, x] = 255
            elif x1 < x <= x2:
                mask[max(0, y2_val) : min(height, y1_val), x] = 255
        elif side == "right":
            if x >= x1:
                mask[:, x] = 255
            elif x2 <= x < x1:
                mask[max(0, y1_val) : min(height, y2_val), x] = 255
        elif side == "front":
            if x1 <= x < x2:
                mask[: max(0, min(height, y1_val)), x] = 255
            elif x2 <= x < x4:
                y_max = _line_y(point1, point2, x2)
                mask[: max(0, min(height, y_max)), x] = 255
            elif x4 <= x < x3:
                mask[: max(0, min(height, y2_val)), x] = 255
        elif side == "behind":
            if x3 <= x < x4:
                mask[max(0, min(height, y2_val)) :, x] = 255
            elif x4 <= x < x2:
                y_min = _line_y(point1, point2, x2)
                mask[max(0, min(height, y_min)) :, x] = 255
            elif x2 <= x < x1:
                mask[max(0, min(height, y1_val)) :, x] = 255
        elif side == "front_other_left":
            if min(x2, x4) <= x < max(x2, x4):
                mask[max(0, y1_val) : half_h, x] = 255
            elif max(x2, x4) <= x < min(x1, x3):
                mask[max(0, y1_val) : half_h, x] = 255
            elif min(x1, x3) <= x:
                mask[:half_h, x] = 255
        elif side == "front_other_right":
            if min(x2, x4) <= x < max(x2, x4):
                mask[half_h : min(height, y3_val), x] = 255
            elif max(x2, x4) <= x < min(x1, x3):
                mask[half_h : min(height, y2_val), x] = 255
            elif min(x1, x3) <= x:
                mask[half_h:, x] = 255
        elif side == "behind_other":
            if x <= min(x1, x3):
                mask[:, x] = 255
            elif min(x1, x3) < x <= min(x2, x4):
                mask[max(0, y2_val) : min(height, y1_val), x] = 255
            elif min(x2, x4) < x <= max(x2, x4):
                mask[max(0, y2_val) : min(height, y3_val), x] = 255
        elif side == "left_other":
            if x1 <= x < x2:
                mask[: max(0, min(height, y1_val)), x] = 255
            elif x2 <= x < x4:
                mask[: max(0, min(height, y3_val)), x] = 255
            elif x4 <= x < x3:
                mask[: max(0, min(height, y2_val)), x] = 255
        elif side == "right_other":
            if x3 <= x < x4:
                mask[max(0, min(height, y2_val)) :, x] = 255
            elif x4 <= x < x2:
                mask[max(0, min(height, y3_val)) :, x] = 255
            elif x2 <= x < x1:
                mask[max(0, min(height, y1_val)) :, x] = 255

    return cv2.bitwise_and(image, mask), mask
