from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.seam.layout_preview import (
    FrontPriorityLayoutPreviewRenderer,
    LayoutPairCandidate,
    LayoutPreviewParams,
)
from deep_shark_studio.seam.layout_sweep import run_front_priority_layout_sweep


def _path(x: int, height: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(height)]


class FrontPriorityLayoutSweepTests(unittest.TestCase):
    def test_side_shift_moves_image_in_expected_directions(self) -> None:
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[:, 1] = (0, 0, 255)
        renderer = FrontPriorityLayoutPreviewRenderer()

        shifted_left = renderer.shift_side_image(image, "left", 2)
        shifted_right = renderer.shift_side_image(image, "right", 1)

        self.assertTrue(np.all(shifted_left[:, 3] == (0, 0, 255)))
        self.assertTrue(np.all(shifted_left[:, :2] == 0))
        self.assertTrue(np.all(shifted_right[:, 0] == (0, 0, 255)))
        self.assertTrue(np.all(shifted_right[:, -1] == 0))

    def test_shift_moves_mask_boundary_with_side(self) -> None:
        height, width = 10, 30
        front = np.full((height, width, 3), (20, 160, 20), dtype=np.uint8)
        side = np.full((height, width, 3), (20, 20, 200), dtype=np.uint8)
        candidate = LayoutPairCandidate(
            pair_id="left_front",
            side_camera="front_left",
            side_position="left",
            boundary_type="vertical",
            seam_points=_path(8, height),
        )

        result = FrontPriorityLayoutPreviewRenderer().render(
            {"front": front, "front_left": side},
            [candidate],
            LayoutPreviewParams(4, width, 0.50, 0),
        )

        self.assertTrue(np.any(result.side_visible_mask[:, 11]))
        self.assertFalse(np.any(result.side_visible_mask[:, 13]))

    def test_side_visible_clamp_reduces_side_invasion(self) -> None:
        height, width = 10, 40
        front = np.full((height, width, 3), (20, 160, 20), dtype=np.uint8)
        side = np.full((height, width, 3), (20, 20, 200), dtype=np.uint8)
        candidate = LayoutPairCandidate(
            pair_id="left_front",
            side_camera="front_left",
            side_position="left",
            boundary_type="vertical",
            seam_points=_path(30, height),
        )
        renderer = FrontPriorityLayoutPreviewRenderer()

        narrow = renderer.render(
            {"front": front, "front_left": side},
            [candidate],
            LayoutPreviewParams(0, width, 0.20, 0),
        )
        wide = renderer.render(
            {"front": front, "front_left": side},
            [candidate],
            LayoutPreviewParams(0, width, 0.60, 0),
        )

        self.assertLess(narrow.metrics["side_invasion_ratio"], wide.metrics["side_invasion_ratio"])

    def test_output_crop_width_without_resize(self) -> None:
        height, width = 12, 50
        front = np.full((height, width, 3), (20, 160, 20), dtype=np.uint8)
        side = np.full((height, width, 3), (20, 20, 200), dtype=np.uint8)
        candidate = LayoutPairCandidate(
            pair_id="front_right",
            side_camera="front_right",
            side_position="right",
            boundary_type="dp",
            seam_points=_path(38, height),
        )

        result = FrontPriorityLayoutPreviewRenderer().render(
            {"front": front, "front_right": side},
            [candidate],
            LayoutPreviewParams(3, 24, 0.30, 16),
        )

        self.assertEqual((height, 24, 3), result.image.shape)
        self.assertEqual(24, result.crop[1] - result.crop[0])

    def test_layout_id_is_stable_and_readable(self) -> None:
        params = LayoutPreviewParams(80, 1900, 0.25, 24)

        self.assertEqual("shift_080_crop_1900_side_025_feather_24", params.layout_id)

    def test_sweep_writes_report_previews_and_keeps_calibration_hash(self) -> None:
        height, width = 16, 48
        front = np.full((height, width, 3), (20, 160, 20), dtype=np.uint8)
        side = np.full((height, width, 3), (20, 20, 200), dtype=np.uint8)
        candidates = [
            LayoutPairCandidate(
                pair_id="left_front",
                side_camera="front_left",
                side_position="left",
                boundary_type="vertical",
                seam_points=_path(16, height),
            )
        ]
        before = file_revision(CONFIG_DIR / "calibration.yaml")

        with tempfile.TemporaryDirectory() as directory:
            result = run_front_priority_layout_sweep(
                {"front": front, "front_left": side},
                candidates,
                Path(directory),
                side_shifts_px=[0, 40],
                output_widths_px=[48, 32],
                side_visible_fractions=[0.20],
                feather_widths=[0, 16, 24],
            )
            report = yaml.safe_load(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(12, result.candidate_count)
            self.assertTrue((result.output_dir / "candidates" / "shift_000_crop_48_side_020_feather_00.png").exists())
            self.assertTrue((result.output_dir / "contact_sheets" / "all_candidates_contact_sheet.png").exists())
            self.assertFalse(report["writes_calibration_yaml"])
            self.assertEqual([0, 40], report["sweep_params"]["side_shifts_px"])
            self.assertEqual([0, 16, 24], report["sweep_params"]["feather_widths"])

        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
