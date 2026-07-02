from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.seam.layout_preview import LayoutPairCandidate, LayoutPreviewParams
from deep_shark_studio.seam.projection_readiness_audit import write_projection_readiness_audit
from deep_shark_studio.seam.vertical_safety import VerticalSafetyParams, VerticalSafetyRenderer
from deep_shark_studio.seam.vertical_safety_sweep import run_vertical_safety_sweep_from_warped_images


def _path(x: int, height: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(height)]


class VerticalSafetyTests(unittest.TestCase):
    def test_output_height_crop_size_without_resize(self) -> None:
        front = np.full((80, 40, 3), (0, 180, 0), dtype=np.uint8)
        side = np.full((80, 40, 3), (0, 0, 220), dtype=np.uint8)
        candidate = LayoutPairCandidate("left_front", "front_left", "left", "vertical", _path(20, 80))

        result = VerticalSafetyRenderer().render(
            {"front": front, "front_left": side},
            [candidate],
            LayoutPreviewParams(0, 30, 0.50, 0),
            VerticalSafetyParams(12, 0.80, 4),
        )

        self.assertEqual((12, 30, 3), result.image.shape)
        self.assertEqual(12, result.crop_y[1] - result.crop_y[0])

    def test_vertical_weight_center_is_one_and_edges_lower(self) -> None:
        weight = VerticalSafetyRenderer.vertical_weight(100, 0.80, 10)

        self.assertAlmostEqual(1.0, float(weight[50]), places=6)
        self.assertLess(float(weight[0]), 0.05)
        self.assertLess(float(weight[-1]), 0.05)

    def test_suppression_only_reduces_side_not_front(self) -> None:
        front = np.full((80, 40, 3), (0, 180, 0), dtype=np.uint8)
        side = np.full((80, 40, 3), (0, 0, 220), dtype=np.uint8)
        candidate = LayoutPairCandidate("left_front", "front_left", "left", "vertical", _path(20, 80))

        result = VerticalSafetyRenderer().render(
            {"front": front, "front_left": side},
            [candidate],
            LayoutPreviewParams(0, 40, 0.50, 0),
            VerticalSafetyParams(80, 0.60, 0),
        )

        self.assertTrue(np.all(result.image[50, 25:] == front[50, 25:]))
        self.assertLessEqual(
            int(np.count_nonzero(result.side_visible_after)),
            int(np.count_nonzero(result.side_visible_before)),
        )
        self.assertLessEqual(
            result.metrics["duplicate_risk_proxy_after"],
            result.metrics["duplicate_risk_proxy_before"],
        )

    def test_layout_id_is_stable(self) -> None:
        self.assertEqual(
            "h640_safe086_fade048",
            VerticalSafetyParams(640, 0.86, 48).layout_id,
        )

    def test_sweep_writes_report_and_keeps_calibration_hash(self) -> None:
        front = np.full((24, 48, 3), (0, 180, 0), dtype=np.uint8)
        side = np.full((24, 48, 3), (0, 0, 220), dtype=np.uint8)
        candidate = LayoutPairCandidate("left_front", "front_left", "left", "vertical", _path(22, 24))
        before = file_revision(CONFIG_DIR / "calibration.yaml")

        with tempfile.TemporaryDirectory() as directory:
            result = run_vertical_safety_sweep_from_warped_images(
                {"front": front, "front_left": side},
                [candidate],
                LayoutPreviewParams(0, 40, 0.50, 0),
                Path(directory),
                output_heights_px=[24, 20],
                vertical_safe_ratios=[1.0, 0.8],
                side_vertical_fade_px=[0, 4],
            )
            report = yaml.safe_load(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(8, result.candidate_count)
            self.assertFalse(report["writes_calibration_yaml"])
            self.assertTrue((result.output_dir / "projection_readiness_audit.md").exists())
            self.assertTrue((result.output_dir / "contact_sheets" / "best_vertical_safety_contact_sheet.png").exists())

        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)

    def test_projection_readiness_audit_is_read_only(self) -> None:
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = write_projection_readiness_audit(Path(directory))
            text = path.read_text(encoding="utf-8")
            self.assertIn("Fisheye Projection Readiness Audit", text)
            self.assertIn("warp/projection layer", text)
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
