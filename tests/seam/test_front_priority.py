from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.seam.boundary import BoundarySearchParams, SideBoundaryCandidateFinder
from deep_shark_studio.seam.cost import (
    FrontPriorityCostParams,
    SeamCostBuilder,
    SeamCostParams,
    SeamCostResult,
)
from deep_shark_studio.seam.dp import DPSeamParams, SeamPath, VerticalDPSeamFinder
from deep_shark_studio.seam.dynamic import simulate_front_priority_dynamic_boundaries
from deep_shark_studio.seam.preview import SeamPreviewRenderer
from deep_shark_studio.seam.projection_audit import write_projection_audit
from deep_shark_studio.seam.quality import FrontPriorityQualityGate
from deep_shark_studio.seam.roles import resolve_front_priority_pair_role
from deep_shark_studio.seam.runner import generate_front_priority_seam_candidates


def _cost_result(cost: np.ndarray, x_range: tuple[int, int]) -> SeamCostResult:
    valid = np.ones_like(cost, dtype=bool)
    zeros = np.zeros_like(cost, dtype=np.float32)
    return SeamCostResult(
        cost=cost.astype(np.float32),
        valid_mask=valid,
        color_cost=zeros,
        gradient_cost=zeros,
        edge_cost=zeros.copy(),
        margin_cost=zeros,
        x_range=x_range,
        front_priority_cost=zeros,
        high_difference_cost=zeros.copy(),
        object_proximity_cost=zeros.copy(),
        side_position="left",
    )


class FrontPrioritySeamTests(unittest.TestCase):
    def test_front_priority_cost_rises_when_side_invades_front(self) -> None:
        image_a = np.full((8, 24, 3), 80, dtype=np.uint8)
        image_b = np.full((8, 24, 3), 82, dtype=np.uint8)
        params = SeamCostParams(
            safe_erode_px=0,
            front_priority=FrontPriorityCostParams(enabled=True),
        )

        left = SeamCostBuilder().build(
            image_a,
            image_b,
            (4, 15),
            params,
            side_position="left",
        )
        right = SeamCostBuilder().build(
            image_a,
            image_b,
            (4, 15),
            params,
            side_position="right",
        )

        self.assertLess(float(left.front_priority_cost[:, 0].mean()), 0.01)
        self.assertGreater(float(left.front_priority_cost[:, -1].mean()), 0.95)
        self.assertLess(float(right.front_priority_cost[:, -1].mean()), 0.01)
        self.assertGreater(float(right.front_priority_cost[:, 0].mean()), 0.95)

    def test_vertical_boundary_candidate_chooses_low_cost_x(self) -> None:
        cost = np.full((16, 21), 4.0, dtype=np.float32)
        cost[:, 6] = 0.02
        result = SideBoundaryCandidateFinder().search(
            _cost_result(cost, (100, 120)),
            SeamPath([(8, y) for y in range(16)], [(108, y) for y in range(16)], 2.0, 2.0, 0.0),
            "left",
            BoundarySearchParams(vertical_fractions=(0.20, 0.30, 0.60)),
        )

        self.assertEqual("vertical", result.vertical.kind)
        self.assertEqual(106, result.vertical.x_canvas)
        self.assertEqual("dp", result.dp.kind)
        self.assertEqual("vertical", result.recommended_boundary_type)

    def test_front_dominant_hard_preview_preserves_front(self) -> None:
        height, width = 10, 20
        front = np.full((height, width, 3), (0, 180, 0), dtype=np.uint8)
        side = np.full((height, width, 3), (0, 0, 220), dtype=np.uint8)
        cost = np.ones((height, 6), dtype=np.float32)
        boundary = SideBoundaryCandidateFinder().search(
            _cost_result(cost, (5, 10)),
            SeamPath([(3, y) for y in range(height)], [(8, y) for y in range(height)], 1.0, 1.0, 0.0),
            "left",
            BoundarySearchParams(vertical_fractions=(0.20,)),
        )

        preview, side_mask, front_mask = SeamPreviewRenderer.front_dominant_preview(
            front,
            side,
            boundary,
            "left",
            0,
        )

        self.assertTrue(np.all(preview[:, 12:] == front[:, 12:]))
        self.assertTrue(np.any(side_mask[:, :8]))
        self.assertTrue(np.all(front_mask[:, 12:]))

    def test_pair_roles_keep_left_and_right_direction(self) -> None:
        left = resolve_front_priority_pair_role("left_front", ["front_left", "front"])
        right = resolve_front_priority_pair_role("front_right", ["front", "front_right"])

        self.assertEqual("front_left", left.side_camera)
        self.assertEqual("left", left.side_position)
        self.assertEqual("front_right", right.side_camera)
        self.assertEqual("right", right.side_position)

    def test_quality_gate_rejects_large_side_invasion(self) -> None:
        height, width = 12, 21
        cost = np.ones((height, width), dtype=np.float32)
        boundary = SideBoundaryCandidateFinder().search(
            _cost_result(cost, (50, 70)),
            SeamPath([(15, y) for y in range(height)], [(65, y) for y in range(height)], 0.1, 0.1, 0.0),
            "left",
            BoundarySearchParams(vertical_fractions=(0.80,)),
        )

        quality = FrontPriorityQualityGate().evaluate(_cost_result(cost, (50, 70)), boundary)

        self.assertEqual("rejected", quality.status)
        self.assertTrue(any("side_invasion_ratio" in reason for reason in quality.reject_reasons))

    def test_front_priority_runner_saves_v2_reports_without_calibration_change(self) -> None:
        height, width = 32, 80
        left = np.zeros((height, width, 3), dtype=np.uint8)
        front = np.zeros((height, width, 3), dtype=np.uint8)
        right = np.zeros((height, width, 3), dtype=np.uint8)
        left[:, :36] = (20, 30, 200)
        front[:, 20:60] = (20, 180, 40)
        right[:, 44:] = (200, 40, 20)
        overlaps = [
            {"name": "left_front", "cameras": ["front_left", "front"], "x_range": [20, 35]},
            {"name": "front_right", "cameras": ["front", "front_right"], "x_range": [44, 59]},
        ]
        before = file_revision(CONFIG_DIR / "calibration.yaml")

        with tempfile.TemporaryDirectory() as directory:
            result = generate_front_priority_seam_candidates(
                {"front_left": left, "front": front, "front_right": right},
                overlaps,
                Path(directory),
            )
            self.assertEqual(2, len(result.pair_results))
            self.assertTrue((result.output_dir / "projection_audit.md").exists())
            for pair in result.pair_results:
                candidate = yaml.safe_load(pair.candidate_path.read_text(encoding="utf-8"))
                self.assertEqual("front_priority", candidate["mode"])
                self.assertFalse(candidate["writes_calibration_yaml"])
                self.assertIn("boundary_candidates", candidate)
                self.assertTrue((pair.directory / "previews" / "front_dominant_hard_preview.png").exists())

        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)

    def test_dynamic_simulation_hysteresis_has_no_switches_for_stable_sequence(self) -> None:
        height, width = 24, 60
        left = np.zeros((height, width, 3), dtype=np.uint8)
        front = np.zeros((height, width, 3), dtype=np.uint8)
        left[:, :35] = (20, 20, 200)
        front[:, 20:] = (20, 180, 20)
        sequence = [{"front_left": left, "front": front} for _ in range(4)]
        overlaps = [{"name": "left_front", "cameras": ["front_left", "front"], "x_range": [20, 34]}]

        with tempfile.TemporaryDirectory() as directory:
            result = simulate_front_priority_dynamic_boundaries(
                sequence,
                overlaps,
                Path(directory),
                update_interval_frames=1,
                hysteresis_score_margin=0.10,
            )

            self.assertEqual(0, result.boundary_switch_count)
            self.assertTrue(result.timeline_path.exists())

    def test_projection_audit_is_read_only(self) -> None:
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = write_projection_audit(Path(directory))
            self.assertTrue(path.exists())
            self.assertIn("distortion correction belongs", path.read_text(encoding="utf-8"))
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
