from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.seam.runner import generate_pairwise_seam_candidates


class SeamCandidateRunnerTests(unittest.TestCase):
    def test_candidate_save_does_not_modify_calibration_yaml(self) -> None:
        height, width = 48, 96
        left = np.zeros((height, width, 3), dtype=np.uint8)
        front = np.zeros((height, width, 3), dtype=np.uint8)
        left[:, :60] = (20, 80, 180)
        front[:, 32:] = (30, 90, 170)
        overlaps = [
            {
                "name": "left_front",
                "cameras": ["front_left", "front"],
                "x_range": [32, 59],
                "seam": "left_front",
            }
        ]
        before = file_revision(CONFIG_DIR / "calibration.yaml")

        with tempfile.TemporaryDirectory() as directory:
            result = generate_pairwise_seam_candidates(
                {"front_left": left, "front": front},
                overlaps,
                Path(directory),
                feather_widths=[0, 16, 24, 32],
            )

            self.assertEqual(1, len(result.pair_results))
            pair = result.pair_results[0]
            self.assertTrue(pair.candidate_path.exists())
            self.assertTrue((pair.directory / "previews" / "cost_heatmap.png").exists())
            self.assertTrue((pair.directory / "previews" / "hard_seam_preview.png").exists())
            candidate = yaml.safe_load(pair.candidate_path.read_text(encoding="utf-8"))
            self.assertFalse(candidate["formal_profile_modified"])
            self.assertFalse(candidate["writes_calibration_yaml"])
            self.assertEqual([0, 16, 24, 32], candidate["feather_width_candidates"])

        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
