from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deep_shark_studio.layout_tuner_cache import (
    LayoutTunerCacheKey,
    frame_content_signature,
    mapping_revision,
    runtime_source_revision,
)


def _key() -> LayoutTunerCacheKey:
    return LayoutTunerCacheKey(
        target="near_field",
        projection_source="current_perspective",
        calibration_revision="calibration-a",
        profile_revision="profile-a",
        projection_revision="projection-a",
        balance=0.6,
        fov_scale=1.0,
        frame_signature="frames-a",
    )


class LayoutTunerCacheKeyTests(unittest.TestCase):
    def test_frame_signature_changes_with_pixels_or_camera_set(self) -> None:
        base = {"front": np.zeros((3, 4, 3), dtype=np.uint8)}
        changed = {"front": base["front"].copy()}
        changed["front"][1, 2] = (1, 2, 3)

        self.assertNotEqual(
            frame_content_signature(base),
            frame_content_signature(changed),
        )
        self.assertNotEqual(
            frame_content_signature(base),
            frame_content_signature(base | {"front_left": base["front"]}),
        )

    def test_selection_match_ignores_frozen_frame_but_not_projection_inputs(self) -> None:
        key = _key()

        self.assertTrue(key.same_selection_as(replace(key, frame_signature="frames-b")))
        self.assertFalse(
            key.same_selection_as(replace(key, projection_source="fisheye_rectilinear_candidate"))
        )
        self.assertFalse(key.same_selection_as(replace(key, balance=0.65)))
        self.assertFalse(
            key.same_selection_as(replace(key, projection_revision="projection-b"))
        )

    def test_mapping_revision_is_order_independent_and_value_sensitive(self) -> None:
        self.assertEqual(
            mapping_revision({"b": 2, "a": 1}),
            mapping_revision({"a": 1, "b": 2}),
        )
        self.assertNotEqual(
            mapping_revision({"a": 1}),
            mapping_revision({"a": 2}),
        )

    def test_runtime_source_revision_detects_same_path_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "candidate.yaml"
            candidate.write_text("schema_version: 2\n", encoding="utf-8")
            before = runtime_source_revision(root)
            candidate.write_text("schema_version: 3\n", encoding="utf-8")

            self.assertNotEqual(before, runtime_source_revision(root))


if __name__ == "__main__":
    unittest.main()
