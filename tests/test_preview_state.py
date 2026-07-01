from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from deep_shark_studio.gui.main_window import (
    MainWindow,
    PreviewContentMode,
    PreviewLayoutMode,
)
from deep_shark_studio.stream_manager import CameraStreamSnapshot, STREAM_FAILED


class PreviewStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.preview_timer.stop()
        self.window.stream_manager.stop()
        self.window.stitch_processor.shutdown()
        self.window.deleteLater()
        self.app.processEvents()

    def test_layout_changes_do_not_touch_capture_or_session(self) -> None:
        session_id = self.window.preview_session_id
        camera_id = self.window.active_camera_keys()[0]

        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, camera_id)
        self.assertEqual(PreviewLayoutMode.FOCUS, self.window.preview_layout_mode)
        self.assertEqual(camera_id, self.window.focused_camera_id)
        self.assertTrue(self.window.preview_tabs.isHidden())
        self.assertFalse(self.window.camera_views[camera_id].isHidden())

        self.window.on_raw_view_double_clicked(camera_id)
        self.assertEqual(PreviewLayoutMode.GRID, self.window.preview_layout_mode)

        self.window.set_preview_layout_mode(PreviewLayoutMode.STITCHED)
        self.assertEqual(PreviewLayoutMode.STITCHED, self.window.preview_layout_mode)
        self.assertEqual(
            self.window.canvas_tab_index,
            self.window.preview_tabs.currentIndex(),
        )

        self.window.set_preview_content_mode(PreviewContentMode.STOPPED)
        self.assertEqual(PreviewLayoutMode.STITCHED, self.window.preview_layout_mode)
        self.assertEqual(session_id, self.window.preview_session_id)
        self.assertEqual({}, self.window.stream_manager.diagnostics())

    def test_root_tab_round_trip_preserves_preview_state(self) -> None:
        camera_id = self.window.active_camera_keys()[0]
        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, camera_id)
        session_id = self.window.preview_session_id

        self.window.root_tabs.setCurrentIndex(2)
        self.app.processEvents()
        self.window.root_tabs.setCurrentIndex(0)
        self.app.processEvents()

        self.assertEqual(PreviewLayoutMode.FOCUS, self.window.preview_layout_mode)
        self.assertEqual(camera_id, self.window.focused_camera_id)
        self.assertEqual(session_id, self.window.preview_session_id)
        self.assertEqual({}, self.window.stream_manager.diagnostics())

    def test_focus_is_invalidated_only_when_camera_is_not_active(self) -> None:
        active = self.window.active_camera_keys()
        inactive = next(key for key in self.window.camera_views if key not in active)

        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, inactive)

        self.assertEqual(PreviewLayoutMode.GRID, self.window.preview_layout_mode)
        self.assertIsNone(self.window.focused_camera_id)

    def test_control_bar_reflects_grid_focus_and_stitched(self) -> None:
        camera_id = self.window.active_camera_keys()[0]
        display_name = self.window.camera_display_name(camera_id)
        self.window.set_preview_content_mode(PreviewContentMode.LIVE)

        self.window.set_preview_layout_mode(PreviewLayoutMode.GRID)
        self.assertEqual(
            self.window.t("Multi-camera monitoring"),
            self.window.preview_mode_label.text(),
        )
        self.assertTrue(self.window.grid_view_button.isChecked())
        self.assertTrue(self.window.back_to_grid_button.isHidden())

        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, camera_id)
        self.assertEqual(
            self.window.t("Single-camera view: {camera}", camera=display_name),
            self.window.preview_mode_label.text(),
        )
        self.assertFalse(self.window.back_to_grid_button.isHidden())

        self.window.set_preview_layout_mode(PreviewLayoutMode.STITCHED)
        self.assertEqual(
            self.window.t("Stitched View"),
            self.window.preview_mode_label.text(),
        )
        self.assertTrue(self.window.stitched_view_button.isChecked())
        self.assertTrue(self.window.back_to_grid_button.isHidden())

    def test_failed_focus_replaces_old_frame_with_error_placeholder(self) -> None:
        camera_id = self.window.active_camera_keys()[0]
        display_name = self.window.camera_display_name(camera_id)
        self.window.frames[camera_id] = np.full((20, 20, 3), 255, dtype=np.uint8)
        self.window.stream_snapshots[camera_id] = CameraStreamSnapshot(
            key=camera_id,
            status=STREAM_FAILED,
            last_error="simulated RTSP failure",
        )
        self.window.set_preview_content_mode(PreviewContentMode.LIVE)
        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, camera_id)

        text = self.window.camera_views[camera_id].text()
        self.assertIn(display_name, text)
        self.assertIn(self.window.t("Video stream unavailable"), text)
        self.assertIn("simulated RTSP failure", text)
        self.assertIn(self.window.t("Use Back to Grid to leave this view."), text)
        pixmap = self.window.camera_views[camera_id].pixmap()
        self.assertTrue(pixmap is None or pixmap.isNull())
        self.assertFalse(self.window.back_to_grid_button.isHidden())

    def test_stopped_and_still_labels_preserve_layout(self) -> None:
        self.window.set_preview_layout_mode(PreviewLayoutMode.STITCHED)

        self.window.set_preview_content_mode(PreviewContentMode.STOPPED)
        self.assertEqual(PreviewLayoutMode.STITCHED, self.window.preview_layout_mode)
        self.assertEqual(
            self.window.t("Live preview stopped"),
            self.window.preview_mode_label.text(),
        )
        self.assertEqual(
            self.window.t("Live preview stopped"),
            self.window.canvas_view.text(),
        )

        self.window.set_preview_content_mode(PreviewContentMode.STILL)
        self.assertEqual(PreviewLayoutMode.STITCHED, self.window.preview_layout_mode)
        self.assertEqual(
            self.window.t("Static preview"),
            self.window.preview_mode_label.text(),
        )

    def test_control_buttons_do_not_touch_runtime_lifecycle(self) -> None:
        camera_id = self.window.active_camera_keys()[0]
        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, camera_id)
        session_id = self.window.preview_session_id

        with (
            patch.object(self.window.stream_manager, "start") as capture_start,
            patch.object(self.window.stream_manager, "stop") as capture_stop,
            patch.object(self.window.stitch_processor, "configure") as stitch_configure,
            patch.object(self.window, "bump_preview_session") as bump_session,
        ):
            self.window.back_to_grid_button.click()
            self.window.stitched_view_button.click()
            self.window.grid_view_button.click()

        capture_start.assert_not_called()
        capture_stop.assert_not_called()
        stitch_configure.assert_not_called()
        bump_session.assert_not_called()
        self.assertEqual(session_id, self.window.preview_session_id)

    def test_camera_count_enables_profile_cameras_without_saving(self) -> None:
        declared_keys = [
            key
            for key in self.window.current_stitch_profile()["active_cameras"]
            if key in self.window.camera_rows
        ]
        target_count = len(declared_keys)

        with patch(
            "deep_shark_studio.gui.main_window.save_config"
        ) as save_config:
            self.window.camera_count.setValue(2)
            self.assertEqual(
                declared_keys[:2],
                self.window.active_camera_keys(),
            )
            self.window.camera_count.setValue(target_count)

        save_config.assert_not_called()
        self.assertEqual(
            declared_keys,
            self.window.active_camera_keys(),
        )

    def test_start_and_close_do_not_implicitly_persist_config(self) -> None:
        with (
            patch.object(
                self.window,
                "persist_camera_config_from_widgets",
            ) as persist,
            patch.object(self.window.stream_manager, "start"),
            patch.object(
                self.window.stream_manager,
                "has_active_workers",
                return_value=True,
            ),
        ):
            self.window.start_live_preview()
            self.window.close()
            self.app.processEvents()

        persist.assert_not_called()

    def test_reload_runtime_state_refreshes_camera_editor_widgets(self) -> None:
        calibration = deepcopy(self.window.calibration_config)
        cameras = deepcopy(self.window.camera_config)
        cameras["active_camera_count"] = 3
        cameras["cameras"]["front"]["enabled"] = True
        cameras["cameras"]["front"]["source"] = "rtsp://reload-test/front"

        def load_config(name: str):
            return calibration if name == "calibration.yaml" else cameras

        with (
            patch(
                "deep_shark_studio.gui.main_window.load_config",
                side_effect=load_config,
            ),
            patch(
                "deep_shark_studio.gui.main_window.config_revision",
                return_value="test-revision",
            ),
            patch(
                "deep_shark_studio.gui.main_window.save_config",
            ) as save_config,
        ):
            self.window.reload_runtime_state()

        save_config.assert_not_called()
        self.assertTrue(self.window.camera_rows["front"].enabled.isChecked())
        self.assertEqual(
            "rtsp://reload-test/front",
            self.window.camera_rows["front"].source.text(),
        )
        self.assertEqual(3, self.window.camera_count.value())
        self.assertEqual(
            ["front_left", "front", "front_right"],
            self.window.active_camera_keys(),
        )

    def test_static_preview_uses_current_triple_profile(self) -> None:
        frames = {
            "front_left": np.full((540, 960, 3), (0, 0, 255), dtype=np.uint8),
            "front": np.full((540, 960, 3), (0, 255, 0), dtype=np.uint8),
            "front_right": np.full((540, 960, 3), (255, 0, 0), dtype=np.uint8),
        }

        with patch(
            "deep_shark_studio.gui.main_window.load_images_from_directory",
            return_value=frames,
        ):
            self.window.run_still_preview()

        self.assertEqual(
            PreviewContentMode.STILL,
            self.window.preview_content_mode,
        )
        self.assertEqual(
            "triple_front_panorama",
            self.window.stitcher.topology_name,
        )
        self.assertEqual((700, 2200, 3), self.window.canvas.shape)

    def test_calibration_page_displays_triple_topology_diagnostics(self) -> None:
        text = self.window.topology_diagnostics_label.text()

        self.assertIn("Topology: triple_front_panorama", text)
        self.assertIn("front_left <-> front", text)
        self.assertIn("front <-> front_right", text)
        self.assertIn("initial_template", text)
        self.assertEqual(160, self.window.feather_width.value())

    def test_feather_edit_updates_current_profile_only_on_explicit_save(self) -> None:
        profile = self.window.current_stitch_profile()
        original = profile["composition"]["feather_width"]
        self.window.feather_width.setValue(original + 20)

        self.assertEqual(original, profile["composition"]["feather_width"])
        with patch.object(
            self.window,
            "persist_calibration_config",
            return_value=True,
        ):
            self.window.save_seam_points()

        self.assertEqual(original + 20, profile["composition"]["feather_width"])
        self.assertTrue(
            all(
                overlap["feather_width"] == original + 20
                for overlap in profile["overlaps"]
            )
        )
        self.assertEqual(
            "field_adjusted",
            profile["calibration_origin"]["feather"],
        )


if __name__ == "__main__":
    unittest.main()
