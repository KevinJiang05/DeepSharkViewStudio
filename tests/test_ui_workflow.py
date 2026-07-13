from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from deep_shark_studio.gui.main_window import MainWindow, i18n, rejection_advice
from deep_shark_studio.gui.runtime_workflow import RuntimeViewPreset
from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)


class RuntimeWorkflowModelTests(unittest.TestCase):
    def test_five_named_views_map_to_existing_runtime_boundaries(self) -> None:
        from deep_shark_studio.gui.runtime_workflow import (
            RuntimeViewPreset,
            preset_for_effective_status,
            preset_for_runtime,
            selection_for_preset,
        )

        expected = {
            RuntimeViewPreset.FAR_DEFAULT: (
                StitchRuntimeMode.FAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                False,
                "template",
            ),
            RuntimeViewPreset.B2_VIEW: (
                StitchRuntimeMode.FAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                False,
                "candidate",
            ),
            RuntimeViewPreset.FAR_CUSTOM: (
                StitchRuntimeMode.FAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                True,
                "template",
            ),
            RuntimeViewPreset.NEAR_CURRENT: (
                StitchRuntimeMode.NEAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                False,
                "template",
            ),
            RuntimeViewPreset.NEAR_FISHEYE: (
                StitchRuntimeMode.NEAR_FIELD,
                ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                False,
                "template",
            ),
        }

        self.assertEqual(set(expected), set(RuntimeViewPreset))
        for preset, values in expected.items():
            selection = selection_for_preset(preset)
            self.assertEqual(values, (
                selection.mode,
                selection.projection_source,
                selection.use_far_field_custom_layout,
                selection.live_stitch_strategy,
            ))
            config = RuntimeStitchConfig(
                mode=selection.mode,
                projection_source=selection.projection_source,
                use_far_field_custom_layout=selection.use_far_field_custom_layout,
            )
            self.assertEqual(
                preset,
                preset_for_runtime(config, selection.live_stitch_strategy),
            )
        self.assertEqual(
            RuntimeViewPreset.B2_VIEW,
            preset_for_effective_status("b2_candidate_view"),
        )
        self.assertEqual(
            RuntimeViewPreset.NEAR_FISHEYE,
            preset_for_effective_status("near_field_fisheye_rectilinear"),
        )


class UiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.logger_patch = patch(
            "deep_shark_studio.gui.main_window.create_application_logger",
            return_value=MagicMock(),
        )
        self.logger_patch.start()
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.preview_timer.stop()
        self.window.stream_manager.stop()
        self.window.stitch_processor.shutdown()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.logger_patch.stop()

    def test_root_navigation_is_task_oriented_and_has_no_empty_projection_tab(self) -> None:
        expected_pages = [
            "realtime_monitor",
            "layout_lab",
            "calibration_candidates",
            "project_management",
            "diagnostics_logs",
        ]
        actual_pages = [
            self.window.root_tabs.widget(index).objectName()
            for index in range(self.window.root_tabs.count())
        ]

        self.assertEqual(expected_pages, actual_pages)
        for index in range(self.window.root_tabs.count()):
            self.assertTrue(self.window.root_tabs.tabToolTip(index).strip())

    def test_realtime_monitor_has_one_explicit_five_view_selector(self) -> None:
        expected = [
            "far_default",
            "b2_view",
            "far_custom",
            "near_current",
            "near_fisheye",
        ]
        self.assertEqual(
            expected,
            [
                self.window.runtime_view_combo.itemData(index)
                for index in range(self.window.runtime_view_combo.count())
            ],
        )
        self.assertEqual("far_default", self.window.runtime_view_combo.currentData())
        self.assertIn("Far Default", self.window.runtime_effective_status_label.text())
        self.assertFalse(hasattr(self.window, "candidate_stitch_button"))
        self.assertFalse(hasattr(self.window, "template_stitch_button"))
        summary = self.window.preview_status_summary.text()
        self.assertNotIn("far_field_default", summary)
        self.assertNotIn("current_perspective", summary)
        self.assertIn("Far Default", summary)

    def test_auto_preview_window_starts_once_and_view_selection_retriggers_it(self) -> None:
        with patch.object(
            MainWindow,
            "ensure_selected_runtime_view_live",
            return_value=True,
            create=True,
        ) as ensure_live:
            window = MainWindow(auto_preview_on_view_change=True)
            window.show()
            self.app.processEvents()

            self.assertTrue(window.runtime_apply_button.isHidden())
            ensure_live.assert_called_once_with()

            ensure_live.reset_mock()
            index = window.runtime_view_combo.findData("b2_view")
            window.runtime_view_combo.setCurrentIndex(index)
            self.app.processEvents()
            ensure_live.assert_called_once_with()

            ensure_live.reset_mock()
            window.hide()
            self.app.processEvents()
            index = window.runtime_view_combo.findData("near_current")
            window.runtime_view_combo.setCurrentIndex(index)
            self.app.processEvents()
            ensure_live.assert_not_called()

            window.show()
            self.app.processEvents()
            ensure_live.assert_called_once_with()

            ensure_live.reset_mock()
            window._runtime_view_selection_dirty = False
            window.hide()
            self.app.processEvents()
            window.show()
            self.app.processEvents()
            ensure_live.assert_not_called()

            window.close()
            window.deleteLater()
            self.app.processEvents()

    def test_candidate_and_projection_sources_are_collapsed_until_needed(self) -> None:
        panel = self.window.stitch_runtime_mode_panel
        self.assertTrue(panel.runtime_details_container.isHidden())
        self.assertFalse(panel.runtime_details_toggle.isChecked())
        self.assertFalse(panel.runtime_details_toggle.isEnabled())

        near_index = self.window.runtime_view_combo.findData("near_current")
        self.window.runtime_view_combo.setCurrentIndex(near_index)
        self.app.processEvents()

        self.assertFalse(panel.runtime_details_container.isHidden())
        self.assertFalse(panel.near_source_label.isHidden())
        self.assertTrue(panel.far_source_label.isHidden())
        self.assertTrue(panel.fisheye_source_label.isHidden())

        far_index = self.window.runtime_view_combo.findData("far_custom")
        self.window.runtime_view_combo.setCurrentIndex(far_index)
        self.app.processEvents()

        self.assertFalse(panel.b2_source_label.isHidden())
        self.assertFalse(panel.far_source_label.isHidden())
        self.assertTrue(panel.near_source_label.isHidden())

    def test_selecting_a_view_is_staged_without_changing_effective_runtime(self) -> None:
        original = self.window.runtime_stitch_config
        index = self.window.runtime_view_combo.findData("near_current")

        self.window.runtime_view_combo.setCurrentIndex(index)
        self.app.processEvents()

        self.assertEqual(original, self.window.runtime_stitch_config)
        self.assertEqual(
            StitchRuntimeMode.NEAR_FIELD,
            self.window.selected_runtime_mode(),
        )
        self.assertEqual(
            ProjectionSource.CURRENT_PERSPECTIVE,
            self.window.selected_projection_source(),
        )
        self.assertTrue(self.window._runtime_view_selection_dirty)
        status_text = self.window.runtime_effective_status_label.text()
        self.assertIn(
            self.window.runtime_view_display_name(RuntimeViewPreset.NEAR_CURRENT),
            status_text,
        )
        self.assertIn(
            self.window.runtime_view_display_name(RuntimeViewPreset.FAR_DEFAULT),
            status_text,
        )

    def test_apply_maps_each_named_view_to_the_existing_runtime_contract(self) -> None:
        self.window.live_candidate_directory = Path("b2/candidate")
        self.window.far_field_layout_candidate = SimpleNamespace(
            path=Path("far/candidate.yaml"),
            schema_version=1,
            profile_id="triple_front_panorama",
            projection_source="b2_far_field_candidate",
            output_width_px=1800,
            output_height_px=700,
        )
        pair = SimpleNamespace(
            side_shift_px=40,
            side_visible_fraction=0.3,
            feather_width_px=24,
        )
        self.window.runtime_layout_candidate = SimpleNamespace(
            path=Path("near/candidate.yaml"),
            schema_version=2,
            profile_id="triple_front_panorama",
            projection=SimpleNamespace(
                source=ProjectionSource.CURRENT_PERSPECTIVE,
            ),
            left_pair=pair,
            right_pair=pair,
            camera_adjust={
                "front": SimpleNamespace(
                    scale=1.0,
                    x_offset_px=0,
                    y_offset_px=0,
                )
            },
            warnings=(),
        )
        self.window.runtime_fisheye_intrinsics_source = SimpleNamespace(
            path=Path("fisheye/candidate.yaml")
        )
        expected = {
            "far_default": (
                StitchRuntimeMode.FAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                False,
                "template",
            ),
            "b2_view": (
                StitchRuntimeMode.FAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                False,
                "candidate",
            ),
            "far_custom": (
                StitchRuntimeMode.FAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                True,
                "template",
            ),
            "near_current": (
                StitchRuntimeMode.NEAR_FIELD,
                ProjectionSource.CURRENT_PERSPECTIVE,
                False,
                "template",
            ),
            "near_fisheye": (
                StitchRuntimeMode.NEAR_FIELD,
                ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                False,
                "template",
            ),
        }

        for preset, contract in expected.items():
            with self.subTest(preset=preset):
                index = self.window.runtime_view_combo.findData(preset)
                self.window.runtime_view_combo.setCurrentIndex(index)
                with patch.object(
                    self.window,
                    "commit_live_runtime_selection",
                    return_value=True,
                ) as commit:
                    self.window.apply_stitch_runtime_config()

                config, strategy = commit.call_args.args[:2]
                self.assertEqual(
                    contract,
                    (
                        config.mode,
                        config.projection_source,
                        config.use_far_field_custom_layout,
                        strategy,
                    ),
                )

    def test_portable_package_is_primary_and_legacy_project_is_explicitly_advanced(self) -> None:
        panel = self.window.project_management_panel
        self.assertFalse(panel.portable_group.isHidden())
        self.assertTrue(panel.advanced_container.isHidden())
        self.assertEqual(
            3,
            len(panel.primary_action_buttons),
        )
        self.assertIn("Portable", panel.portable_group.title())
        self.assertNotIn(".dsvs", panel.portable_group.title())

        panel.advanced_toggle.click()
        self.app.processEvents()

        self.assertFalse(panel.advanced_container.isHidden())
        self.assertIn(".dsvs.yaml", panel.legacy_note.text())
        self.assertIn("Legacy", panel.legacy_group.title())

    def test_project_panel_separates_active_package_from_last_action(self) -> None:
        panel = self.window.project_management_panel
        active_path = Path("moved/package/project.dcsvs.yaml")

        self.window.active_project_manifest_path = active_path
        self.window.refresh_project_active_status()
        panel.project_status.setText("validation completed")

        self.assertIn(str(active_path), panel.active_project_status.text())
        self.assertEqual("validation completed", panel.project_status.text())

    def test_english_runtime_status_and_warnings_do_not_fall_back_to_chinese(self) -> None:
        self.window.language = "en"
        self.window.source_contract_log_messages.add("source warning")

        self.window.update_preview_status_summary()
        summary = self.window.preview_status_summary.text()
        warnings = self.window.preview_warning_messages()

        self.assertNotIn("实际拼接链路", summary)
        self.assertTrue(
            any("Source-coordinate warnings" in warning for warning in warnings)
        )
        self.assertIn(
            "Live preview is stopped",
            self.window.stitched_view_notice_text(),
        )

    def test_small_window_keeps_navigation_and_primary_runtime_controls_usable(self) -> None:
        self.window.resize(900, 600)
        self.app.processEvents()

        hint = self.window.minimumSizeHint()
        self.assertLessEqual(hint.width(), 900)
        self.assertLessEqual(hint.height(), 600)
        self.assertTrue(self.window.root_tabs.tabBar().usesScrollButtons())
        self.assertLessEqual(
            self.window.stitch_runtime_mode_panel.sizeHint().height(),
            220,
        )
        self.assertTrue(self.window.runtime_view_combo.isVisible())
        self.assertTrue(self.window.runtime_apply_button.isVisible())

    def test_small_project_page_keeps_legacy_tools_scrollable(self) -> None:
        self.window.resize(900, 600)
        project_index = next(
            index
            for index in range(self.window.root_tabs.count())
            if self.window.root_tabs.widget(index).objectName()
            == "project_management"
        )
        self.window.root_tabs.setCurrentIndex(project_index)
        panel = self.window.project_management_panel
        panel.advanced_toggle.click()
        self.app.processEvents()

        self.assertFalse(panel.advanced_container.isHidden())
        self.assertTrue(panel.open_legacy_project_button.isVisible())
        self.assertTrue(panel.export_runtime_button.isVisible())
        self.assertEqual(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            self.window.project_management_scroll_area.horizontalScrollBarPolicy(),
        )

    def test_five_view_names_and_project_semantics_are_localized(self) -> None:
        keys = [
            "Far Default",
            "B-2 View",
            "Far Custom",
            "Near Current",
            "Near Fisheye",
            "Portable Project Package",
            "Legacy Project File (.dsvs.yaml)",
        ]
        for key in keys:
            self.assertNotEqual(key, i18n("zh_CN", key), key)
            self.assertEqual(key, i18n("en", key), key)

    def test_english_calibration_wizard_does_not_fall_back_to_chinese(self) -> None:
        self.window.language = "en"
        wizard_page = self.window._build_calibration_wizard_page()
        self.app.processEvents()

        self.assertEqual(
            "Step 1/5: Prepare Equipment",
            self.window.wizard_step_title.text(),
        )
        self.assertEqual(
            "Create Capture Session",
            self.window.wizard_create_session_button.text(),
        )
        self.assertEqual(
            "No capture session created or loaded",
            self.window.wizard_session_label.text(),
        )
        self.assertIn(
            "Move the board closer",
            rejection_advice("board area is too small", "en"),
        )
        self.assertIn(
            "让标定板靠近",
            rejection_advice("board area is too small"),
        )
        wizard_page.deleteLater()


if __name__ == "__main__":
    unittest.main()
