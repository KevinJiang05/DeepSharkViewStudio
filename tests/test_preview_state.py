from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from deep_shark_studio.calibration_session import CalibrationSession
from deep_shark_studio.calibration_snapshot import topology_diagnostics
from deep_shark_studio.gui.main_window import (
    CalibrationCandidateDialog,
    MainWindow,
    PreviewContentMode,
    PreviewLayoutMode,
    rejection_advice,
)
from deep_shark_studio.stream_manager import (
    CameraStreamSnapshot,
    STREAM_FAILED,
    STREAM_LIVE,
)


def create_wizard_session(root: Path) -> CalibrationSession:
    return CalibrationSession.create(
        sessions_root=root,
        topology="triple_front_panorama",
        resolution=(1920, 1080),
        source_coordinate_space="raw_frame_pixels",
        source_reference_size=[1920, 1080],
        source_contract_version=1,
        board_definition={
            "type": "chessboard",
            "inner_columns": 11,
            "inner_rows": 8,
            "square_size_mm": 25.0,
        },
        board_confirmed=True,
        captured_at=datetime(
            2026,
            7,
            1,
            16,
            0,
            tzinfo=timezone.utc,
        ),
    )


def set_wizard_counts(
    session: CalibrationSession,
    intrinsic_counts: tuple[int, int, int],
    pair_counts: tuple[int, int],
) -> None:
    zones = [
        f"{row}_{column}"
        for row in ("top", "middle", "bottom")
        for column in ("left", "center", "right")
    ]
    for camera, count in zip(
        ("front_left", "front", "front_right"),
        intrinsic_counts,
    ):
        session.data["intrinsics"][camera]["samples"] = [
            {
                "accepted": True,
                "rejection_reasons": [],
                "warnings": [],
                "metrics": {
                    "coverage_zone": zones[index % len(zones)],
                    "pose_signature": [0.1, 0.1, 0.1, 0.1],
                },
            }
            for index in range(count)
        ]
    pair_zones = [
        f"{position}_{distance}"
        for position in ("left", "center", "right")
        for distance in ("far", "middle", "near")
    ]
    for pair, count in zip(
        ("front_left__front", "front__front_right"),
        pair_counts,
    ):
        session.data["stereo_pairs"][pair]["samples"] = [
            {
                "accepted": True,
                "rejection_reasons": [],
                "warnings": [],
                "pair_coverage_zone": pair_zones[index % len(pair_zones)],
                "pair_pose_signature": [0.1] * 8,
            }
            for index in range(count)
        ]


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

    def test_initial_window_fits_available_desktop(self) -> None:
        available = self.app.primaryScreen().availableGeometry()
        frame = self.window.frameGeometry()

        self.assertLessEqual(frame.width(), available.width())
        self.assertLessEqual(frame.height(), available.height())
        self.assertTrue(available.contains(frame))
        self.assertLessEqual(
            self.window.minimumSizeHint().height(),
            700,
        )

    def test_source_contract_is_shared_by_profile_runtime_editor_and_snapshot(
        self,
    ) -> None:
        profile = self.window.current_stitch_profile()
        snapshot = topology_diagnostics(self.window.calibration_config)

        self.assertEqual(
            profile["source_coordinate_space"],
            self.window.stitcher.source_coordinate_space,
        )
        self.assertEqual(
            profile["source_coordinate_space"],
            self.window.calibration_image_view.source_coordinate_space,
        )
        self.assertEqual(
            profile["source_coordinate_space"],
            snapshot["source_coordinate_space"],
        )
        self.assertEqual(
            profile["source_reference_size"],
            self.window.stitcher.source_reference_size,
        )
        self.assertEqual(
            profile["source_reference_size"],
            self.window.calibration_image_view.source_reference_size,
        )
        self.assertEqual(
            profile["source_reference_size"],
            snapshot["source_reference_size"],
        )

    def test_calibration_workspace_scrolls_instead_of_growing_window(self) -> None:
        self.window.root_tabs.setCurrentIndex(2)
        self.app.processEvents()
        available = self.app.primaryScreen().availableGeometry()

        self.assertLessEqual(
            self.window.frameGeometry().height(),
            available.height(),
        )
        self.assertEqual(
            0,
            self.window.calibration_scroll_area.horizontalScrollBar().maximum(),
        )
        self.assertGreaterEqual(
            self.window.calibration_scroll_area.verticalScrollBar().maximum(),
            0,
        )

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
        profile = self.window.current_stitch_profile()
        height = float(profile["canvas"]["height"])
        for overlap in profile["overlaps"]:
            center_x = sum(overlap["x_range"]) / 2.0
            profile["stitch_points"][overlap["seam"]] = [
                [center_x, 0.0],
                [center_x, height],
            ]
        self.window.stitcher = self.window.create_stitcher()
        frames = {
            "front_left": np.full((1080, 1920, 3), (0, 0, 255), dtype=np.uint8),
            "front": np.full((1080, 1920, 3), (0, 255, 0), dtype=np.uint8),
            "front_right": np.full((1080, 1920, 3), (255, 0, 0), dtype=np.uint8),
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

    def test_geometry_diagnostics_do_not_touch_profile_or_lifecycle(self) -> None:
        self.window.frames = {
            "front_left": np.full(
                (1080, 1920, 3),
                (0, 0, 255),
                dtype=np.uint8,
            ),
            "front": np.full(
                (1080, 1920, 3),
                (0, 255, 0),
                dtype=np.uint8,
            ),
            "front_right": np.full(
                (1080, 1920, 3),
                (255, 0, 0),
                dtype=np.uint8,
            ),
        }
        before = deepcopy(self.window.calibration_config)
        session_id = self.window.preview_session_id

        with (
            patch.object(self.window.stream_manager, "start") as capture_start,
            patch.object(self.window.stream_manager, "stop") as capture_stop,
            patch.object(
                self.window.stitch_processor,
                "configure",
            ) as stitch_configure,
        ):
            self.window.refresh_geometry_diagnostics()

        capture_start.assert_not_called()
        capture_stop.assert_not_called()
        stitch_configure.assert_not_called()
        self.assertEqual(before, self.window.calibration_config)
        self.assertEqual(session_id, self.window.preview_session_id)
        self.assertEqual(6, self.window.geometry_diagnostics_tabs.count())

    def test_pairwise_controls_expose_declared_pairs_without_lifecycle_changes(
        self,
    ) -> None:
        session_id = self.window.preview_session_id
        pair_names = [
            self.window.pairwise_pair_combo.itemData(index)
            for index in range(self.window.pairwise_pair_combo.count())
        ]
        fake_result = object()
        output = self.window.output_dir / "pairwise-test"
        with (
            patch(
                "deep_shark_studio.gui.main_window."
                "run_automatic_pairwise_candidates",
                return_value=[fake_result],
            ),
            patch(
                "deep_shark_studio.gui.main_window.save_pairwise_results",
                return_value=output,
            ),
            patch.object(
                self.window,
                "render_pairwise_candidate_diagnostics",
            ) as render,
            patch(
                "deep_shark_studio.gui.main_window.save_config",
            ) as save_config,
        ):
            self.window.run_pairwise_button.click()

        self.assertEqual(["left_front", "front_right"], pair_names)
        render.assert_called_once_with([fake_result], output)
        save_config.assert_not_called()
        self.assertEqual(session_id, self.window.preview_session_id)
        self.assertEqual({}, self.window.stream_manager.diagnostics())

    def test_session_capture_uses_current_frames_without_runtime_lifecycle(
        self,
    ) -> None:
        session = MagicMock()
        session.capture_intrinsic.return_value = {
            "sample_id": "sample-1",
            "accepted": True,
            "rejection_reasons": [],
            "warnings": [],
        }
        self.window.calibration_session = session
        self.window.preview_content_mode = PreviewContentMode.STILL
        self.window.frames = {
            "front_left": np.zeros((1080, 1920, 3), dtype=np.uint8)
        }
        session_id = self.window.preview_session_id
        with (
            patch.object(self.window.stream_manager, "start") as start,
            patch.object(self.window.stream_manager, "stop") as stop,
            patch.object(self.window, "bump_preview_session") as bump,
            patch.object(
                self.window,
                "refresh_calibration_session_status",
            ),
            patch(
                "deep_shark_studio.gui.main_window.save_config",
            ) as save_config,
        ):
            self.window.capture_calibration_session_sample()

        session.capture_intrinsic.assert_called_once()
        start.assert_not_called()
        stop.assert_not_called()
        bump.assert_not_called()
        save_config.assert_not_called()
        self.assertEqual(session_id, self.window.preview_session_id)

    def test_unconfirmed_board_does_not_create_gui_session(self) -> None:
        self.window.session_board_confirmed.setChecked(False)
        with (
            patch(
                "deep_shark_studio.gui.main_window.CalibrationSession.create",
            ) as create,
            patch(
                "deep_shark_studio.gui.main_window.QMessageBox.information",
            ),
        ):
            self.window.create_calibration_sample_session()

        create.assert_not_called()

    def test_wizard_requires_board_confirmation_before_session(self) -> None:
        self.window.wizard_board_confirmed.setChecked(False)
        with (
            patch.object(
                self.window,
                "create_calibration_sample_session",
            ) as create,
            patch(
                "deep_shark_studio.gui.main_window.QMessageBox.information",
            ),
        ):
            self.window.wizard_create_session()

        create.assert_not_called()

    def test_wizard_enforces_intrinsic_then_pair_minimum_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_wizard_session(Path(directory))
            self.window.calibration_session = session
            self.window.apply_calibration_session_to_controls(session)
            self.window.set_calibration_wizard_step(2)
            with patch(
                "deep_shark_studio.gui.main_window.QMessageBox.information",
            ) as information:
                self.window.wizard_next_step()
            self.assertEqual(
                2,
                self.window.calibration_wizard_stack.currentIndex(),
            )
            information.assert_called_once()

            set_wizard_counts(session, (20, 20, 20), (0, 0))
            self.window.refresh_calibration_wizard()
            self.window.wizard_next_step()
            self.assertEqual(
                3,
                self.window.calibration_wizard_stack.currentIndex(),
            )
            with patch(
                "deep_shark_studio.gui.main_window.QMessageBox.information",
            ):
                self.window.wizard_next_step()
            self.assertEqual(
                3,
                self.window.calibration_wizard_stack.currentIndex(),
            )

            set_wizard_counts(session, (20, 20, 20), (15, 15))
            self.window.refresh_calibration_wizard()
            self.window.wizard_next_step()
            self.assertEqual(
                4,
                self.window.calibration_wizard_stack.currentIndex(),
            )
            self.assertEqual(
                "可进入 B-2 求解",
                self.window.wizard_readiness_label.text(),
            )

    def test_wizard_experimental_b2_entry_is_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_wizard_session(Path(directory))
            set_wizard_counts(session, (20, 20, 20), (15, 15))
            for group in session.data["stereo_pairs"].values():
                for sample in group["samples"]:
                    sample["pair_coverage_zone"] = "center_far"
            self.window.calibration_session = session
            self.window.apply_calibration_session_to_controls(session)
            self.window.set_calibration_wizard_step(4)
            launcher = MagicMock()
            self.window.set_b2_candidate_solver_launcher(launcher)
            session_id = self.window.preview_session_id

            with (
                patch.object(self.window.stream_manager, "start") as start,
                patch.object(self.window.stream_manager, "stop") as stop,
                patch.object(
                    self.window.stitch_processor,
                    "configure",
                ) as configure,
                patch.object(self.window, "bump_preview_session") as bump,
                patch(
                    "deep_shark_studio.gui.main_window.save_config",
                ) as save_config,
            ):
                self.window.wizard_next_button.click()

            self.assertTrue(self.window.wizard_next_button.isEnabled())
            self.assertEqual(
                "实验性候选求解（仅报告）",
                self.window.wizard_next_button.text(),
            )
            self.assertIn(
                "[experimental]",
                self.window.wizard_b2_mode_label.text(),
            )
            launcher.assert_called_once()
            context = launcher.call_args.args[0]
            self.assertEqual("experimental", context["mode"])
            self.assertTrue(context["report_only"])
            self.assertFalse(context["apply_allowed"])
            self.assertIn("outlier_report", context["expected_outputs"])
            start.assert_not_called()
            stop.assert_not_called()
            configure.assert_not_called()
            bump.assert_not_called()
            save_config.assert_not_called()
            self.assertEqual(session_id, self.window.preview_session_id)

    def test_wizard_b2_entry_follows_ready_and_blocked_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_wizard_session(Path(directory))
            self.window.calibration_session = session
            self.window.apply_calibration_session_to_controls(session)
            self.window.set_calibration_wizard_step(4)
            self.assertFalse(self.window.wizard_next_button.isEnabled())
            self.assertEqual(
                "暂不可进入 B-2",
                self.window.wizard_next_button.text(),
            )

            set_wizard_counts(session, (20, 20, 20), (15, 15))
            self.window.refresh_calibration_wizard()
            launcher = MagicMock()
            self.window.set_b2_candidate_solver_launcher(launcher)
            self.assertTrue(self.window.wizard_next_button.isEnabled())
            self.assertEqual(
                "进入 B-2 候选求解",
                self.window.wizard_next_button.text(),
            )

            self.window.wizard_next_button.click()

            context = launcher.call_args.args[0]
            self.assertEqual("official", context["mode"])
            self.assertFalse(context["report_only"])
            self.assertTrue(context["apply_allowed"])

    def test_experimental_candidate_dialog_never_enables_apply(self) -> None:
        candidate = {
            "session_id": "session-test",
            "resolution": [1920, 1080],
            "experimental": True,
            "rig": {"complete": True},
            "intrinsics": {
                camera: {
                    "status": "success",
                    "rms_px": 0.8,
                    "used_sample_count": 20,
                    "outlier_count": 1,
                }
                for camera in ("front_left", "front", "front_right")
            },
            "stereo_pairs": {
                pair: {
                    "status": "success",
                    "rms_px": 1.2,
                    "used_sample_count": 15,
                    "outlier_count": 1,
                    "time_delta_statistics": {"risk_count": 0},
                }
                for pair in (
                    "front_left__front",
                    "front__front_right",
                )
            },
            "readiness": {"quality_issues": ["Pair 覆盖不足"]},
            "virtual_panorama": {
                "projection": "equirectangular_rotation_only",
            },
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "deep_shark_studio.gui.main_window."
                "load_calibration_candidate",
                return_value=(candidate, {}),
            ),
        ):
            dialog = CalibrationCandidateDialog(directory, self.window)

        self.assertFalse(dialog.apply_candidate_button.isEnabled())
        self.assertIn(
            "experimental",
            dialog.apply_candidate_button.toolTip(),
        )
        self.assertIn(
            "为什么不能应用",
            dialog.candidate_safety_notice.text(),
        )
        self.assertIn(
            "Pair 覆盖不足",
            dialog.candidate_safety_notice.text(),
        )
        self.assertIn(
            "光心不重合",
            dialog.candidate_safety_notice.text(),
        )
        dialog.deleteLater()

    def test_preview_status_summary_reports_modes_and_health(self) -> None:
        active = self.window.active_camera_keys()
        self.assertTrue(active)
        self.window.set_preview_content_mode(PreviewContentMode.LIVE)
        snapshots = {
            active[0]: CameraStreamSnapshot(
                key=active[0],
                status=STREAM_LIVE,
                frame_count=12,
            )
        }
        if len(active) > 1:
            snapshots[active[1]] = CameraStreamSnapshot(
                key=active[1],
                status=STREAM_FAILED,
                failed_read_count=3,
                last_error="timeout",
            )
        self.window.stream_snapshots = snapshots

        self.window.update_preview_status_summary()

        text = self.window.preview_status_summary.text()
        self.assertIn("Topology:", text)
        self.assertIn("布局:", text)
        self.assertIn("内容: Live", text)
        self.assertIn("Live 1", text)
        if len(active) > 1:
            self.assertIn("Failed 1", text)

    def test_stitched_notice_and_camera_error_placeholders_are_actionable(
        self,
    ) -> None:
        active = self.window.active_camera_keys()
        self.assertTrue(active)
        self.window.set_preview_content_mode(PreviewContentMode.LIVE)
        self.window.set_preview_layout_mode(PreviewLayoutMode.STITCHED)

        self.assertTrue(self.window.stitched_view_notice.isVisible())
        self.assertIn("近景", self.window.stitched_view_notice.text())

        key = active[0]
        self.window.set_preview_layout_mode(PreviewLayoutMode.FOCUS, key)
        self.window.stream_snapshots = {
            key: CameraStreamSnapshot(
                key=key,
                status=STREAM_FAILED,
                last_error="open timeout",
                failed_read_count=1,
            )
        }
        self.window.refresh_raw_preview(force=True)

        placeholder = self.window.camera_views[key].text()
        self.assertIn("可能原因", placeholder)
        self.assertIn("下一步", placeholder)

    def test_rejection_reason_advice_is_user_actionable(self) -> None:
        self.assertIn(
            "靠近",
            rejection_advice("front: board area is too small."),
        )
        self.assertIn(
            "停稳",
            rejection_advice("time delta 0.2s exceeds threshold"),
        )
        self.assertIn(
            "边缘",
            rejection_advice("Pair pose and position duplicate an accepted sample."),
        )

    def test_ui_closure_does_not_change_formal_calibration_hash(self) -> None:
        calibration_path = Path("configs/calibration.yaml")
        before = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

        self.window.update_preview_status_summary()
        self.window.set_preview_layout_mode(PreviewLayoutMode.STITCHED)
        self.window.set_preview_layout_mode(PreviewLayoutMode.GRID)
        self.window.refresh_calibration_wizard()

        after = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_builtin_b2_solver_does_not_touch_live_lifecycle(self) -> None:
        context = {
            "experimental": True,
            "session_directory": "session-test",
        }
        output = self.window.output_dir / "candidate-test"
        dialog = MagicMock()
        session_id = self.window.preview_session_id
        with (
            patch(
                "deep_shark_studio.gui.main_window."
                "generate_calibration_candidate",
                return_value=output,
            ) as generate,
            patch(
                "deep_shark_studio.gui.main_window."
                "CalibrationCandidateDialog",
                return_value=dialog,
            ),
            patch.object(self.window.stream_manager, "start") as start,
            patch.object(self.window.stream_manager, "stop") as stop,
            patch.object(self.window.stitch_processor, "configure") as configure,
            patch.object(self.window, "bump_preview_session") as bump,
            patch(
                "deep_shark_studio.gui.main_window.save_config",
            ) as save_config,
        ):
            self.window.run_b2_candidate_solver(context)

        generate.assert_called_once_with("session-test")
        dialog.exec.assert_called_once()
        start.assert_not_called()
        stop.assert_not_called()
        configure.assert_not_called()
        bump.assert_not_called()
        save_config.assert_not_called()
        self.assertEqual(session_id, self.window.preview_session_id)

    def test_live_stitch_strategy_switch_does_not_restart_capture(self) -> None:
        self.window.live_candidate_directory = Path(
            "candidate-runtime-test"
        )
        self.window.live_stitch_mode = "template"
        self.window.preview_content_mode = PreviewContentMode.LIVE
        session_id = self.window.preview_session_id
        with (
            patch.object(
                self.window.stitch_processor,
                "configure",
            ) as configure,
            patch.object(self.window.stream_manager, "start") as start,
            patch.object(self.window.stream_manager, "stop") as stop,
            patch(
                "deep_shark_studio.gui.main_window.save_config",
            ) as save_config,
        ):
            self.window.set_live_stitch_mode("candidate")

        self.assertEqual("candidate", self.window.live_stitch_mode)
        self.assertEqual(session_id + 1, self.window.preview_session_id)
        configure.assert_called_once()
        self.assertEqual(
            "candidate",
            configure.call_args.kwargs["processor_mode"],
        )
        self.assertEqual(
            "candidate-runtime-test",
            configure.call_args.kwargs["candidate_directory"],
        )
        start.assert_not_called()
        stop.assert_not_called()
        save_config.assert_not_called()

    def test_wizard_restores_incomplete_session_at_correct_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_wizard_session(Path(directory))
            set_wizard_counts(session, (20, 4, 0), (0, 0))
            session._save()
            with patch(
                "deep_shark_studio.gui.main_window."
                "QFileDialog.getExistingDirectory",
                return_value=str(session.directory),
            ):
                self.window.wizard_load_session()

            self.assertEqual(
                2,
                self.window.calibration_wizard_stack.currentIndex(),
            )
            self.assertIn(
                "front：已接受 4",
                self.window.wizard_intrinsic_progress.toPlainText(),
            )
            self.assertEqual(
                str(session.directory),
                str(self.window.calibration_session.directory),
            )

    def test_wizard_capture_order_does_not_touch_runtime_or_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = create_wizard_session(Path(directory))
            self.window.calibration_session = session
            self.window.apply_calibration_session_to_controls(session)
            self.window.wizard_intrinsic_camera.setCurrentIndex(1)
            session_id = self.window.preview_session_id
            with (
                patch.object(
                    self.window,
                    "capture_calibration_session_sample",
                ) as capture,
                patch.object(self.window.stream_manager, "start") as start,
                patch.object(self.window.stream_manager, "stop") as stop,
                patch.object(self.window, "bump_preview_session") as bump,
                patch(
                    "deep_shark_studio.gui.main_window.save_config",
                ) as save_config,
                patch(
                    "deep_shark_studio.gui.main_window."
                    "QMessageBox.information",
                ),
            ):
                self.window.wizard_capture_intrinsic()
                capture.assert_not_called()
                set_wizard_counts(session, (20, 0, 0), (0, 0))
                self.window.refresh_calibration_wizard()
                self.window.wizard_intrinsic_camera.setCurrentIndex(1)
                self.window.wizard_capture_intrinsic()

            capture.assert_called_once()
            start.assert_not_called()
            stop.assert_not_called()
            bump.assert_not_called()
            save_config.assert_not_called()
            self.assertEqual(session_id, self.window.preview_session_id)

    def test_feather_edit_updates_current_profile_only_on_explicit_save(self) -> None:
        profile = self.window.current_stitch_profile()
        original = profile["composition"]["feather_width"]
        for overlap in profile["overlaps"]:
            seam_position = sum(overlap["x_range"]) / 2.0
            seam_name = overlap["seam"]
            for point_index in (0, 1):
                row = self.window._seam_table_row(seam_name, point_index)
                self.window.seam_table.item(row, 2).setText(
                    str(seam_position)
                )
                self.window.seam_table.item(row, 3).setText(
                    str(0.0 if point_index == 0 else profile["canvas"]["height"])
                )
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

    def test_invalid_seam_is_logged_and_not_persisted(self) -> None:
        profile = self.window.current_stitch_profile()
        overlap = profile["overlaps"][0]
        invalid_x = float(overlap["x_range"][1]) + 100.0
        for point_index in (0, 1):
            row = self.window._seam_table_row(overlap["seam"], point_index)
            self.window.seam_table.item(row, 2).setText(str(invalid_x))
        with (
            patch.object(
                self.window,
                "persist_calibration_config",
            ) as persist,
            patch(
                "deep_shark_studio.gui.main_window.QMessageBox.warning",
            ) as warning,
        ):
            self.window.save_seam_points()

        persist.assert_not_called()
        warning.assert_called_once()
        log_text = self.window.application_log.toPlainText()
        self.assertIn("Seam validation failed", log_text)
        self.assertIn("front_left/front", log_text)
        self.assertIn("outside overlap", log_text)

    def test_center_seams_repairs_editor_without_saving(self) -> None:
        before = deepcopy(
            self.window.current_stitch_profile()["stitch_points"]
        )
        with patch.object(
            self.window,
            "persist_calibration_config",
        ) as persist:
            self.window.center_seams_in_overlaps()

        persist.assert_not_called()
        self.assertEqual(
            before,
            self.window.current_stitch_profile()["stitch_points"],
        )
        points = self.window._stitch_points_from_table()
        for overlap in self.window.current_stitch_profile()["overlaps"]:
            center_x = sum(overlap["x_range"]) / 2.0
            self.assertEqual(
                [center_x, center_x],
                [point[0] for point in points[overlap["seam"]]],
            )

    def test_log_view_and_clipboard_redact_connection_details(self) -> None:
        self.window.log(
            "Failed rtsp://192.168.1.12:554/user=admin&password=secret",
            level="ERROR",
        )

        text = self.window.application_log.toPlainText()
        self.assertIn("[ERROR]", text)
        self.assertIn("rtsp://<redacted>", text)
        self.assertNotIn("192.168.1.12", text)
        self.assertNotIn("secret", text)
        self.window.copy_application_log()
        self.assertEqual(
            text,
            QApplication.clipboard().text(),
        )


if __name__ == "__main__":
    unittest.main()
