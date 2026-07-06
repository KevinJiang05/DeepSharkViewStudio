from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from deep_shark_studio.config import CONFIG_DIR, file_revision
from deep_shark_studio.gui.main_window import (
    MainWindow,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSlider,
    NoWheelSpinBox,
    PreviewContentMode,
    PreviewLayoutMode,
)
from deep_shark_studio.gui.panels import StitchRuntimeModePanel
from deep_shark_studio.projection.runtime_providers import ProjectionResult
from deep_shark_studio.seam.layout_candidate_runtime import load_layout_candidate_for_runtime
from deep_shark_studio.seam.layout_tuner import render_front_priority_layout_preview
from deep_shark_studio.seam.near_field_compositor import render_near_field_from_warped
from deep_shark_studio.stitch_runtime_controller import RuntimeStitchController
from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)


def _candidate_data() -> dict:
    return {
        "schema_version": 2,
        "candidate_type": "front_priority_layout",
        "profile_id": "triple_front_panorama",
        "source": {
            "mode": "unit_test",
            "calibration_hash": file_revision(CONFIG_DIR / "calibration.yaml"),
        },
        "camera_adjust": {
            "front_left": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
            "front": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
            "front_right": {"x_offset_px": 0, "y_offset_px": 0, "scale": 1.0},
        },
        "left_pair": {
            "side_shift_px": 40,
            "side_visible_fraction": 0.30,
            "feather_width_px": 24,
        },
        "right_pair": {
            "side_shift_px": 40,
            "side_visible_fraction": 0.30,
            "feather_width_px": 24,
        },
        "output": {"width_px": 1800, "height_px": 700},
        "vertical_safety": {
            "enabled": False,
            "vertical_safe_ratio": 1.0,
            "side_vertical_fade_px": 0,
        },
        "formal_profile_modified": False,
        "writes_calibration_yaml": False,
    }


def _write_candidate(directory: Path) -> Path:
    path = directory / "candidate.yaml"
    path.write_text(yaml.safe_dump(_candidate_data(), sort_keys=False), encoding="utf-8")
    return path


def _warped_images(width: int = 2200, height: int = 700) -> dict[str, np.ndarray]:
    front = np.zeros((height, width, 3), dtype=np.uint8)
    front[:, 650:1550] = (0, 160, 0)
    left = np.zeros_like(front)
    left[:, :850] = (0, 0, 200)
    right = np.zeros_like(front)
    right[:, 1350:] = (200, 0, 0)
    return {"front_left": left, "front": front, "front_right": right}


class FakeStitcher:
    def __init__(self):
        self.process_calls = 0
        self.warp_all_calls = 0
        self.warped = _warped_images()

    def process(self, frames):
        self.process_calls += 1
        return self.warped, np.full((700, 2200, 3), 7, dtype=np.uint8)

    def warp_all(self, frames):
        self.warp_all_calls += 1
        return self.warped


class FakeProjectionProvider:
    def __init__(self, warped: dict[str, np.ndarray]):
        self.warped = warped
        self.project_calls = 0

    def project(self, frames):
        self.project_calls += 1
        return ProjectionResult(
            warped_images=self.warped,
            valid_masks={
                camera: np.any(image != 0, axis=2)
                for camera, image in self.warped.items()
            },
            metadata={
                "projection_source": "current_perspective",
                "provider_name": "FakeProjectionProvider",
                "canvas_size": [2200, 700],
                "uses_intrinsics": False,
                "uses_fisheye": False,
                "uses_remap": False,
                "valid_mask_source": "unit_test",
                "warning_reasons": [],
            },
            timings={
                "projection_total_ms": 1.0,
                "warp_all_ms": 0.5,
            },
        )


class RaisingProjectionProvider:
    def project(self, frames):
        raise AssertionError("Far-field must not use ProjectionProvider.")


class StitchRuntimeTests(unittest.TestCase):
    def test_runtime_config_defaults_to_far_field(self) -> None:
        config = RuntimeStitchConfig()

        self.assertEqual(StitchRuntimeMode.FAR_FIELD, config.mode)
        self.assertEqual(ProjectionSource.CURRENT_PERSPECTIVE, config.projection_source)

    def test_far_field_calls_existing_process(self) -> None:
        stitcher = FakeStitcher()
        result = RuntimeStitchController(stitcher, RuntimeStitchConfig()).process({})

        self.assertEqual(1, stitcher.process_calls)
        self.assertEqual(0, stitcher.warp_all_calls)
        self.assertEqual(StitchRuntimeMode.FAR_FIELD, result.mode)

    def test_near_field_requires_layout_candidate(self) -> None:
        stitcher = FakeStitcher()
        controller = RuntimeStitchController(
            stitcher,
            RuntimeStitchConfig(mode=StitchRuntimeMode.NEAR_FIELD),
        )

        with self.assertRaisesRegex(RuntimeError, "Layout Candidate V2"):
            controller.process({})

    def test_auto_mode_falls_back_to_far_field(self) -> None:
        stitcher = FakeStitcher()
        result = RuntimeStitchController(
            stitcher,
            RuntimeStitchConfig(mode=StitchRuntimeMode.AUTO),
        ).process({})

        self.assertEqual(1, stitcher.process_calls)
        self.assertEqual("auto_fallback_far_field", result.status)
        self.assertTrue(result.warnings)

    def test_fisheye_projection_requires_intrinsics_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stitcher = FakeStitcher()
            controller = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                    layout_candidate_path=_write_candidate(Path(temp)),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "requires"):
                controller.process({})
            self.assertEqual(0, stitcher.warp_all_calls)

    def test_equirectangular_projection_remains_research_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stitcher = FakeStitcher()
            controller = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    projection_source=ProjectionSource.EQUIRECTANGULAR_CANDIDATE,
                    layout_candidate_path=_write_candidate(Path(temp)),
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "research-only"):
                controller.process({})
            self.assertEqual(0, stitcher.warp_all_calls)

    def test_near_field_runtime_processes_warped_images_without_calibration_write(self) -> None:
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as temp:
            stitcher = FakeStitcher()
            result = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    layout_candidate_path=_write_candidate(Path(temp)),
                ),
            ).process({})

            self.assertEqual(0, stitcher.process_calls)
            self.assertEqual(1, stitcher.warp_all_calls)
            self.assertEqual((700, 1800), result.canvas.shape[:2])
            self.assertEqual(StitchRuntimeMode.NEAR_FIELD, result.mode)
            self.assertEqual(
                "CurrentPerspectiveProjectionProvider",
                result.metrics["projection"]["provider_name"],
            )
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)

    def test_near_field_runtime_uses_projection_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stitcher = FakeStitcher()
            provider = FakeProjectionProvider(stitcher.warped)
            result = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    layout_candidate_path=_write_candidate(Path(temp)),
                ),
                projection_provider=provider,
            ).process({})

            self.assertEqual(1, provider.project_calls)
            self.assertEqual(0, stitcher.process_calls)
            self.assertEqual(0, stitcher.warp_all_calls)
            self.assertEqual((700, 1800), result.canvas.shape[:2])
            self.assertEqual("FakeProjectionProvider", result.metrics["projection"]["provider_name"])
            self.assertEqual(1.0, result.metrics["timing"]["projection_total_ms"])
            self.assertEqual(0.5, result.metrics["timing"]["warp_all_ms"])

    def test_near_field_runtime_can_select_fisheye_projection_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stitcher = FakeStitcher()
            provider = FakeProjectionProvider(stitcher.warped)
            result = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                    layout_candidate_path=_write_candidate(Path(temp)),
                ),
                projection_provider=provider,
            ).process({})

            self.assertEqual(1, provider.project_calls)
            self.assertEqual((700, 1800), result.canvas.shape[:2])

    def test_far_field_does_not_use_projection_provider(self) -> None:
        stitcher = FakeStitcher()
        result = RuntimeStitchController(
            stitcher,
            RuntimeStitchConfig(mode=StitchRuntimeMode.FAR_FIELD),
            projection_provider=RaisingProjectionProvider(),
        ).process({})

        self.assertEqual(1, stitcher.process_calls)
        self.assertEqual(0, stitcher.warp_all_calls)
        self.assertEqual(StitchRuntimeMode.FAR_FIELD, result.mode)

    def test_near_field_runtime_canvas_has_no_preview_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = load_layout_candidate_for_runtime(_write_candidate(Path(temp)))
            runtime = render_near_field_from_warped(_warped_images(), candidate)
            preview = render_front_priority_layout_preview(
                _warped_images(),
                candidate.pair_candidates,
                candidate.to_layout_params(),
            )

            runtime_black_label_pixels = int(
                np.count_nonzero(np.all(runtime.canvas[:42, :760] == 0, axis=2))
            )
            preview_black_label_pixels = int(
                np.count_nonzero(np.all(preview.image[:42, :760] == 0, axis=2))
            )
            self.assertLess(runtime_black_label_pixels, 500)
            self.assertGreater(preview_black_label_pixels, 5000)
            self.assertIn("timing", runtime.metrics)
            self.assertEqual(
                "nonzero_pixels_runtime_compatible",
                runtime.metrics["valid_mask_source"],
            )

    def test_near_field_compositor_accepts_valid_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate = load_layout_candidate_for_runtime(_write_candidate(Path(temp)))
            warped = _warped_images()
            masks = {
                camera: np.any(image != 0, axis=2).astype(np.uint8)
                for camera, image in warped.items()
            }

            runtime = render_near_field_from_warped(
                warped,
                candidate,
                valid_masks=masks,
            )

            self.assertEqual((700, 1800), runtime.canvas.shape[:2])
            self.assertEqual("provider_valid_masks", runtime.metrics["valid_mask_source"])

    def test_near_field_runtime_calls_core_compositor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate_path = _write_candidate(Path(temp))
            with patch(
                "deep_shark_studio.seam.near_field_compositor.render_front_priority_layout_core",
                return_value=SimpleNamespace(
                    image=np.zeros((12, 34, 3), dtype=np.uint8),
                    metrics={"runtime_mode": "near_field", "timing": {}},
                    layout_id="unit_core",
                    crop_x=(0, 34),
                    crop_y=(0, 12),
                    vertical_safety_enabled=False,
                    adjusted=SimpleNamespace(metadata={}),
                ),
            ) as core:
                result = RuntimeStitchController(
                    FakeStitcher(),
                    RuntimeStitchConfig(
                        mode=StitchRuntimeMode.NEAR_FIELD,
                        layout_candidate_path=candidate_path,
                    ),
                ).process({})

            self.assertEqual((12, 34), result.canvas.shape[:2])
            core.assert_called_once()
            self.assertIn("valid_masks", core.call_args.kwargs)
            self.assertEqual(
                {"front_left", "front", "front_right"},
                set(core.call_args.kwargs["valid_masks"]),
            )


class StitchRuntimeGuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_runtime_mode_controls_exist(self) -> None:
        self.assertIsInstance(
            self.window.stitch_runtime_mode_panel,
            StitchRuntimeModePanel,
        )
        self.assertTrue(hasattr(self.window, "runtime_mode_combo"))
        self.assertTrue(hasattr(self.window, "runtime_projection_label"))
        self.assertTrue(hasattr(self.window, "runtime_candidate_status"))
        self.assertTrue(hasattr(self.window, "far_field_custom_layout_check"))
        self.assertTrue(hasattr(self.window, "far_field_load_candidate_button"))
        self.assertTrue(hasattr(self.window, "far_field_layout_candidate_status"))
        self.assertTrue(hasattr(self.window, "runtime_apply_button"))

    def test_qgc_output_service_controls_start_direct_canvas_sink(self) -> None:
        self.assertTrue(hasattr(self.window, "qgc_output_start_button"))
        self.assertTrue(hasattr(self.window, "qgc_output_stop_button"))
        self.assertFalse(self.window.qgc_output_stop_button.isEnabled())
        output_modes = [
            self.window.qgc_output_kind.itemData(index)
            for index in range(self.window.qgc_output_kind.count())
        ]
        self.assertEqual(["udp_mpegts", "rtsp"], output_modes)

        self.window.qgc_output_url.setText("rtsp://127.0.0.1:8554/deepshark")
        self.window.canvas = np.full((24, 32, 3), 9, dtype=np.uint8)
        writes: list[np.ndarray] = []

        class FakeSink:
            def __init__(self, config):
                self.config = config

            def write(self, frame):
                writes.append(np.asarray(frame).copy())

            def close(self):
                pass

        with patch("deep_shark_studio.gui.main_window.FfmpegVideoSink", FakeSink):
            self.window.start_qgc_output_service()

        self.assertTrue(self.window.qgc_output_active)
        self.assertFalse(self.window.qgc_output_start_button.isEnabled())
        self.assertTrue(self.window.qgc_output_stop_button.isEnabled())
        self.assertEqual(1, len(writes))
        self.assertTrue(np.array_equal(self.window.canvas, writes[0]))

    def test_qgc_output_writes_current_canvas_after_stitch_result(self) -> None:
        writes: list[np.ndarray] = []

        class FakeSink:
            def write(self, frame):
                writes.append(np.asarray(frame).copy())

            def close(self):
                pass

        self.window.qgc_video_sink = FakeSink()
        self.window.qgc_output_active = True
        canvas = np.full((24, 32, 3), 17, dtype=np.uint8)

        self.window.write_qgc_output_frame(canvas)

        self.assertEqual(1, len(writes))
        self.assertTrue(np.array_equal(canvas, writes[0]))

    def test_qgc_output_priority_reduces_gui_preview_refresh_rate(self) -> None:
        self.window.performance_config["preview_fps"] = 30

        self.window.qgc_output_active = False
        self.assertLess(self.window.effective_preview_interval(), 0.1)

        self.window.qgc_output_active = True
        self.assertGreaterEqual(self.window.effective_preview_interval(), 0.2)

    def test_qgc_output_priority_throttles_canvas_render_only(self) -> None:
        self.window.preview_layout_mode = PreviewLayoutMode.STITCHED
        self.window.qgc_output_active = True
        self.window.last_canvas_render_time = time.perf_counter()

        self.assertFalse(self.window.should_render_canvas_now())

        self.window.last_canvas_render_time -= 0.25
        self.assertTrue(self.window.should_render_canvas_now())

    def test_stitch_fps_limit_allows_higher_qgc_rates(self) -> None:
        self.assertGreaterEqual(self.window.process_fps.maximum(), 60)

    def test_b2_opencl_performance_toggle_is_saved(self) -> None:
        self.assertTrue(hasattr(self.window, "b2_candidate_opencl"))

        self.window.b2_candidate_opencl.setChecked(True)
        self.window.collect_camera_config_from_widgets()

        self.assertTrue(
            self.window.camera_config["performance"]["b2_candidate_opencl"]
        )

    def test_root_workflow_tabs_exist(self) -> None:
        tab_texts = [
            self.window.root_tabs.tabText(index)
            for index in range(self.window.root_tabs.count())
        ]

        self.assertEqual(6, len(tab_texts))
        self.assertTrue(any("实时" in text or "Realtime" in text for text in tab_texts))
        self.assertTrue(any("布局" in text or "Layout" in text for text in tab_texts))
        self.assertTrue(any("标定" in text or "Calibration" in text for text in tab_texts))
        self.assertTrue(any("投影" in text or "Projection" in text for text in tab_texts))
        self.assertTrue(any("项目" in text or "Project" in text for text in tab_texts))
        self.assertTrue(any("诊断" in text or "Diagnostics" in text for text in tab_texts))

    def test_project_package_buttons_exist(self) -> None:
        self.assertTrue(hasattr(self.window, "project_package_export_button"))
        self.assertTrue(hasattr(self.window, "project_package_import_button"))
        self.assertTrue(hasattr(self.window, "project_package_validate_button"))
        texts = {
            self.window.project_package_export_button.text(),
            self.window.project_package_import_button.text(),
            self.window.project_package_validate_button.text(),
        }

        self.assertTrue(any("Package" in text or "项目包" in text for text in texts))

    def test_layout_tuner_is_top_level_workspace_not_preview_subtab(self) -> None:
        preview_tab_texts = [
            self.window.preview_tabs.tabText(index)
            for index in range(self.window.preview_tabs.count())
        ]
        root_tab_texts = [
            self.window.root_tabs.tabText(index)
            for index in range(self.window.root_tabs.count())
        ]

        self.assertFalse(any("Layout" in text or "布局调参" in text for text in preview_tab_texts))
        self.assertTrue(any("Layout" in text or "布局调参" in text for text in root_tab_texts))

    def test_runtime_ui_copy_is_not_misleading(self) -> None:
        auto_item = self.window.runtime_mode_combo.model().item(2)

        self.assertIsNotNone(auto_item)
        self.assertFalse(auto_item.isEnabled())
        projection_sources = [
            self.window.runtime_projection_combo.itemData(index)
            for index in range(self.window.runtime_projection_combo.count())
        ]
        self.assertIn(ProjectionSource.CURRENT_PERSPECTIVE.value, projection_sources)
        self.assertIn(
            ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
            projection_sources,
        )
        self.assertNotIn(ProjectionSource.EQUIRECTANGULAR_CANDIDATE.value, projection_sources)
        self.assertEqual(
            ProjectionSource.CURRENT_PERSPECTIVE.value,
            self.window.runtime_projection_combo.currentData(),
        )
        self.assertIn("Layout", self.window.runtime_load_candidate_button.text())
        self.assertNotIn("Runtime", self.window.runtime_apply_button.text())
        self.assertIn("calibration.yaml", self.window.runtime_apply_button.toolTip())
        self.assertIn("B-2", self.window.candidate_stitch_button.text())
        self.assertFalse(self.window.template_stitch_button.isVisible())
        far_text = self.window.far_field_load_candidate_button.text()
        near_text = self.window.runtime_load_candidate_button.text()
        self.assertTrue("Far-field" in far_text or "远景" in far_text)
        self.assertTrue("Near-field" in near_text or "近景" in near_text)

    def test_runtime_mode_combo_can_select_near_field_before_apply(self) -> None:
        near_index = self.window.runtime_mode_combo.findData(
            StitchRuntimeMode.NEAR_FIELD.value
        )

        self.assertGreaterEqual(near_index, 0)
        self.window.runtime_mode_combo.setCurrentIndex(near_index)

        self.assertEqual(
            StitchRuntimeMode.NEAR_FIELD,
            self.window.selected_runtime_mode(),
        )
        self.assertTrue(self.window.runtime_projection_combo.isEnabled())

    def test_canvas_overlay_uses_current_runtime_mode(self) -> None:
        self.window.preview_content_mode = PreviewContentMode.LIVE
        self.window.runtime_stitch_config = RuntimeStitchConfig(
            mode=StitchRuntimeMode.NEAR_FIELD
        )
        self.window.canvas = np.zeros((24, 32, 3), dtype=np.uint8)

        self.window.render_canvas_view()

        self.assertIn("近景", self.window.canvas_view.overlay_text)
        self.assertNotIn("远景", self.window.canvas_view.overlay_text)

    def test_far_field_layout_tuner_note_says_not_b2(self) -> None:
        index = self.window.layout_tuner_target_combo.findData("far_field")

        self.assertGreaterEqual(index, 0)
        self.window.layout_tuner_target_combo.setCurrentIndex(index)

        self.assertIn("B-2", self.window.layout_tuner_note.text())

    def test_spin_boxes_do_not_use_mouse_wheel_classes(self) -> None:
        spin_boxes = self.window.findChildren(NoWheelSpinBox)
        double_spin_boxes = self.window.findChildren(NoWheelDoubleSpinBox)

        self.assertGreater(len(spin_boxes), 0)
        self.assertGreater(len(double_spin_boxes), 0)

    def test_layout_tuner_sliders_do_not_use_mouse_wheel_classes(self) -> None:
        sliders = self.window.findChildren(NoWheelSlider)

        self.assertGreater(len(sliders), 0)

    def test_combo_boxes_do_not_use_mouse_wheel_classes(self) -> None:
        combo_boxes = self.window.findChildren(NoWheelComboBox)

        self.assertGreater(len(combo_boxes), 0)

    def test_loading_candidate_updates_status_label(self) -> None:
        before = file_revision(CONFIG_DIR / "calibration.yaml")
        with tempfile.TemporaryDirectory() as temp:
            path = _write_candidate(Path(temp))
            self.window.load_runtime_layout_candidate_path(path)

            text = self.window.runtime_candidate_status.text()
            self.assertIn("schema_version: 2", text)
            self.assertIn("profile_id: triple_front_panorama", text)
            self.assertIn("left_pair:", text)
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
