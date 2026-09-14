from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

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
from deep_shark_studio.stream_manager import CameraStreamSnapshot, STREAM_LIVE


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


def _write_candidate(directory: Path, data: dict | None = None) -> Path:
    path = directory / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(data or _candidate_data(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _write_fisheye_layout_candidate(directory: Path) -> Path:
    source_dir = directory / "fisheye_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = {
        camera: {
            "status": "success",
            "model": "opencv_fisheye",
            "resolution": [32, 24],
            "camera_matrix": [
                [24.0, 0.0, 16.0],
                [0.0, 24.0, 12.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
            "rms_px": 0.1,
            "accepted_input_count": 8,
        }
        for camera in ("front_left", "front", "front_right")
    }
    source_path = source_dir / "candidate.yaml"
    source_path.write_text(
        yaml.safe_dump(
            {
                "format": "DeepSharkFisheyeCalibrationCandidate",
                "topology": "triple_front_panorama",
                "resolution": [32, 24],
                "experimental": True,
                "apply_allowed": False,
                "intrinsics": intrinsics,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data = _candidate_data()
    data["schema_version"] = 3
    data["projection"] = {
        "source": "fisheye_rectilinear",
        "intrinsics_source_path": str(source_path),
        "balance": 0.6,
        "fov_scale": 1.0,
    }
    return _write_candidate(directory, data)


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
        self.assertEqual("far_field_default", result.status)

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
            candidate_path = _write_fisheye_layout_candidate(Path(temp))
            controller = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                    layout_candidate_path=candidate_path,
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
                "near_field_current_perspective",
                result.status,
            )
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
            candidate_path = _write_fisheye_layout_candidate(Path(temp))
            result = RuntimeStitchController(
                stitcher,
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                    layout_candidate_path=candidate_path,
                ),
                projection_provider=provider,
            ).process({})

            self.assertEqual(1, provider.project_calls)
            self.assertEqual((700, 1800), result.canvas.shape[:2])
            self.assertEqual(
                "near_field_fisheye_rectilinear",
                result.status,
            )

    def test_near_field_rejects_selected_current_with_fisheye_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate_path = _write_fisheye_layout_candidate(Path(temp))

            with self.assertRaisesRegex(RuntimeError, "does not match selected"):
                RuntimeStitchController(
                    FakeStitcher(),
                    RuntimeStitchConfig(
                        mode=StitchRuntimeMode.NEAR_FIELD,
                        projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
                        layout_candidate_path=candidate_path,
                    ),
                )

    def test_near_field_rejects_selected_fisheye_with_current_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            candidate_path = _write_candidate(Path(temp))

            with self.assertRaisesRegex(RuntimeError, "does not match selected"):
                RuntimeStitchController(
                    FakeStitcher(),
                    RuntimeStitchConfig(
                        mode=StitchRuntimeMode.NEAR_FIELD,
                        projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
                        layout_candidate_path=candidate_path,
                    ),
                )

    def test_near_field_rejects_candidate_crop_larger_than_projected_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            small_warped = _warped_images(width=100, height=40)
            provider = FakeProjectionProvider(small_warped)
            controller = RuntimeStitchController(
                FakeStitcher(),
                RuntimeStitchConfig(
                    mode=StitchRuntimeMode.NEAR_FIELD,
                    layout_candidate_path=_write_candidate(Path(temp)),
                ),
                projection_provider=provider,
            )

            with self.assertRaisesRegex(RuntimeError, "exceeds.*projected canvas"):
                controller.process({})

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
        self.logger_patch = patch(
            "deep_shark_studio.gui.main_window.create_application_logger",
            return_value=MagicMock(),
        )
        self.logger_patch.start()
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.logger_patch.stop()

    def test_runtime_mode_controls_exist(self) -> None:
        self.assertIsInstance(
            self.window.stitch_runtime_mode_panel,
            StitchRuntimeModePanel,
        )
        self.assertTrue(hasattr(self.window, "runtime_view_combo"))
        self.assertFalse(hasattr(self.window, "runtime_mode_combo"))
        self.assertFalse(hasattr(self.window, "runtime_projection_combo"))
        self.assertTrue(hasattr(self.window, "runtime_candidate_status"))
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
                self.opened = False
                self.frames_written = 0

            def preflight(self):
                return "fake ffmpeg"

            def open(self, width, height, fps):
                self.opened = True

            def write(self, frame):
                writes.append(np.asarray(frame).copy())
                self.frames_written += 1
                return True

            def status(self):
                return SimpleNamespace(
                    state="running",
                    running=True,
                    pid=1234,
                    frames_submitted=self.frames_written,
                    frames_written=self.frames_written,
                    frames_dropped=0,
                    pending_frames=0,
                    restart_count=0,
                    last_restart_reason="",
                    last_error="",
                    last_stderr_line="",
                    exit_code=None,
                )

            def close(self):
                self.opened = False

        with patch("deep_shark_studio.gui.main_window.FfmpegVideoSink", FakeSink):
            self.window.start_qgc_output_service()

        self.assertTrue(self.window.qgc_output_active)
        self.assertFalse(self.window.qgc_output_start_button.isEnabled())
        self.assertTrue(self.window.qgc_output_stop_button.isEnabled())
        self.assertEqual(1, len(writes))
        self.assertTrue(np.array_equal(self.window.canvas, writes[0]))

    def test_qgc_output_does_not_claim_running_without_a_canvas(self) -> None:
        self.window.canvas = None

        class FakeSink:
            def __init__(self, config):
                self.config = config

            def close(self):
                return None

        with patch(
            "deep_shark_studio.gui.main_window.FfmpegVideoSink",
            FakeSink,
        ):
            self.window.start_qgc_output_service()

        self.assertFalse(self.window.qgc_output_active)
        self.assertTrue(self.window.qgc_output_start_button.isEnabled())
        self.assertFalse(self.window.qgc_output_stop_button.isEnabled())
        self.assertIn("canvas", self.window.qgc_output_last_error.lower())

    def test_qgc_status_uses_effective_start_config_and_locks_widgets(self) -> None:
        self.window.canvas = np.zeros((24, 32, 3), dtype=np.uint8)
        self.window.qgc_output_url.setText(
            "udp://127.0.0.1:5600?pkt_size=1316"
        )

        class FakeSink:
            def __init__(self, config):
                self.config = config

            def preflight(self):
                return "fake ffmpeg"

            def open(self, width, height, fps):
                return None

            def write(self, frame):
                return True

            def status(self):
                return SimpleNamespace(
                    state="running",
                    running=True,
                    pid=4321,
                    frames_submitted=4,
                    frames_written=3,
                    frames_dropped=1,
                    pending_frames=0,
                    restart_count=0,
                    last_restart_reason="",
                    last_error="",
                    last_stderr_line="",
                    exit_code=None,
                )

            def close(self):
                return None

        with patch(
            "deep_shark_studio.gui.main_window.FfmpegVideoSink",
            FakeSink,
        ):
            self.window.start_qgc_output_service()

        self.window.qgc_output_url.setText("udp://127.0.0.1:9999")
        self.window.refresh_qgc_output_status(force=True)

        status_text = self.window.qgc_output_service_status.text()
        self.assertIn("127.0.0.1:5600", status_text)
        self.assertNotIn("127.0.0.1:9999", status_text)
        self.assertFalse(self.window.qgc_output_url.isEnabled())
        self.assertIn("written 3", status_text)
        self.assertIn("dropped 1", status_text)

        self.window.stop_qgc_output_service()
        self.assertTrue(self.window.qgc_output_url.isEnabled())

    def test_qgc_async_failure_updates_actual_ui_state(self) -> None:
        self.window.canvas = np.zeros((24, 32, 3), dtype=np.uint8)
        state = {"value": "running"}

        class FakeSink:
            def __init__(self, config):
                self.config = config

            def preflight(self):
                return "fake ffmpeg"

            def open(self, width, height, fps):
                return None

            def write(self, frame):
                return True

            def status(self):
                failed = state["value"] == "failed"
                return SimpleNamespace(
                    state=state["value"],
                    running=not failed,
                    pid=None if failed else 5678,
                    frames_submitted=2,
                    frames_written=1,
                    frames_dropped=1,
                    pending_frames=0,
                    restart_count=3 if failed else 0,
                    last_restart_reason="broken pipe" if failed else "",
                    last_error="receiver unavailable" if failed else "",
                    last_stderr_line="connection refused" if failed else "",
                    exit_code=1 if failed else None,
                )

            def close(self):
                return None

        with patch(
            "deep_shark_studio.gui.main_window.FfmpegVideoSink",
            FakeSink,
        ):
            self.window.start_qgc_output_service()

        state["value"] = "failed"
        self.window.refresh_qgc_output_status(force=True)

        self.assertFalse(self.window.qgc_output_active)
        self.assertTrue(self.window.qgc_output_start_button.isEnabled())
        self.assertIn("receiver unavailable", self.window.qgc_output_service_status.text())
        self.assertIn("exit 1", self.window.qgc_output_service_status.text())

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
        self.window.performance_config["preview_fps"] = 30
        self.window.preview_layout_mode = PreviewLayoutMode.STITCHED
        self.window.qgc_output_active = True
        self.window.last_canvas_render_time = time.perf_counter()

        with patch.object(self.window, "should_render_canvas", return_value=True):
            self.assertFalse(self.window.should_render_canvas_now())

            self.window.last_canvas_render_time -= 0.25
            self.assertTrue(self.window.should_render_canvas_now())

    def test_canvas_render_is_gated_by_visible_realtime_canvas_page(self) -> None:
        self.window.show()
        self.app.processEvents()
        self.window.root_tabs.setCurrentIndex(0)
        self.window.set_preview_layout_mode(PreviewLayoutMode.STITCHED)

        self.assertTrue(self.window.should_render_canvas())

        previous_render_time = self.window.last_canvas_render_time
        self.window.root_tabs.setCurrentIndex(1)
        self.assertFalse(self.window.should_render_canvas())
        self.assertFalse(self.window.should_render_canvas_now(force=True))
        self.assertEqual(previous_render_time, self.window.last_canvas_render_time)

        self.window.root_tabs.setCurrentIndex(0)
        self.window.preview_tabs.hide()
        self.assertFalse(self.window.should_render_canvas())

        self.window.preview_tabs.show()
        self.window.preview_tabs.setCurrentIndex(self.window.warped_tab_index)
        self.assertFalse(self.window.should_render_canvas())

    def test_canvas_render_respects_preview_fps_without_qgc(self) -> None:
        self.window.performance_config["preview_fps"] = 10
        self.window.qgc_output_active = False
        self.window.last_canvas_render_time = time.perf_counter()

        with patch.object(self.window, "should_render_canvas", return_value=True):
            self.assertFalse(self.window.should_render_canvas_now())

            self.window.last_canvas_render_time -= 0.11
            self.assertTrue(self.window.should_render_canvas_now())

    def test_preview_summary_reports_worker_telemetry_and_frame_age(self) -> None:
        self.window.language = "en"
        active_key = self.window.active_camera_keys()[0]
        self.window.preview_content_mode = PreviewContentMode.LIVE
        self.window.frames = {
            active_key: np.zeros((8, 8, 3), dtype=np.uint8),
        }
        self.window.stream_snapshots = {
            active_key: CameraStreamSnapshot(
                key=active_key,
                status=STREAM_LIVE,
                frame_timestamp=999.6,
                frame_count=4,
                thread_alive=True,
            ),
        }
        telemetry = replace(
            self.window.stitch_processor.state(),
            submitted=12,
            dropped=3,
            completed=8,
            failed=1,
            rolling_result_fps=6.2,
            rolling_elapsed_p50_ms=11.2,
            rolling_elapsed_p95_ms=18.7,
        )

        with (
            patch.object(
                self.window.stitch_processor,
                "state",
                return_value=telemetry,
            ),
            patch(
                "deep_shark_studio.gui.main_window.time.time",
                return_value=1000.0,
            ),
        ):
            self.window.update_preview_status_summary()

        summary = self.window.preview_status_summary.text()
        self.assertIn(
            "Worker: submitted 12 / dropped 3 / completed 8 / failed 1",
            summary,
        )
        self.assertIn("result FPS 6.2", summary)
        self.assertIn("p50 11.2 ms / p95 18.7 ms", summary)
        self.assertIn("max frame age 0.4 s", summary)

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

        self.assertEqual(5, len(tab_texts))
        self.assertEqual(
            [
                "realtime_monitor",
                "layout_lab",
                "calibration_candidates",
                "project_management",
                "diagnostics_logs",
            ],
            [
                self.window.root_tabs.widget(index).objectName()
                for index in range(self.window.root_tabs.count())
            ],
        )

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
        self.assertFalse(any("Layout" in text or "布局调参" in text for text in preview_tab_texts))
        self.assertIn(
            "layout_lab",
            [
                self.window.root_tabs.widget(index).objectName()
                for index in range(self.window.root_tabs.count())
            ],
        )

    def test_runtime_ui_copy_is_not_misleading(self) -> None:
        view_presets = [
            self.window.runtime_view_combo.itemData(index)
            for index in range(self.window.runtime_view_combo.count())
        ]
        self.assertEqual(
            [
                "far_default",
                "b2_view",
                "far_custom",
                "near_current",
                "near_fisheye",
            ],
            view_presets,
        )
        self.assertEqual(
            "far_default",
            self.window.runtime_view_combo.currentData(),
        )
        self.assertTrue(
            "Layout" in self.window.runtime_load_candidate_button.text()
            or "布局" in self.window.runtime_load_candidate_button.text()
        )
        self.assertIn("calibration.yaml", self.window.runtime_apply_button.toolTip())
        self.assertFalse(hasattr(self.window, "candidate_stitch_button"))
        self.assertFalse(hasattr(self.window, "template_stitch_button"))
        far_text = self.window.far_field_load_candidate_button.text()
        near_text = self.window.runtime_load_candidate_button.text()
        self.assertTrue("Far" in far_text or "远景" in far_text)
        self.assertTrue("Near" in near_text or "近景" in near_text)

    def test_runtime_view_can_select_near_field_before_apply(self) -> None:
        near_index = self.window.runtime_view_combo.findData("near_current")

        self.assertGreaterEqual(near_index, 0)
        self.window.runtime_view_combo.setCurrentIndex(near_index)

        self.assertEqual(
            StitchRuntimeMode.NEAR_FIELD,
            self.window.selected_runtime_mode(),
        )
        self.assertEqual(
            ProjectionSource.CURRENT_PERSPECTIVE,
            self.window.selected_projection_source(),
        )

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

    def test_layout_tuner_target_switch_invalidates_cached_projection(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        self.window.layout_tuner_warped = {"front": image}
        self.window.layout_tuner_valid_masks = {
            "front": np.ones(image.shape[:2], dtype=bool)
        }
        self.window.layout_tuner_cache_key = (
            self.window.build_layout_tuner_cache_key("frozen-frame")
        )
        self.window.layout_tuner_save_button.setEnabled(True)

        index = self.window.layout_tuner_target_combo.findData("far_field")
        self.window.layout_tuner_target_combo.setCurrentIndex(index)

        self.assertIsNone(self.window.layout_tuner_cache_key)
        self.assertEqual({}, self.window.layout_tuner_warped)
        self.assertFalse(self.window.layout_tuner_save_button.isEnabled())

    def test_layout_tuner_projection_parameter_change_invalidates_cache(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        self.window.layout_tuner_warped = {"front": image}
        self.window.layout_tuner_valid_masks = {
            "front": np.ones(image.shape[:2], dtype=bool)
        }
        self.window.layout_tuner_cache_key = (
            self.window.build_layout_tuner_cache_key("frozen-frame")
        )

        self.window.layout_tuner_fisheye_balance.setValue(0.65)

        self.assertIsNone(self.window.layout_tuner_cache_key)
        self.assertEqual({}, self.window.layout_tuner_warped)

    def test_layout_tuner_capture_requires_raw_frames_not_legacy_warp_cache(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        self.window.frames = {}
        self.window.warped = {
            camera: image.copy()
            for camera in ("front_left", "front", "front_right")
        }

        with patch(
            "deep_shark_studio.gui.main_window.QMessageBox.information"
        ) as information:
            self.window.capture_layout_tuner_preview_frame()

        self.assertTrue(information.called)
        self.assertEqual({}, self.window.layout_tuner_warped)
        self.assertIsNone(self.window.layout_tuner_cache_key)

    def test_layout_tuner_capture_records_projection_provenance_key(self) -> None:
        frames = {
            camera: np.full((4, 6, 3), index, dtype=np.uint8)
            for index, camera in enumerate(
                ("front_left", "front", "front_right"),
                start=1,
            )
        }
        masks = {
            camera: np.ones(frame.shape[:2], dtype=bool)
            for camera, frame in frames.items()
        }
        projection = ProjectionResult(
            warped_images={camera: frame.copy() for camera, frame in frames.items()},
            valid_masks=masks,
            metadata={
                "projection_source": ProjectionSource.CURRENT_PERSPECTIVE.value,
                "valid_mask_source": "geometry",
            },
            timings={"projection_total_ms": 1.0},
        )
        self.window.frames = frames

        with (
            patch.object(
                self.window,
                "project_layout_tuner_frames",
                return_value=projection,
            ),
            patch.object(self.window, "refresh_layout_tuner_preview") as refresh,
        ):
            self.window.capture_layout_tuner_preview_frame()

        self.assertIsNotNone(self.window.layout_tuner_cache_key)
        self.assertEqual(
            self.window.layout_tuner_cache_key.to_dict(),
            self.window.layout_tuner_source_info["cache_key"],
        )
        self.assertEqual(
            self.window.layout_tuner_cache_key.frame_signature,
            self.window.layout_tuner_source_info["frame_info"]["frame_signature"],
        )
        refresh.assert_called_once_with()

    def test_layout_tuner_save_rejects_stale_projection_cache(self) -> None:
        self.window.layout_tuner_preview_result = SimpleNamespace(
            image=np.zeros((4, 6, 3), dtype=np.uint8),
            metrics={},
        )

        with (
            patch(
                "deep_shark_studio.gui.main_window.QMessageBox.information"
            ) as information,
            patch(
                "deep_shark_studio.gui.main_window.save_layout_tuner_candidate"
            ) as save_candidate,
        ):
            self.window.save_layout_tuner_candidate()

        self.assertTrue(information.called)
        save_candidate.assert_not_called()
        self.assertIsNone(self.window.layout_tuner_preview_result)

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
            details = self.window.runtime_candidate_status.toolTip()
            self.assertIn("schema_version: 2", text)
            self.assertIn("profile_id: triple_front_panorama", text)
            self.assertIn("left_pair:", details)
            self.assertIn("right_pair:", details)
        after = file_revision(CONFIG_DIR / "calibration.yaml")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
