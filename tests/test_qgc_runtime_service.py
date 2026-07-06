from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from deep_shark_studio.qgc.runtime_service import (
    DeepSharkRuntimeService,
    DeepSharkRuntimeServiceConfig,
    _configured_video_output,
    build_runtime_service_config,
    build_stream_configs,
)
from deep_shark_studio.qgc.video_output import (
    NullVideoSink,
    VideoOutputConfig,
    build_ffmpeg_command,
)
from deep_shark_studio.stitch_runtime_modes import StitchRuntimeMode


def _camera_config() -> dict:
    return {
        "camera_order": ["front_left", "front_right", "front", "behind"],
        "cameras": {
            "front_left": {
                "enabled": True,
                "source_type": "rtsp",
                "source": "rtsp://left",
            },
            "front_right": {
                "enabled": True,
                "source_type": "rtsp",
                "source": "rtsp://right",
            },
            "front": {
                "enabled": True,
                "source_type": "rtsp",
                "source": "rtsp://front",
            },
            "behind": {
                "enabled": True,
                "source_type": "rtsp",
                "source": "rtsp://behind",
            },
        },
    }


def _calibration_config() -> dict:
    return {
        "stitch_topology": "triple_front_panorama",
        "topologies": {
            "triple_front_panorama": {
                "active_cameras": ["front_left", "front", "front_right"],
                "canvas": {"width": 64, "height": 24},
                "cameras": {},
            }
        },
    }


class FakeStreamManager:
    def __init__(self, frames: dict[str, np.ndarray]):
        self.frames = frames
        self.started_configs = None
        self.stopped = False

    def start(self, configs):
        self.started_configs = list(configs)

    def stop(self):
        self.stopped = True

    def latest_frames(self):
        return dict(self.frames), {}


class FakeController:
    def __init__(self, canvas: np.ndarray):
        self.canvas = canvas
        self.calls = 0
        self.last_frames = None

    def process(self, frames):
        self.calls += 1
        self.last_frames = dict(frames)
        return SimpleNamespace(
            canvas=self.canvas,
            status="far_field",
            metrics={"timing": {"far_field_total_ms": 1.0}},
        )


class QGCRuntimeServiceTests(unittest.TestCase):
    def test_build_stream_configs_uses_active_topology_cameras_only(self) -> None:
        configs = build_stream_configs(_camera_config(), _calibration_config())

        self.assertEqual(["front_left", "front_right", "front"], [item.key for item in configs])
        self.assertNotIn("behind", [item.key for item in configs])

    def test_process_once_writes_stitched_canvas_to_sink(self) -> None:
        frames = {
            "front_left": np.zeros((8, 8, 3), dtype=np.uint8),
            "front": np.zeros((8, 8, 3), dtype=np.uint8),
            "front_right": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        canvas = np.full((24, 64, 3), 127, dtype=np.uint8)
        sink = NullVideoSink()
        controller = FakeController(canvas)
        service = DeepSharkRuntimeService(
            _camera_config(),
            _calibration_config(),
            DeepSharkRuntimeServiceConfig(mode=StitchRuntimeMode.FAR_FIELD),
            stream_manager=FakeStreamManager(frames),
            video_sink=sink,
            controller=controller,
        )

        self.assertTrue(service.process_once())

        self.assertEqual(1, controller.calls)
        self.assertEqual(1, len(sink.frames))
        self.assertEqual((24, 64, 3), sink.frames[0].shape)
        self.assertEqual(1, service.status().frames_written)
        self.assertEqual("", service.status().last_error)

    def test_process_once_waits_for_all_active_cameras_by_default(self) -> None:
        frames = {
            "front_left": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        sink = NullVideoSink()
        controller = FakeController(np.zeros((24, 64, 3), dtype=np.uint8))
        service = DeepSharkRuntimeService(
            _camera_config(),
            _calibration_config(),
            DeepSharkRuntimeServiceConfig(mode=StitchRuntimeMode.FAR_FIELD),
            stream_manager=FakeStreamManager(frames),
            video_sink=sink,
            controller=controller,
        )

        self.assertFalse(service.process_once())

        self.assertEqual(0, controller.calls)
        self.assertIn("Missing active camera frames", service.status().last_error)

    def test_candidate_processor_mode_outputs_b2_candidate_canvas(self) -> None:
        frames = {
            "front_left": np.zeros((8, 8, 3), dtype=np.uint8),
            "front": np.zeros((8, 8, 3), dtype=np.uint8),
            "front_right": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        candidate_canvas = np.full((24, 64, 3), 33, dtype=np.uint8)

        class FakeCandidateProcessor:
            def __init__(self, _directory, use_opencl=False):
                self.last_timings = {}
                self.use_opencl = use_opencl
                pass

            def process(self, _frames):
                self.last_timings = {
                    "candidate_remap_ms": 3.0,
                    "candidate_compose_ms": 1.0,
                }
                return {}, candidate_canvas

        with patch(
            "deep_shark_studio.qgc.runtime_service.CandidatePanoramaProcessor",
            FakeCandidateProcessor,
        ):
            sink = NullVideoSink()
            service = DeepSharkRuntimeService(
                _camera_config(),
                _calibration_config(),
                DeepSharkRuntimeServiceConfig(
                    mode=StitchRuntimeMode.FAR_FIELD,
                    processor_mode="candidate",
                    candidate_directory="D:/tmp/fake-candidate",
                ),
                stream_manager=FakeStreamManager(frames),
                video_sink=sink,
            )

            self.assertTrue(service.process_once())

        self.assertEqual(1, len(sink.frames))
        self.assertTrue(np.array_equal(candidate_canvas, sink.frames[0]))
        self.assertEqual("b2_candidate_view", service.status().last_runtime_status)
        self.assertEqual(
            3.0,
            service.status().last_runtime_metrics["timing"]["candidate_remap_ms"],
        )

    def test_ffmpeg_command_is_low_latency_mpegts_udp(self) -> None:
        config = VideoOutputConfig(
            kind="udp_mpegts",
            url="udp://192.168.1.50:5600?pkt_size=1316",
            fps=15.0,
            bitrate="4000k",
            ffmpeg_path="ffmpeg",
        )

        command = build_ffmpeg_command(config, width=2200, height=700, fps=15.0)

        self.assertIn("rawvideo", command)
        self.assertIn("bgr24", command)
        self.assertIn("libx264", command)
        self.assertIn("zerolatency", command)
        self.assertIn("-use_wallclock_as_timestamps", command)
        self.assertIn("-framerate", command)
        self.assertNotIn("-r", command)
        self.assertIn("-fflags", command)
        self.assertIn("nobuffer", command)
        self.assertIn("-flush_packets", command)
        self.assertIn("mpegts", command)
        self.assertEqual("udp://192.168.1.50:5600?pkt_size=1316", command[-1])

    def test_ffmpeg_command_supports_rtsp_output(self) -> None:
        config = VideoOutputConfig(
            kind="rtsp",
            url="rtsp://192.168.1.20:8554/deepshark",
            fps=15.0,
            bitrate="4000k",
            rtsp_transport="tcp",
            ffmpeg_path="ffmpeg",
        )

        command = build_ffmpeg_command(config, width=2200, height=700, fps=15.0)

        self.assertIn("rawvideo", command)
        self.assertIn("bgr24", command)
        self.assertIn("libx264", command)
        self.assertIn("zerolatency", command)
        self.assertIn("-use_wallclock_as_timestamps", command)
        self.assertIn("-framerate", command)
        self.assertIn("rtsp", command)
        self.assertIn("-rtsp_transport", command)
        self.assertIn("tcp", command)
        self.assertEqual("rtsp://192.168.1.20:8554/deepshark", command[-1])

    def test_video_output_config_can_come_from_camera_config(self) -> None:
        args = SimpleNamespace(
            output_kind=None,
            output_url=None,
            output_fps=None,
            output_bitrate=None,
            ffmpeg=None,
            rtsp_transport=None,
        )
        camera_config = {
            "qgc_video_output": {
                "kind": "rtsp",
                "url": "rtsp://10.0.0.2:8554/deepshark",
                "fps": 12.0,
                "bitrate": "5000k",
                "ffmpeg_path": "custom-ffmpeg",
                "rtsp_transport": "tcp",
                "output_width": 1800,
                "output_height": 600,
            },
            "performance": {"b2_candidate_opencl": True},
        }

        config = _configured_video_output(args, camera_config)

        self.assertEqual("rtsp", config.kind)
        self.assertEqual("rtsp://10.0.0.2:8554/deepshark", config.url)
        self.assertEqual(12.0, config.fps)
        self.assertEqual("5000k", config.bitrate)
        self.assertEqual("custom-ffmpeg", config.ffmpeg_path)
        self.assertEqual(1800, config.output_width)
        self.assertEqual(600, config.output_height)

    def test_video_output_config_defaults_to_udp_local_bridge(self) -> None:
        args = SimpleNamespace(
            output_kind=None,
            output_url=None,
            output_fps=None,
            output_bitrate=None,
            ffmpeg=None,
            rtsp_transport=None,
        )

        config = _configured_video_output(args, {})

        self.assertEqual("udp_mpegts", config.kind)
        self.assertEqual("udp://127.0.0.1:5600?pkt_size=1316", config.url)

    def test_runtime_config_accepts_candidate_processor_mode(self) -> None:
        args = SimpleNamespace(
            mode=StitchRuntimeMode.FAR_FIELD.value,
            output_kind=None,
            output_url=None,
            output_fps=None,
            output_bitrate=None,
            ffmpeg=None,
            rtsp_transport=None,
            far_field_layout_candidate="",
            near_field_layout_candidate="",
            processor_mode="candidate",
            candidate_directory="D:/tmp/fake-candidate",
            projection_source="current_perspective",
            projection_intrinsics_source="",
            fisheye_balance=0.6,
            fisheye_fov_scale=1.0,
            max_input_width=960,
            use_intrinsics=False,
            process_fps=15.0,
            allow_partial_frames=False,
        )

        config = build_runtime_service_config(args, _camera_config())

        self.assertEqual("candidate", config.processor_mode)
        self.assertEqual("D:\\tmp\\fake-candidate", str(config.candidate_directory))

    def test_runtime_config_reads_candidate_opencl_from_performance(self) -> None:
        args = SimpleNamespace(
            mode=StitchRuntimeMode.FAR_FIELD.value,
            output_kind=None,
            output_url=None,
            output_fps=None,
            output_bitrate=None,
            ffmpeg=None,
            rtsp_transport=None,
            far_field_layout_candidate="",
            near_field_layout_candidate="",
            processor_mode="candidate",
            candidate_directory="D:/tmp/fake-candidate",
            candidate_opencl=False,
            projection_source="current_perspective",
            projection_intrinsics_source="",
            fisheye_balance=0.6,
            fisheye_fov_scale=1.0,
            max_input_width=960,
            use_intrinsics=False,
            process_fps=15.0,
            allow_partial_frames=False,
        )

        config = build_runtime_service_config(
            args,
            {"performance": {"b2_candidate_opencl": True}},
        )

        self.assertTrue(config.candidate_use_opencl)


if __name__ == "__main__":
    unittest.main()
