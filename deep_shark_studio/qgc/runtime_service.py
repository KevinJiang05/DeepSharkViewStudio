"""Headless DeepShark stitched-video payload service for QGC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping

import numpy as np

from deep_shark_studio.config import CONFIG_DIR, DEFAULT_PROCESS_FPS, load_yaml
from deep_shark_studio.calibration_candidate import CandidatePanoramaProcessor
from deep_shark_studio.stream_manager import CameraStreamConfig, CameraStreamManager
from deep_shark_studio.stitch_runtime_controller import RuntimeStitchController, RuntimeStitchResult
from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)
from deep_shark_studio.stitcher import SurroundStitcher
from deep_shark_studio.topology import active_topology_camera_keys

from .video_output import FfmpegVideoSink, VideoOutputConfig, VideoSink


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeepSharkRuntimeServiceConfig:
    """Read-only deployment config for the headless stitched video service."""

    mode: StitchRuntimeMode = StitchRuntimeMode.FAR_FIELD
    output: VideoOutputConfig = field(default_factory=VideoOutputConfig)
    far_field_layout_candidate_path: Path | None = None
    near_field_layout_candidate_path: Path | None = None
    processor_mode: str = "runtime"
    candidate_directory: Path | None = None
    candidate_use_opencl: bool = False
    projection_source: ProjectionSource = ProjectionSource.CURRENT_PERSPECTIVE
    projection_intrinsics_source_path: Path | None = None
    fisheye_balance: float = 0.6
    fisheye_fov_scale: float = 1.0
    max_input_width: int | None = 960
    use_intrinsics: bool = False
    process_fps: float = float(DEFAULT_PROCESS_FPS)
    require_all_active_cameras: bool = True
    max_frame_age_seconds: float = 3.0
    failure_backoff_initial_seconds: float = 0.1
    failure_backoff_max_seconds: float = 2.0
    max_consecutive_failures: int = 300

    @property
    def uses_far_field_custom(self) -> bool:
        return (
            self.mode == StitchRuntimeMode.FAR_FIELD
            and self.far_field_layout_candidate_path is not None
        )

    def runtime_stitch_config(self) -> RuntimeStitchConfig:
        return RuntimeStitchConfig(
            mode=self.mode,
            projection_source=self.projection_source,
            layout_candidate_path=self.near_field_layout_candidate_path,
            use_far_field_custom_layout=self.uses_far_field_custom,
            far_field_layout_candidate_path=self.far_field_layout_candidate_path,
            projection_intrinsics_source_path=self.projection_intrinsics_source_path,
            fisheye_balance=self.fisheye_balance,
            fisheye_fov_scale=self.fisheye_fov_scale,
            auto_enabled=False,
        )


@dataclass(frozen=True)
class DeepSharkRuntimeStatus:
    running: bool
    required_cameras: tuple[str, ...]
    frames_submitted: int
    frames_written: int
    duplicate_frames_skipped: int
    consecutive_failures: int
    last_error: str
    last_event: str
    frame_ages_seconds: dict[str, float | None]
    last_runtime_status: str
    last_runtime_metrics: dict[str, Any] | None
    output_state: str
    output_pid: int | None
    output_exit_code: int | None
    output_restart_count: int
    output_last_stderr_line: str


class DeepSharkRuntimeService:
    """Owns headless capture, stitching, and stitched-video output."""

    def __init__(
        self,
        camera_config: Mapping[str, Any],
        calibration_config: Mapping[str, Any],
        service_config: DeepSharkRuntimeServiceConfig,
        stream_manager: CameraStreamManager | None = None,
        video_sink: VideoSink | None = None,
        controller: RuntimeStitchController | None = None,
    ):
        self.camera_config = dict(camera_config)
        self.calibration_config = dict(calibration_config)
        self.service_config = service_config
        self.stream_manager = stream_manager or CameraStreamManager()
        self.video_sink = video_sink or FfmpegVideoSink(service_config.output)
        if controller is None:
            if service_config.processor_mode == "candidate":
                if service_config.candidate_directory is None:
                    raise ValueError(
                        "Candidate output mode requires a candidate directory."
                    )
                controller = CandidateRuntimeController(
                    service_config.candidate_directory,
                    use_opencl=service_config.candidate_use_opencl,
                )
            else:
                stitcher = SurroundStitcher(
                    self.calibration_config,
                    max_input_width=service_config.max_input_width,
                    use_intrinsics=service_config.use_intrinsics,
                )
                controller = RuntimeStitchController(
                    stitcher,
                    service_config.runtime_stitch_config(),
                )
        self.controller = controller
        self._running = False
        self._stream_configs: list[CameraStreamConfig] = []
        self._required_camera_keys: tuple[str, ...] = ()
        self._frames_submitted = 0
        self._duplicate_frames_skipped = 0
        self._consecutive_failures = 0
        self._last_error = ""
        self._last_event = "created"
        self._last_frame_signature: tuple[tuple[str, int, float], ...] = ()
        self._frame_ages_seconds: dict[str, float | None] = {}
        self._last_runtime_status = ""
        self._last_runtime_metrics: dict[str, Any] | None = None

    def start(self) -> None:
        if self._running:
            return
        stream_configs = build_stream_configs(
            self.camera_config,
            self.calibration_config,
        )
        enabled_configs = [config for config in stream_configs if config.enabled]
        if not enabled_configs:
            raise RuntimeError(
                "Headless startup preflight found no enabled cameras in the active topology."
            )
        missing_sources = [
            config.key
            for config in enabled_configs
            if config.source in (None, "")
        ]
        if missing_sources:
            raise RuntimeError(
                "Headless startup preflight found empty camera sources: "
                + ", ".join(missing_sources)
            )
        preflight = getattr(self.video_sink, "preflight", None)
        if callable(preflight):
            preflight()
        self._stream_configs = stream_configs
        self._required_camera_keys = tuple(
            config.key for config in enabled_configs
        )
        self._last_frame_signature = ()
        self._consecutive_failures = 0
        self._last_error = ""
        self._last_event = (
            "starting capture for " + ", ".join(self._required_camera_keys)
        )
        self.stream_manager.start(stream_configs)
        self._running = True

    def stop(self) -> None:
        if not self._running and self._last_event == "stopped":
            return
        self._running = False
        self.stream_manager.stop()
        self.video_sink.close()
        self._last_event = "stopped"

    def run_forever(self) -> None:
        self.start()
        stop_requested = False

        def _request_stop(_signum, _frame) -> None:
            nonlocal stop_requested
            stop_requested = True

        previous_sigint = signal.signal(signal.SIGINT, _request_stop)
        previous_sigterm = signal.signal(signal.SIGTERM, _request_stop)
        last_reported_error = ""
        try:
            interval = 1.0 / max(0.1, float(self.service_config.process_fps))
            while self._running and not stop_requested:
                loop_start = time.perf_counter()
                success = self.process_once()
                if self._last_error and self._last_error != last_reported_error:
                    LOGGER.error("Headless output: %s", self._last_error)
                    last_reported_error = self._last_error
                elif success:
                    last_reported_error = ""
                if (
                    self.service_config.max_consecutive_failures > 0
                    and self._consecutive_failures
                    >= self.service_config.max_consecutive_failures
                ):
                    raise RuntimeError(
                        "Headless output stopped after "
                        f"{self._consecutive_failures} consecutive failures: "
                        f"{self._last_error}"
                    )
                elapsed = time.perf_counter() - loop_start
                failure_delay = 0.0
                if self._consecutive_failures:
                    initial = max(
                        0.0,
                        float(
                            self.service_config.failure_backoff_initial_seconds
                        ),
                    )
                    maximum = max(
                        initial,
                        float(self.service_config.failure_backoff_max_seconds),
                    )
                    failure_delay = min(
                        maximum,
                        initial
                        * (2 ** min(10, self._consecutive_failures - 1)),
                    )
                time.sleep(max(failure_delay, interval - elapsed, 0.0))
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)
            self.stop()

    def process_once(self) -> bool:
        if not self._required_camera_keys:
            self._required_camera_keys = tuple(
                config.key
                for config in build_stream_configs(
                    self.camera_config,
                    self.calibration_config,
                )
                if config.enabled
            )
        sink_status_fn = getattr(self.video_sink, "status", None)
        if callable(sink_status_fn):
            sink_status = sink_status_fn()
            if sink_status.state == "failed":
                detail = (
                    sink_status.last_error
                    or sink_status.last_stderr_line
                    or "unknown FFmpeg failure"
                )
                return self._record_failure(f"FFmpeg output failed: {detail}")
        frames, snapshots = self.stream_manager.latest_frames(
            max_frame_age_seconds=max(
                0.0,
                float(self.service_config.max_frame_age_seconds),
            )
        )
        required = set(self._required_camera_keys)
        frames = {
            key: frame for key, frame in frames.items() if key in required
        }
        now = time.time()
        self._frame_ages_seconds = {
            key: (
                max(0.0, now - float(snapshot.frame_timestamp))
                if float(getattr(snapshot, "frame_timestamp", 0.0)) > 0.0
                else None
            )
            for key, snapshot in snapshots.items()
            if key in required
        }
        if self.service_config.require_all_active_cameras:
            missing = [
                key for key in self._required_camera_keys if key not in frames
            ]
            if missing:
                return self._record_failure(
                    "Missing enabled camera frames: " + ", ".join(missing)
                )
        if not frames:
            return self._record_failure("No fresh enabled camera frames available.")
        signature = tuple(
            sorted(
                (
                    key,
                    int(getattr(snapshots.get(key), "frame_count", 0)),
                    float(
                        getattr(snapshots.get(key), "frame_timestamp", 0.0)
                    ),
                )
                for key in frames
            )
        )
        if signature and signature == self._last_frame_signature:
            self._duplicate_frames_skipped += 1
            self._last_error = ""
            self._last_event = "Skipped duplicate camera frame set."
            return False
        try:
            result = self.controller.process(frames)
            self._last_runtime_status = result.status
            self._last_runtime_metrics = result.metrics
            if result.canvas is None:
                return self._record_failure(
                    "Runtime stitcher returned no canvas."
                )
            accepted = self.video_sink.write(result.canvas)
            if accepted is False:
                return self._record_failure(
                    "Video output rejected the stitched canvas."
                )
            self._frames_submitted += 1
            self._last_frame_signature = signature
            self._last_error = ""
            self._last_event = "Submitted a fresh stitched canvas."
            self._consecutive_failures = 0
            return True
        except Exception as exc:
            return self._record_failure(
                f"{type(exc).__name__}: {exc}"
            )

    def _record_failure(self, message: str) -> bool:
        self._last_error = str(message)
        self._last_event = self._last_error
        self._consecutive_failures += 1
        return False

    def status(self) -> DeepSharkRuntimeStatus:
        output_status_fn = getattr(self.video_sink, "status", None)
        output_status = output_status_fn() if callable(output_status_fn) else None
        frames_written = (
            int(output_status.frames_written)
            if output_status is not None
            else self._frames_submitted
        )
        return DeepSharkRuntimeStatus(
            running=self._running,
            required_cameras=self._required_camera_keys,
            frames_submitted=self._frames_submitted,
            frames_written=frames_written,
            duplicate_frames_skipped=self._duplicate_frames_skipped,
            consecutive_failures=self._consecutive_failures,
            last_error=self._last_error,
            last_event=self._last_event,
            frame_ages_seconds=dict(self._frame_ages_seconds),
            last_runtime_status=self._last_runtime_status,
            last_runtime_metrics=self._last_runtime_metrics,
            output_state=(
                str(output_status.state)
                if output_status is not None
                else ("running" if self._running else "stopped")
            ),
            output_pid=(
                output_status.pid if output_status is not None else None
            ),
            output_exit_code=(
                output_status.exit_code if output_status is not None else None
            ),
            output_restart_count=(
                int(output_status.restart_count)
                if output_status is not None
                else 0
            ),
            output_last_stderr_line=(
                str(output_status.last_stderr_line)
                if output_status is not None
                else ""
            ),
        )


class CandidateRuntimeController:
    """Expose the B-2 candidate live view through the same service controller API."""

    def __init__(self, candidate_directory: str | Path, use_opencl: bool = False):
        self.processor = CandidatePanoramaProcessor(
            candidate_directory,
            use_opencl=use_opencl,
        )

    def process(self, frames: Mapping[str, np.ndarray]) -> RuntimeStitchResult:
        warped, canvas = self.processor.process(dict(frames))
        timing = getattr(self.processor, "last_timings", {})
        return RuntimeStitchResult(
            warped=warped,
            canvas=canvas,
            mode=StitchRuntimeMode.FAR_FIELD,
            status="b2_candidate_view",
            warnings=(
                "B-2 candidate view is experimental and read-only; it does not write calibration.yaml.",
            ),
            metrics={
                "runtime_mode": "b2_candidate_view",
                "timing": timing if isinstance(timing, dict) else {},
            },
        )


def build_stream_configs(
    camera_config: Mapping[str, Any],
    calibration_config: Mapping[str, Any],
) -> list[CameraStreamConfig]:
    active = set(active_topology_camera_keys(dict(calibration_config)))
    cameras = camera_config.get("cameras", {})
    if not isinstance(cameras, Mapping):
        return []
    configs: list[CameraStreamConfig] = []
    for key in camera_config.get("camera_order", list(cameras)):
        key_text = str(key)
        if key_text not in active:
            continue
        raw = cameras.get(key_text, {})
        if not isinstance(raw, Mapping):
            continue
        source = raw.get("source", "")
        source_type = str(raw.get("source_type", "rtsp"))
        if source_type == "usb":
            try:
                source = int(source)
            except (TypeError, ValueError):
                source = 0
        configs.append(
            CameraStreamConfig(
                key=key_text,
                source_type=source_type,
                source=source,
                enabled=bool(raw.get("enabled", True)),
            )
        )
    return configs


def _configured_video_output(
    args: argparse.Namespace,
    camera_config: dict | None = None,
) -> VideoOutputConfig:
    raw = (camera_config or {}).get("qgc_video_output", {})
    if not isinstance(raw, dict):
        raw = {}
    kind = str(args.output_kind or raw.get("kind", "udp_mpegts"))
    url = str(
        args.output_url
        or raw.get(
            "url",
            (
                "udp://127.0.0.1:5600?pkt_size=1316"
                if kind == "udp_mpegts"
                else "rtsp://127.0.0.1:8554/deepshark"
            ),
        )
    )

    def _optional_positive_int(*values: Any) -> int | None:
        for value in values:
            if value in (None, ""):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return None

    return VideoOutputConfig(
        kind=kind,
        url=url,
        fps=float(args.output_fps or raw.get("fps", 15.0)),
        bitrate=str(args.output_bitrate or raw.get("bitrate", "6000k")),
        ffmpeg_path=str(args.ffmpeg or raw.get("ffmpeg_path", "ffmpeg")),
        rtsp_transport=str(args.rtsp_transport or raw.get("rtsp_transport", "tcp")),
        output_width=_optional_positive_int(
            getattr(args, "output_width", None),
            raw.get("output_width"),
            raw.get("width"),
        ),
        output_height=_optional_positive_int(
            getattr(args, "output_height", None),
            raw.get("output_height"),
            raw.get("height"),
        ),
    )


def build_runtime_service_config(
    args: argparse.Namespace,
    camera_config: dict | None = None,
) -> DeepSharkRuntimeServiceConfig:
    mode = StitchRuntimeMode(str(args.mode))
    projection_source = ProjectionSource(str(args.projection_source))
    performance = (camera_config or {}).get("performance", {})
    if not isinstance(performance, Mapping):
        performance = {}
    return DeepSharkRuntimeServiceConfig(
        mode=mode,
        output=_configured_video_output(args, camera_config),
        far_field_layout_candidate_path=(
            Path(args.far_field_layout_candidate)
            if args.far_field_layout_candidate
            else None
        ),
        near_field_layout_candidate_path=(
            Path(args.near_field_layout_candidate)
            if args.near_field_layout_candidate
            else None
        ),
        processor_mode=str(args.processor_mode),
        candidate_directory=(
            Path(args.candidate_directory)
            if args.candidate_directory
            else None
        ),
        candidate_use_opencl=bool(
            getattr(args, "candidate_opencl", False)
            or performance.get("b2_candidate_opencl", False)
        ),
        projection_source=projection_source,
        projection_intrinsics_source_path=(
            Path(args.projection_intrinsics_source)
            if args.projection_intrinsics_source
            else None
        ),
        fisheye_balance=float(args.fisheye_balance),
        fisheye_fov_scale=float(args.fisheye_fov_scale),
        max_input_width=(None if int(args.max_input_width) <= 0 else int(args.max_input_width)),
        use_intrinsics=bool(args.use_intrinsics),
        process_fps=float(args.process_fps),
        require_all_active_cameras=not bool(args.allow_partial_frames),
        max_frame_age_seconds=float(
            getattr(args, "max_frame_age_seconds", 3.0)
        ),
        failure_backoff_initial_seconds=float(
            getattr(args, "failure_backoff_initial", 0.1)
        ),
        failure_backoff_max_seconds=float(
            getattr(args, "failure_backoff_max", 2.0)
        ),
        max_consecutive_failures=int(
            getattr(args, "max_consecutive_failures", 300)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = _argument_parser()
    args = parser.parse_args(argv)
    camera_config = load_yaml(args.cameras_config)
    calibration_config = load_yaml(args.calibration_config)
    service = DeepSharkRuntimeService(
        camera_config,
        calibration_config,
        build_runtime_service_config(args, camera_config),
    )
    try:
        service.run_forever()
    except KeyboardInterrupt:
        logging.info("DeepShark headless service stopped by operator.")
        return 0
    except Exception as exc:
        print(f"DeepShark headless service failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepShark as a headless stitched-video payload service."
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=CONFIG_DIR / "cameras.yaml",
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=CONFIG_DIR / "calibration.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=[StitchRuntimeMode.FAR_FIELD.value, StitchRuntimeMode.NEAR_FIELD.value],
        default=StitchRuntimeMode.FAR_FIELD.value,
    )
    parser.add_argument("--far-field-layout-candidate", default="")
    parser.add_argument("--near-field-layout-candidate", default="")
    parser.add_argument(
        "--processor-mode",
        choices=["runtime", "candidate"],
        default="runtime",
    )
    parser.add_argument("--candidate-directory", default="")
    parser.add_argument(
        "--candidate-opencl",
        action="store_true",
        help="Use experimental OpenCL acceleration for B-2 candidate processor.",
    )
    parser.add_argument(
        "--projection-source",
        choices=[
            ProjectionSource.CURRENT_PERSPECTIVE.value,
            ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
        ],
        default=ProjectionSource.CURRENT_PERSPECTIVE.value,
    )
    parser.add_argument("--projection-intrinsics-source", default="")
    parser.add_argument("--fisheye-balance", type=float, default=0.6)
    parser.add_argument("--fisheye-fov-scale", type=float, default=1.0)
    parser.add_argument(
        "--process-fps",
        type=float,
        default=float(DEFAULT_PROCESS_FPS),
    )
    parser.add_argument("--max-frame-age-seconds", type=float, default=3.0)
    parser.add_argument("--failure-backoff-initial", type=float, default=0.1)
    parser.add_argument("--failure-backoff-max", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=300)
    parser.add_argument(
        "--output-kind",
        choices=["rtsp", "udp_mpegts"],
        default=None,
    )
    parser.add_argument("--output-url", default=None)
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--output-bitrate", default=None)
    parser.add_argument("--output-width", type=int, default=None)
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp"], default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--max-input-width", type=int, default=960)
    parser.add_argument("--use-intrinsics", action="store_true")
    parser.add_argument("--allow-partial-frames", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
