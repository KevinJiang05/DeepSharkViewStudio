"""Video output sinks for the headless DeepShark payload service."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoOutputConfig:
    """FFmpeg output settings for a QGC-consumable stitched video stream."""

    kind: str = "udp_mpegts"
    url: str = "udp://127.0.0.1:5600?pkt_size=1316"
    fps: float = 15.0
    codec: str = "libx264"
    bitrate: str = "6000k"
    preset: str = "ultrafast"
    tune: str = "zerolatency"
    pixel_format: str = "yuv420p"
    rtsp_transport: str = "tcp"
    ffmpeg_path: str = "ffmpeg"
    output_width: int | None = None
    output_height: int | None = None
    restart_limit: int = 3
    restart_backoff_seconds: float = 0.5


@dataclass(frozen=True)
class VideoSinkStatus:
    """Thread-safe snapshot of the effective FFmpeg output lifecycle."""

    state: str
    running: bool
    pid: int | None
    frames_submitted: int
    frames_written: int
    frames_dropped: int
    pending_frames: int
    restart_count: int
    last_restart_reason: str
    last_error: str
    last_stderr_line: str
    recent_stderr: tuple[str, ...]
    exit_code: int | None
    width: int
    height: int
    fps: float
    url: str
    started_at: float | None
    last_write_at: float | None


class VideoSink(Protocol):
    def open(self, width: int, height: int, fps: float) -> None:
        ...

    def write(self, frame_bgr: np.ndarray) -> bool:
        ...

    def close(self) -> None:
        ...


class NullVideoSink:
    """Test sink that counts frames without starting an encoder."""

    def __init__(self):
        self.opened = False
        self.frames: list[np.ndarray] = []
        self.size: tuple[int, int] | None = None
        self.fps = 0.0

    def open(self, width: int, height: int, fps: float) -> None:
        self.opened = True
        self.size = (int(width), int(height))
        self.fps = float(fps)

    def write(self, frame_bgr: np.ndarray) -> bool:
        if not self.opened:
            height, width = frame_bgr.shape[:2]
            self.open(width, height, self.fps or 1.0)
        self.frames.append(np.asarray(frame_bgr).copy())
        return True

    def close(self) -> None:
        self.opened = False


class FfmpegVideoSink:
    """Bounded latest-frame FFmpeg writer for a standard QGC video source.

    ``write`` only validates and replaces a single pending frame. Resize,
    conversion, pipe I/O, stderr draining and bounded restart all happen on
    daemon worker threads, so a slow encoder or receiver cannot block Qt or
    the stitch-result consumer.
    """

    def __init__(self, config: VideoOutputConfig):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._condition = threading.Condition(threading.RLock())
        self._lifecycle_lock = threading.RLock()
        self._writer_thread: threading.Thread | None = None
        self._stderr_threads: list[threading.Thread] = []
        self._pending_frame: np.ndarray | None = None
        self._stop_requested = False
        self._state = "stopped"
        self._frames_submitted = 0
        self._frames_written = 0
        self._frames_dropped = 0
        self._restart_count = 0
        self._last_restart_reason = ""
        self._last_error = ""
        self._last_stderr_line = ""
        self._recent_stderr: deque[str] = deque(maxlen=20)
        self._exit_code: int | None = None
        self._started_at: float | None = None
        self._last_write_at: float | None = None
        self._ffmpeg_version = ""

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        if self.config.output_width and self.config.output_height:
            return int(self.config.output_width), int(self.config.output_height)
        return int(width), int(height)

    def preflight(self) -> str:
        """Validate the executable and output contract without starting output."""
        if self.config.kind not in {"rtsp", "udp_mpegts"}:
            raise ValueError(
                "Unsupported QGC video output kind. Expected 'rtsp' or 'udp_mpegts'."
            )
        if not str(self.config.url).strip():
            raise ValueError("QGC video output URL is empty.")
        if float(self.config.fps) <= 0.0:
            raise ValueError("QGC video output FPS must be positive.")
        executable = str(self.config.ffmpeg_path).strip()
        resolved = shutil.which(executable)
        if resolved is None and Path(executable).is_file():
            resolved = str(Path(executable))
        if resolved is None:
            raise RuntimeError(f"FFmpeg executable was not found: {executable}")
        kwargs: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "timeout": 5.0,
            "check": False,
        }
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creation_flags:
            kwargs["creationflags"] = creation_flags
        try:
            completed = subprocess.run(
                [resolved, "-hide_banner", "-version"],
                **kwargs,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"FFmpeg preflight failed: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                "FFmpeg preflight failed"
                + (f": {detail.splitlines()[-1]}" if detail else ".")
            )
        version = (completed.stdout or "").splitlines()
        self._ffmpeg_version = version[0] if version else "FFmpeg available"
        return self._ffmpeg_version

    def open(self, width: int, height: int, fps: float) -> None:
        target_width, target_height = self._target_size(width, height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("FFmpeg output dimensions must be positive.")
        if float(fps) <= 0.0:
            raise ValueError("FFmpeg output FPS must be positive.")
        with self._lifecycle_lock:
            with self._condition:
                writer_alive = bool(
                    self._writer_thread is not None
                    and self._writer_thread.is_alive()
                )
                process_alive = bool(
                    self.process is not None
                    and self.process.poll() is None
                )
                if writer_alive and process_alive:
                    if (target_width, target_height) != (
                        self.width,
                        self.height,
                    ):
                        raise ValueError(
                            f"Output size is already {self.width}x{self.height}; "
                            f"cannot reopen as {target_width}x{target_height}."
                        )
                    return
            if not self._ffmpeg_version:
                self.preflight()
            with self._condition:
                self.width = target_width
                self.height = target_height
                self.fps = float(fps)
                self._stop_requested = False
                self._pending_frame = None
                self._state = "starting"
                self._frames_submitted = 0
                self._frames_written = 0
                self._frames_dropped = 0
                self._restart_count = 0
                self._last_restart_reason = ""
                self._last_error = ""
                self._last_stderr_line = ""
                self._recent_stderr.clear()
                self._exit_code = None
                self._started_at = time.time()
                self._last_write_at = None
            try:
                self._start_process()
            except Exception as exc:
                with self._condition:
                    self._state = "failed"
                    self._last_error = str(exc)
                raise
            writer = threading.Thread(
                target=self._writer_loop,
                name="DeepSharkFfmpegWriter",
                daemon=True,
            )
            with self._condition:
                self._writer_thread = writer
                self._state = "running"
            writer.start()

    def write(self, frame_bgr: np.ndarray) -> bool:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video sink expects a BGR uint8 image.")
        if frame.dtype != np.uint8:
            raise ValueError("Video sink expects uint8 frames.")
        height, width = frame.shape[:2]
        target_width, target_height = self._target_size(width, height)
        with self._condition:
            needs_open = self._writer_thread is None or not self._writer_thread.is_alive()
        if needs_open:
            self.open(width, height, self.config.fps)
        if (target_width, target_height) != (self.width, self.height):
            raise ValueError(
                f"Frame size changed from {self.width}x{self.height} to "
                f"{target_width}x{target_height}."
            )
        queued = np.ascontiguousarray(frame).copy()
        with self._condition:
            if self._stop_requested or self._state in {"stopping", "stopped"}:
                return False
            if self._state == "failed":
                raise RuntimeError(self._last_error or "FFmpeg output failed.")
            if self._pending_frame is not None:
                self._frames_dropped += 1
            self._pending_frame = queued
            self._frames_submitted += 1
            self._condition.notify_all()
        return True

    def _start_process(self) -> None:
        command = build_ffmpeg_command(
            self.config,
            self.width,
            self.height,
            self.fps,
        )
        kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
        }
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creation_flags:
            kwargs["creationflags"] = creation_flags
        process = subprocess.Popen(command, **kwargs)
        if process.stdin is None or process.stderr is None:
            self._terminate_process(process)
            raise RuntimeError("FFmpeg process did not expose stdin/stderr pipes.")
        with self._condition:
            if self._stop_requested:
                self._terminate_process(process)
                raise RuntimeError("FFmpeg output was stopped during startup.")
            self.process = process
            self._exit_code = None
        stderr_thread = threading.Thread(
            target=self._consume_stderr,
            args=(process,),
            name=f"DeepSharkFfmpegStderr-{process.pid}",
            daemon=True,
        )
        with self._condition:
            self._stderr_threads.append(stderr_thread)
        stderr_thread.start()

    def _consume_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for raw_line in stream:
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw_line).strip()
                if not line:
                    continue
                with self._condition:
                    self._last_stderr_line = line
                    self._recent_stderr.append(line)
        except (OSError, ValueError) as exc:
            with self._condition:
                if not self._stop_requested:
                    self._recent_stderr.append(
                        f"stderr reader stopped: {type(exc).__name__}: {exc}"
                    )

    def _writer_loop(self) -> None:
        while True:
            with self._condition:
                if self._pending_frame is None and not self._stop_requested:
                    self._condition.wait(timeout=0.1)
                if self._stop_requested:
                    return
                frame = self._pending_frame
                self._pending_frame = None
                process = self.process
            if process is None or process.poll() is not None:
                exit_code = None if process is None else process.poll()
                reason = (
                    "FFmpeg process is unavailable."
                    if exit_code is None
                    else f"FFmpeg exited with code {exit_code}."
                )
                if not self._restart_process(reason):
                    return
                if frame is None:
                    continue
                with self._condition:
                    process = self.process
            if frame is None:
                continue
            try:
                output_frame = frame
                source_height, source_width = frame.shape[:2]
                if (source_width, source_height) != (self.width, self.height):
                    interpolation = (
                        cv2.INTER_AREA
                        if self.width <= source_width
                        and self.height <= source_height
                        else cv2.INTER_LINEAR
                    )
                    output_frame = cv2.resize(
                        frame,
                        (self.width, self.height),
                        interpolation=interpolation,
                    )
                output_frame = np.ascontiguousarray(output_frame)
                if process is None or process.stdin is None:
                    raise BrokenPipeError("FFmpeg stdin is unavailable.")
                process.stdin.write(output_frame.tobytes())
                with self._condition:
                    self._frames_written += 1
                    self._last_write_at = time.time()
            except (BrokenPipeError, OSError, ValueError) as exc:
                reason = f"FFmpeg frame write failed: {type(exc).__name__}: {exc}"
                with self._condition:
                    self._frames_dropped += 1
                if not self._restart_process(reason):
                    return

    def _restart_process(self, reason: str) -> bool:
        with self._condition:
            if self._stop_requested:
                return False
            old_process = self.process
            self.process = None
            self._last_error = reason
            self._last_restart_reason = reason
            if old_process is not None:
                self._exit_code = old_process.poll()
            if self._restart_count >= max(0, int(self.config.restart_limit)):
                self._state = "failed"
                return False
            self._restart_count += 1
            restart_number = self._restart_count
            self._state = "restarting"
        if old_process is not None:
            self._terminate_process(old_process)
        delay = max(0.0, float(self.config.restart_backoff_seconds)) * (
            2 ** max(0, restart_number - 1)
        )
        if delay > 0.0:
            deadline = time.monotonic() + min(delay, 5.0)
            while time.monotonic() < deadline:
                with self._condition:
                    if self._stop_requested:
                        return False
                time.sleep(min(0.05, deadline - time.monotonic()))
        try:
            self._start_process()
        except Exception as exc:
            with self._condition:
                self._last_error = (
                    f"{reason} Restart {restart_number} failed: {exc}"
                )
            return self._restart_process(self._last_error)
        with self._condition:
            self._state = "running"
        return True

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> int | None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            else:
                process.wait(timeout=0.2)
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass
        for stream in (process.stdin, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass
        return process.poll()

    def close(self) -> None:
        with self._lifecycle_lock:
            with self._condition:
                writer = self._writer_thread
                process = self.process
                if (
                    self._state == "stopped"
                    and process is None
                    and (writer is None or not writer.is_alive())
                ):
                    return
                self._state = "stopping"
                self._stop_requested = True
                if self._pending_frame is not None:
                    self._frames_dropped += 1
                    self._pending_frame = None
                self._condition.notify_all()
            if process is not None:
                exit_code = self._terminate_process(process)
                with self._condition:
                    self._exit_code = exit_code
            if writer is not None and writer is not threading.current_thread():
                writer.join(timeout=2.0)
            with self._condition:
                stderr_threads = list(self._stderr_threads)
            for thread in stderr_threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=0.2)
            with self._condition:
                self.process = None
                self._writer_thread = None
                self._stderr_threads = [
                    thread for thread in self._stderr_threads if thread.is_alive()
                ]
                self._state = "stopped"

    def status(self) -> VideoSinkStatus:
        with self._condition:
            process = self.process
            exit_code = self._exit_code
            observed: int | None = None
            if process is not None:
                observed = process.poll()
                if observed is not None:
                    exit_code = observed
            process_alive = bool(process is not None and observed is None)
            writer_alive = bool(
                self._writer_thread is not None
                and self._writer_thread.is_alive()
            )
            running = bool(
                self._state == "running"
                and process_alive
                and writer_alive
            )
            reported_state = self._state
            last_error = self._last_error
            if self._state == "running" and not running:
                if self._stop_requested:
                    reported_state = "stopping"
                elif writer_alive and not process_alive:
                    reported_state = "restarting"
                    if not last_error:
                        last_error = (
                            "FFmpeg process is unavailable."
                            if observed is None
                            else f"FFmpeg exited with code {observed}."
                        )
                else:
                    reported_state = "failed"
                    if not last_error:
                        last_error = "FFmpeg writer thread stopped unexpectedly."
            return VideoSinkStatus(
                state=reported_state,
                running=running,
                pid=process.pid if process_alive else None,
                frames_submitted=self._frames_submitted,
                frames_written=self._frames_written,
                frames_dropped=self._frames_dropped,
                pending_frames=1 if self._pending_frame is not None else 0,
                restart_count=self._restart_count,
                last_restart_reason=self._last_restart_reason,
                last_error=last_error,
                last_stderr_line=self._last_stderr_line,
                recent_stderr=tuple(self._recent_stderr),
                exit_code=exit_code,
                width=self.width,
                height=self.height,
                fps=self.fps,
                url=self.config.url,
                started_at=self._started_at,
                last_write_at=self._last_write_at,
            )


def build_ffmpeg_command(
    config: VideoOutputConfig,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    """Build a low-latency FFmpeg command for raw BGR frame input."""
    base_command = [
        config.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-use_wallclock_as_timestamps",
        "1",
        "-framerate",
        f"{float(fps):.3f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        config.codec,
        "-preset",
        config.preset,
        "-tune",
        config.tune,
        "-b:v",
        config.bitrate,
        "-bf",
        "0",
        "-flags",
        "low_delay",
        "-g",
        str(max(1, int(round(float(fps))))),
        "-pix_fmt",
        config.pixel_format,
    ]
    if config.kind == "rtsp":
        return base_command + [
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            config.rtsp_transport,
            config.url,
        ]
    if config.kind == "udp_mpegts":
        return base_command + [
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-flush_packets",
            "1",
            "-f",
            "mpegts",
            config.url,
        ]
    raise ValueError(
        "Unsupported QGC video output kind. Expected 'rtsp' or 'udp_mpegts'."
    )
