"""Video output sinks for the headless DeepShark payload service."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
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


class VideoSink(Protocol):
    def open(self, width: int, height: int, fps: float) -> None:
        ...

    def write(self, frame_bgr: np.ndarray) -> None:
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

    def write(self, frame_bgr: np.ndarray) -> None:
        if not self.opened:
            height, width = frame_bgr.shape[:2]
            self.open(width, height, self.fps or 1.0)
        self.frames.append(np.asarray(frame_bgr).copy())

    def close(self) -> None:
        self.opened = False


class FfmpegVideoSink:
    """Pipe BGR frames into FFmpeg for a standard QGC video source."""

    def __init__(self, config: VideoOutputConfig):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.width = 0
        self.height = 0

    def _target_size(self, width: int, height: int) -> tuple[int, int]:
        if self.config.output_width and self.config.output_height:
            return int(self.config.output_width), int(self.config.output_height)
        return int(width), int(height)

    def open(self, width: int, height: int, fps: float) -> None:
        if self.process is not None:
            return
        self.width, self.height = self._target_size(width, height)
        command = build_ffmpeg_command(self.config, self.width, self.height, fps)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video sink expects a BGR uint8 image.")
        if frame.dtype != np.uint8:
            raise ValueError("Video sink expects uint8 frames.")
        height, width = frame.shape[:2]
        target_width, target_height = self._target_size(width, height)
        if self.process is None:
            self.open(width, height, self.config.fps)
        if (target_width, target_height) != (self.width, self.height):
            raise ValueError(
                f"Frame size changed from {self.width}x{self.height} to "
                f"{target_width}x{target_height}."
            )
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FFmpeg process is not writable.")
        if self.process.poll() is not None:
            raise RuntimeError("FFmpeg process exited before frame write.")
        if (width, height) != (target_width, target_height):
            interpolation = (
                cv2.INTER_AREA
                if target_width <= width and target_height <= height
                else cv2.INTER_LINEAR
            )
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
        frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()


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
