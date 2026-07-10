"""Threaded live camera capture for DeepShark View Studio."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, Lock, Thread
import time
from typing import Any

import cv2
import numpy as np


STREAM_IDLE = "Idle"
STREAM_CONNECTING = "Connecting"
STREAM_LIVE = "Live"
STREAM_FAILED = "Failed"
STREAM_STOPPING = "Stopping"
STREAM_STOPPED = "Stopped"
DEFAULT_STOP_TIMEOUT_SECONDS = 2.0
DEFAULT_RTSP_OPEN_TIMEOUT_MS = 5000
DEFAULT_RTSP_READ_TIMEOUT_MS = 2000
DUPLICATE_WORKER_ERROR = "Previous worker is still stopping; refusing duplicate capture connection."


@dataclass(frozen=True)
class CameraStreamConfig:
    """Resolved runtime source for one live camera."""

    key: str
    source_type: str
    source: str | int
    enabled: bool = True


@dataclass
class CameraStreamSnapshot:
    """Thread-safe public state exposed to the GUI."""

    key: str
    status: str = STREAM_IDLE
    source_type: str = ""
    source: str = ""
    frame: np.ndarray | None = None
    frame_timestamp: float = 0.0
    frame_count: int = 0
    failed_read_count: int = 0
    last_error: str = ""
    thread_alive: bool = False
    backend: str = ""
    timeout_mode: str = ""


class _CameraStreamWorker:
    """Owns one cv2.VideoCapture and continuously keeps only the newest frame."""

    def __init__(self, config: CameraStreamConfig):
        self.config = config
        self._lock = Lock()
        self._stop_event = Event()
        self._thread = Thread(
            target=self._run,
            name=f"CameraStream-{config.key}",
            daemon=True,
        )
        self._status = STREAM_IDLE
        self._frame: np.ndarray | None = None
        self._frame_timestamp = 0.0
        self._frame_count = 0
        self._failed_read_count = 0
        self._last_error = ""
        self._backend = ""
        self._timeout_mode = ""

    def start(self) -> None:
        self._set_state(status=STREAM_CONNECTING, last_error="")
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._mark_stopping()

    def wait(self, timeout: float) -> bool:
        self._thread.join(timeout=max(0.0, timeout))
        return not self._thread.is_alive()

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> bool:
        self.request_stop()
        stopped = self.wait(timeout)
        if not stopped:
            self.mark_stop_timeout()
        return stopped

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def is_stopping(self) -> bool:
        return self._stop_event.is_set() and self._thread.is_alive()

    def mark_duplicate_start_refused(self) -> None:
        self._mark_stopping(DUPLICATE_WORKER_ERROR)

    def mark_stop_timeout(self) -> None:
        self._mark_stopping("Stop timeout; capture thread still exiting")

    def snapshot(self, include_frame: bool = True) -> CameraStreamSnapshot:
        with self._lock:
            can_publish_frame = (
                include_frame
                and self._status == STREAM_LIVE
                and not self._stop_event.is_set()
                and self._frame is not None
            )
            frame = self._frame.copy() if can_publish_frame else None
            return CameraStreamSnapshot(
                key=self.config.key,
                status=self._status,
                source_type=self.config.source_type,
                source=str(self.config.source),
                frame=frame,
                frame_timestamp=self._frame_timestamp,
                frame_count=self._frame_count,
                failed_read_count=self._failed_read_count,
                last_error=self._last_error,
                thread_alive=self._thread.is_alive(),
                backend=self._backend,
                timeout_mode=self._timeout_mode,
            )

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            capture = self._open_capture()
            if self._stop_event.is_set():
                return
            if not capture.isOpened():
                self._set_state(
                    status=STREAM_FAILED,
                    last_error=self._open_failure_message(),
                )
                return

            self._set_state(status=STREAM_LIVE, last_error="")
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if self._stop_event.is_set():
                    break
                if not ok or frame is None:
                    error = self._read_failure_message()
                    self._set_failure(error)
                    break
                self._set_frame(frame)
        except Exception as exc:  # OpenCV backends can raise on device/network errors.
            self._set_failure(str(exc))
        finally:
            # VideoCapture has strict thread ownership: only this worker creates,
            # reads, and releases its local capture.
            if capture is not None:
                capture.release()
            if self._stop_event.is_set():
                self._set_state(status=STREAM_STOPPED)

    def _open_capture(self) -> cv2.VideoCapture:
        if self.config.source_type != "rtsp":
            self._set_capture_diagnostics(backend="AUTO", timeout_mode="not-configured")
            return cv2.VideoCapture(self.config.source)

        timeout_properties = (
            getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None),
            getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None),
        )
        ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
        if ffmpeg_backend is not None and all(value is not None for value in timeout_properties):
            params = [
                timeout_properties[0],
                DEFAULT_RTSP_OPEN_TIMEOUT_MS,
                timeout_properties[1],
                DEFAULT_RTSP_READ_TIMEOUT_MS,
            ]
            try:
                capture = cv2.VideoCapture(self.config.source, ffmpeg_backend, params)
                backend = self._capture_backend_name(capture, fallback="FFMPEG")
                self._set_capture_diagnostics(backend=backend, timeout_mode="ffmpeg-properties")
                return capture
            except (TypeError, cv2.error) as exc:
                self._set_capture_diagnostics(
                    backend="FFMPEG",
                    timeout_mode=f"unsupported-fallback: {exc}",
                )

        # Older OpenCV builds may reject constructor timeout parameters. This
        # fallback preserves worker ownership but cannot guarantee bounded I/O.
        backend = ffmpeg_backend if ffmpeg_backend is not None else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.config.source, backend)
        self._set_capture_diagnostics(
            backend=self._capture_backend_name(capture, fallback="FFMPEG"),
            timeout_mode="unsupported-fallback",
        )
        return capture

    def _capture_backend_name(self, capture: cv2.VideoCapture, fallback: str) -> str:
        if not capture.isOpened():
            return fallback
        try:
            return capture.getBackendName()
        except cv2.error:
            return fallback

    def _open_failure_message(self) -> str:
        with self._lock:
            timeout_mode = self._timeout_mode
        if timeout_mode == "ffmpeg-properties":
            return f"Failed to open source (timeout limit {DEFAULT_RTSP_OPEN_TIMEOUT_MS} ms)"
        return "Failed to open source; backend does not support a verified open timeout"

    def _read_failure_message(self) -> str:
        with self._lock:
            timeout_mode = self._timeout_mode
        if timeout_mode == "ffmpeg-properties":
            return f"Frame read failed or timed out ({DEFAULT_RTSP_READ_TIMEOUT_MS} ms)"
        return "Frame read failed"

    def _set_capture_diagnostics(self, backend: str, timeout_mode: str) -> None:
        with self._lock:
            self._backend = backend
            self._timeout_mode = timeout_mode

    def _set_frame(self, frame: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            self._frame = frame
            self._frame_timestamp = now
            self._frame_count += 1
            self._status = STREAM_LIVE
            self._last_error = ""

    def _set_failure(self, error: str) -> None:
        with self._lock:
            self._failed_read_count += 1
            self._status = STREAM_FAILED
            self._frame = None
            self._last_error = error

    def _set_state(self, status: str, last_error: str | None = None) -> None:
        with self._lock:
            self._status = status
            if status != STREAM_LIVE:
                # Keep counters and the last timestamp for diagnostics, but never
                # retain an image that could be mistaken for a current live frame.
                self._frame = None
            if last_error is not None:
                self._last_error = last_error

    def _mark_stopping(self, last_error: str | None = None) -> None:
        with self._lock:
            if self._status == STREAM_STOPPED:
                return
            self._status = STREAM_STOPPING
            self._frame = None
            if last_error is not None:
                self._last_error = last_error


class CameraStreamManager:
    """Small manager that keeps live capture threads out of the GUI thread."""

    def __init__(self):
        self._lock = Lock()
        self._workers: dict[str, _CameraStreamWorker] = {}
        self._states: dict[str, CameraStreamSnapshot] = {}

    def start(self, configs: list[CameraStreamConfig]) -> None:
        if not self._only_has_stopping_workers():
            self.stop()
        self._cleanup_finished_workers()
        with self._lock:
            for config in configs:
                existing = self._workers.get(config.key)
                if existing is not None and existing.is_alive():
                    existing.mark_duplicate_start_refused()
                    self._states[config.key] = existing.snapshot(include_frame=False)
                    continue
                if not config.enabled:
                    self._states[config.key] = CameraStreamSnapshot(
                        key=config.key,
                        status=STREAM_IDLE,
                        source_type=config.source_type,
                        source=str(config.source),
                    )
                    continue
                if config.source_type == "image_dir":
                    self._states[config.key] = CameraStreamSnapshot(
                        key=config.key,
                        status=STREAM_IDLE,
                        source_type=config.source_type,
                        source=str(config.source),
                    )
                    continue
                if config.source in ("", None):
                    self._states[config.key] = CameraStreamSnapshot(
                        key=config.key,
                        status=STREAM_FAILED,
                        source_type=config.source_type,
                        source="",
                        failed_read_count=1,
                        last_error="Source is empty",
                    )
                    continue
                worker = _CameraStreamWorker(config)
                self._workers[config.key] = worker
                self._states[config.key] = worker.snapshot(include_frame=False)
                worker.start()

    def stop(self, timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> bool:
        with self._lock:
            workers = list(self._workers.items())

        for _, worker in workers:
            worker.request_stop()

        deadline = time.monotonic() + max(0.0, timeout)
        all_stopped = True
        for key, worker in workers:
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                worker.wait(remaining)
            with self._lock:
                self._states[key] = worker.snapshot(include_frame=False)
                if worker.is_alive():
                    all_stopped = False
                    worker.mark_stop_timeout()
                    self._states[key] = worker.snapshot(include_frame=False)
                else:
                    # Re-snapshot after the liveness decision so a worker that
                    # exited at the deadline cannot leave a stale Stopping state.
                    self._states[key] = worker.snapshot(include_frame=False)
                    self._workers.pop(key, None)
        if not all_stopped:
            with self._lock:
                all_stopped = not any(
                    worker.is_alive() for worker in self._workers.values()
                )
        return all_stopped

    def has_active_workers(self) -> bool:
        self._cleanup_finished_workers()
        with self._lock:
            return any(worker.is_alive() for worker in self._workers.values())

    def snapshots(self, include_frames: bool = True) -> dict[str, CameraStreamSnapshot]:
        self._cleanup_finished_workers()
        with self._lock:
            workers = dict(self._workers)
            states = {
                key: replace(
                    snapshot,
                    frame=(
                        snapshot.frame.copy()
                        if include_frames
                        and snapshot.status == STREAM_LIVE
                        and snapshot.frame is not None
                        else None
                    ),
                )
                for key, snapshot in self._states.items()
            }
        for key, worker in workers.items():
            states[key] = worker.snapshot(include_frame=include_frames)
        return states

    def latest_frames(
        self,
        max_frame_age_seconds: float | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, CameraStreamSnapshot]]:
        snapshots = self.snapshots(include_frames=True)
        max_age = (
            None
            if max_frame_age_seconds is None
            else max(0.0, float(max_frame_age_seconds))
        )
        now = time.time() if max_age is not None else 0.0
        frames = {
            key: snapshot.frame
            for key, snapshot in snapshots.items()
            if snapshot.status == STREAM_LIVE
            and snapshot.frame is not None
            and (
                max_age is None
                or (
                    snapshot.frame_timestamp > 0.0
                    and now - snapshot.frame_timestamp <= max_age
                )
            )
        }
        return frames, snapshots

    def reconnect(self, key: str, config: CameraStreamConfig) -> None:
        """Reserved simple reconnect entrypoint for future UI actions."""
        with self._lock:
            worker = self._workers.get(key)
        if worker is not None:
            worker.stop()
            if worker.is_alive():
                worker.mark_duplicate_start_refused()
                with self._lock:
                    self._states[key] = worker.snapshot(include_frame=False)
                return
            with self._lock:
                self._workers.pop(key, None)
        replacement = _CameraStreamWorker(config)
        with self._lock:
            self._workers[key] = replacement
            self._states[key] = replacement.snapshot(include_frame=False)
        replacement.start()

    def _cleanup_finished_workers(self) -> None:
        with self._lock:
            finished = [
                (key, worker)
                for key, worker in self._workers.items()
                if not worker.is_alive()
            ]
            for key, worker in finished:
                self._states[key] = worker.snapshot(include_frame=False)
                self._workers.pop(key, None)

    def _only_has_stopping_workers(self) -> bool:
        with self._lock:
            workers = list(self._workers.values())
        return bool(workers) and all(worker.is_stopping() for worker in workers)

    def diagnostics(self) -> dict[str, dict[str, Any]]:
        snapshots = self.snapshots(include_frames=False)
        return {
            key: {
                "status": item.status,
                "source_type": item.source_type,
                "source": item.source,
                "frame_timestamp": item.frame_timestamp,
                "frame_count": item.frame_count,
                "failed_read_count": item.failed_read_count,
                "last_error": item.last_error,
                "thread_alive": item.thread_alive,
                "backend": item.backend,
                "timeout_mode": item.timeout_mode,
            }
            for key, item in snapshots.items()
        }
