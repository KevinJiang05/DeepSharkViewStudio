from __future__ import annotations

from threading import Event, get_ident
import time
import unittest
from unittest.mock import patch

import numpy as np

from deep_shark_studio.stream_manager import (
    CameraStreamConfig,
    CameraStreamManager,
    CameraStreamSnapshot,
    DUPLICATE_WORKER_ERROR,
    STREAM_FAILED,
    STREAM_LIVE,
    STREAM_STOPPED,
    STREAM_STOPPING,
)


class BlockingCapture:
    def __init__(self, *args, **kwargs):
        self.read_started = Event()
        self.allow_read_to_return = Event()
        self.read_thread_id: int | None = None
        self.release_thread_id: int | None = None
        self.release_count = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def getBackendName(self) -> str:  # noqa: N802
        return "FFMPEG"

    def read(self):
        self.read_thread_id = get_ident()
        self.read_started.set()
        self.allow_read_to_return.wait()
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_thread_id = get_ident()
        self.release_count += 1


class FrameThenBlockingCapture:
    def __init__(self, *args, **kwargs):
        self.first_frame_read = Event()
        self.blocking_read_started = Event()
        self.allow_blocking_read_to_return = Event()
        self.read_thread_id: int | None = None
        self.release_thread_id: int | None = None
        self.release_count = 0
        self.read_count = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def getBackendName(self) -> str:  # noqa: N802
        return "FFMPEG"

    def read(self):
        self.read_thread_id = get_ident()
        self.read_count += 1
        if self.read_count == 1:
            self.first_frame_read.set()
            return True, np.ones((4, 4, 3), dtype=np.uint8)
        self.blocking_read_started.set()
        self.allow_blocking_read_to_return.wait()
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_thread_id = get_ident()
        self.release_count += 1


class FrameFailureWithBlockingReleaseCapture:
    def __init__(self, *args, **kwargs):
        self.first_frame_read = Event()
        self.allow_failure = Event()
        self.release_started = Event()
        self.allow_release_to_return = Event()
        self.read_thread_id: int | None = None
        self.release_thread_id: int | None = None
        self.release_count = 0
        self.read_count = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def getBackendName(self) -> str:  # noqa: N802
        return "FFMPEG"

    def read(self):
        self.read_thread_id = get_ident()
        self.read_count += 1
        if self.read_count == 1:
            self.first_frame_read.set()
            return True, np.ones((4, 4, 3), dtype=np.uint8)
        self.allow_failure.wait()
        return False, None

    def release(self) -> None:
        self.release_thread_id = get_ident()
        self.release_count += 1
        self.release_started.set()
        self.allow_release_to_return.wait()


class ImmediateCapture:
    def __init__(self, *args, **kwargs):
        self.first_frame_read = Event()
        self.read_thread_id: int | None = None
        self.release_thread_id: int | None = None
        self.release_count = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def getBackendName(self) -> str:  # noqa: N802
        return "FFMPEG"

    def read(self):
        self.read_thread_id = get_ident()
        self.first_frame_read.set()
        time.sleep(0.001)
        return True, np.ones((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_thread_id = get_ident()
        self.release_count += 1


class CameraStreamManagerOwnershipTests(unittest.TestCase):
    def test_stop_never_releases_capture_from_manager_thread(self) -> None:
        captures: list[BlockingCapture] = []

        def create_capture(*args, **kwargs):
            capture = BlockingCapture(*args, **kwargs)
            captures.append(capture)
            return capture

        config = CameraStreamConfig(
            key="front_left",
            source_type="rtsp",
            source="rtsp://example.invalid/stream",
        )
        manager = CameraStreamManager()
        with patch("deep_shark_studio.stream_manager.cv2.VideoCapture", side_effect=create_capture):
            manager.start([config])
            self.assertTrue(captures[0].read_started.wait(1.0))

            stopped = manager.stop(timeout=0.05)
            self.assertIs(stopped, False)
            snapshot = manager.snapshots(include_frames=False)["front_left"]
            self.assertEqual(STREAM_STOPPING, snapshot.status)
            self.assertTrue(snapshot.thread_alive)
            self.assertEqual(0, captures[0].release_count)

            manager.start([config])
            snapshot = manager.snapshots(include_frames=False)["front_left"]
            self.assertEqual(DUPLICATE_WORKER_ERROR, snapshot.last_error)
            self.assertEqual(1, len(captures))

            captures[0].allow_read_to_return.set()
            deadline = time.monotonic() + 1.0
            while manager.has_active_workers() and time.monotonic() < deadline:
                time.sleep(0.01)

        snapshot = manager.snapshots(include_frames=False)["front_left"]
        self.assertEqual(STREAM_STOPPED, snapshot.status)
        self.assertFalse(snapshot.thread_alive)
        self.assertEqual(1, captures[0].release_count)
        self.assertEqual(captures[0].read_thread_id, captures[0].release_thread_id)

    def test_stop_timeout_hides_last_frame_and_worker_is_daemon(self) -> None:
        capture = FrameThenBlockingCapture()
        config = CameraStreamConfig(
            key="front_left",
            source_type="rtsp",
            source="rtsp://example.invalid/stream",
        )
        manager = CameraStreamManager()
        with patch(
            "deep_shark_studio.stream_manager.cv2.VideoCapture",
            return_value=capture,
        ):
            manager.start([config])
            self.assertTrue(capture.first_frame_read.wait(1.0))
            self.assertTrue(capture.blocking_read_started.wait(1.0))
            worker = manager._workers["front_left"]

            stopped = manager.stop(timeout=0.05)
            frames, snapshots = manager.latest_frames()
            snapshot = snapshots["front_left"]
            worker_is_daemon = worker._thread.daemon

            capture.allow_blocking_read_to_return.set()
            deadline = time.monotonic() + 1.0
            while manager.has_active_workers() and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertIs(stopped, False)
        self.assertTrue(worker_is_daemon)
        self.assertEqual(STREAM_STOPPING, snapshot.status)
        self.assertIsNone(snapshot.frame)
        self.assertNotIn("front_left", frames)
        self.assertEqual(1, snapshot.frame_count)
        self.assertGreater(snapshot.frame_timestamp, 0.0)
        self.assertEqual(1, capture.release_count)
        self.assertEqual(capture.read_thread_id, capture.release_thread_id)

    def test_read_failure_hides_last_frame_before_release_finishes(self) -> None:
        capture = FrameFailureWithBlockingReleaseCapture()
        config = CameraStreamConfig(
            key="front_left",
            source_type="rtsp",
            source="rtsp://example.invalid/stream",
        )
        manager = CameraStreamManager()
        with patch(
            "deep_shark_studio.stream_manager.cv2.VideoCapture",
            return_value=capture,
        ):
            manager.start([config])
            self.assertTrue(capture.first_frame_read.wait(1.0))
            capture.allow_failure.set()
            self.assertTrue(capture.release_started.wait(1.0))

            frames, snapshots = manager.latest_frames()
            snapshot = snapshots["front_left"]

            capture.allow_release_to_return.set()
            deadline = time.monotonic() + 1.0
            while manager.has_active_workers() and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertEqual(STREAM_FAILED, snapshot.status)
        self.assertTrue(snapshot.thread_alive)
        self.assertIsNone(snapshot.frame)
        self.assertNotIn("front_left", frames)
        self.assertEqual(1, snapshot.frame_count)
        self.assertGreater(snapshot.frame_timestamp, 0.0)
        self.assertEqual(1, capture.release_count)
        self.assertEqual(capture.read_thread_id, capture.release_thread_id)

    def test_stop_returns_true_when_worker_exits_within_deadline(self) -> None:
        capture = ImmediateCapture()
        config = CameraStreamConfig(
            key="front_left",
            source_type="rtsp",
            source="rtsp://example.invalid/stream",
        )
        manager = CameraStreamManager()
        with patch(
            "deep_shark_studio.stream_manager.cv2.VideoCapture",
            return_value=capture,
        ):
            manager.start([config])
            self.assertTrue(capture.first_frame_read.wait(1.0))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if manager.snapshots(include_frames=False)["front_left"].frame_count >= 1:
                    break
                time.sleep(0.01)
            worker = manager._workers["front_left"]
            stopped = manager.stop(timeout=1.0)
            worker.mark_stop_timeout()

        snapshot = manager.snapshots(include_frames=True)["front_left"]
        worker_snapshot = worker.snapshot(include_frame=True)
        self.assertTrue(stopped)
        self.assertEqual(STREAM_STOPPED, snapshot.status)
        self.assertEqual(STREAM_STOPPED, worker_snapshot.status)
        self.assertFalse(snapshot.thread_alive)
        self.assertIsNone(snapshot.frame)
        self.assertGreaterEqual(snapshot.frame_count, 1)
        self.assertGreater(snapshot.frame_timestamp, 0.0)
        self.assertEqual(1, capture.release_count)
        self.assertEqual(capture.read_thread_id, capture.release_thread_id)


class CameraStreamManagerFrameFilteringTests(unittest.TestCase):
    def test_latest_frames_only_returns_live_fresh_frames(self) -> None:
        now = time.time()
        manager = CameraStreamManager()
        manager._states = {
            "fresh": CameraStreamSnapshot(
                key="fresh",
                status=STREAM_LIVE,
                frame=np.ones((2, 2, 3), dtype=np.uint8),
                frame_timestamp=now - 0.25,
                frame_count=1,
                thread_alive=True,
            ),
            "stale": CameraStreamSnapshot(
                key="stale",
                status=STREAM_LIVE,
                frame=np.ones((2, 2, 3), dtype=np.uint8),
                frame_timestamp=now - 5.0,
                frame_count=1,
                thread_alive=True,
            ),
            "failed": CameraStreamSnapshot(
                key="failed",
                status=STREAM_FAILED,
                frame=np.ones((2, 2, 3), dtype=np.uint8),
                frame_timestamp=now,
                frame_count=1,
                thread_alive=True,
            ),
        }

        unbounded_frames, _ = manager.latest_frames()
        fresh_frames, snapshots = manager.latest_frames(max_frame_age_seconds=1.0)

        self.assertEqual({"fresh", "stale"}, set(unbounded_frames))
        self.assertEqual({"fresh"}, set(fresh_frames))
        self.assertIsNone(snapshots["failed"].frame)


if __name__ == "__main__":
    unittest.main()
