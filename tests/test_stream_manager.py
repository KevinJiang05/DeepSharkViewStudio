from __future__ import annotations

from threading import Event, get_ident
import time
import unittest
from unittest.mock import patch

import numpy as np

from deep_shark_studio.stream_manager import (
    CameraStreamConfig,
    CameraStreamManager,
    DUPLICATE_WORKER_ERROR,
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

            manager.stop(timeout=0.05)
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


if __name__ == "__main__":
    unittest.main()
