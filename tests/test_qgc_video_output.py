from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from deep_shark_studio.qgc.video_output import (
    FfmpegVideoSink,
    VideoOutputConfig,
)


class _BlockingStdin:
    def __init__(self) -> None:
        self.write_started = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.payloads: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.write_started.set()
        self.release.wait(0.4)
        if self.closed:
            raise BrokenPipeError("stdin closed")
        self.payloads.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True
        self.release.set()


class _FakeStderr:
    def __init__(self, lines: tuple[bytes, ...] = ()) -> None:
        self.lines = lines
        self.read_started = threading.Event()
        self.closed = False

    def __iter__(self):
        self.read_started.set()
        yield from self.lines

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    _next_pid = 41000

    def __init__(self, stderr_lines: tuple[bytes, ...] = ()) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.stdin = _BlockingStdin()
        self.stderr = _FakeStderr(stderr_lines)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0
        self.stdin.release.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdin.release.set()


class _FailingStdin(_BlockingStdin):
    def write(self, payload: bytes) -> int:
        self.write_started.set()
        raise BrokenPipeError("simulated encoder exit")


class _ThreadState:
    def __init__(self, alive: bool) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class FfmpegVideoSinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.full((12, 20, 3), 17, dtype=np.uint8)
        self.config = VideoOutputConfig(
            kind="udp_mpegts",
            url="udp://127.0.0.1:5600?pkt_size=1316",
            fps=15.0,
            ffmpeg_path="ffmpeg",
        )

    def test_write_never_waits_for_the_ffmpeg_pipe(self) -> None:
        process = _FakeProcess()
        sink = FfmpegVideoSink(self.config)
        with (
            patch.object(sink, "preflight"),
            patch(
                "deep_shark_studio.qgc.video_output.subprocess.Popen",
                return_value=process,
            ),
        ):
            started = time.monotonic()
            accepted = sink.write(self.frame)
            elapsed = time.monotonic() - started

            self.assertTrue(accepted)
            self.assertLess(elapsed, 0.1)
            self.assertTrue(process.stdin.write_started.wait(0.5))
            process.stdin.release.set()
            self.assertTrue(
                _wait_until(lambda: sink.status().frames_written == 1)
            )
            sink.close()

    def test_bounded_latest_frame_queue_drops_superseded_frames(self) -> None:
        process = _FakeProcess()
        sink = FfmpegVideoSink(self.config)
        with (
            patch.object(sink, "preflight"),
            patch(
                "deep_shark_studio.qgc.video_output.subprocess.Popen",
                return_value=process,
            ),
        ):
            sink.write(self.frame)
            self.assertTrue(process.stdin.write_started.wait(0.5))
            sink.write(self.frame + 1)
            sink.write(self.frame + 2)

            status = sink.status()
            self.assertEqual(3, status.frames_submitted)
            self.assertGreaterEqual(status.frames_dropped, 1)
            process.stdin.release.set()
            sink.close()

    def test_stderr_is_consumed_and_exposed_in_status(self) -> None:
        process = _FakeProcess((b"encoder warning\n", b"network failed\n"))
        sink = FfmpegVideoSink(self.config)
        with (
            patch.object(sink, "preflight"),
            patch(
                "deep_shark_studio.qgc.video_output.subprocess.Popen",
                return_value=process,
            ),
        ):
            sink.open(20, 12, 15.0)
            self.assertTrue(process.stderr.read_started.wait(0.5))
            self.assertTrue(
                _wait_until(
                    lambda: "network failed" in sink.status().last_stderr_line
                )
            )
            status = sink.status()
            self.assertIn("encoder warning", "\n".join(status.recent_stderr))
            self.assertEqual(process.pid, status.pid)
            sink.close()

    def test_stop_and_close_are_idempotent_and_reap_the_process(self) -> None:
        process = _FakeProcess()
        sink = FfmpegVideoSink(self.config)
        with (
            patch.object(sink, "preflight"),
            patch(
                "deep_shark_studio.qgc.video_output.subprocess.Popen",
                return_value=process,
            ),
        ):
            sink.open(20, 12, 15.0)
            sink.close()
            sink.close()

        status = sink.status()
        self.assertEqual("stopped", status.state)
        self.assertFalse(status.running)
        self.assertIsNotNone(status.exit_code)
        self.assertTrue(process.terminated or process.stdin.closed)

    def test_broken_pipe_restarts_with_observable_reason(self) -> None:
        first = _FakeProcess((b"simulated failure\n",))
        first.stdin = _FailingStdin()
        second = _FakeProcess()
        config = VideoOutputConfig(
            kind="udp_mpegts",
            url=self.config.url,
            fps=15.0,
            ffmpeg_path="ffmpeg",
            restart_limit=2,
            restart_backoff_seconds=0.0,
        )
        sink = FfmpegVideoSink(config)
        with (
            patch.object(sink, "preflight"),
            patch(
                "deep_shark_studio.qgc.video_output.subprocess.Popen",
                side_effect=[first, second],
            ),
        ):
            sink.write(self.frame)
            self.assertTrue(
                _wait_until(
                    lambda: sink.status().restart_count == 1
                    and sink.status().pid == second.pid
                )
            )
            status = sink.status()
            self.assertEqual("running", status.state)
            self.assertIn("BrokenPipeError", status.last_restart_reason)
            sink.write(self.frame)
            self.assertTrue(second.stdin.write_started.wait(0.5))
            second.stdin.release.set()
            self.assertTrue(
                _wait_until(lambda: sink.status().frames_written == 1)
            )
            sink.close()

    def test_status_does_not_claim_running_for_an_exited_process(self) -> None:
        process = _FakeProcess()
        process.returncode = 17
        sink = FfmpegVideoSink(self.config)
        with sink._condition:
            sink.process = process
            sink._writer_thread = _ThreadState(alive=True)  # type: ignore[assignment]
            sink._state = "running"

        status = sink.status()

        self.assertFalse(status.running)
        self.assertEqual("restarting", status.state)
        self.assertIsNone(status.pid)
        self.assertEqual(17, status.exit_code)
        self.assertIn("17", status.last_error)


if __name__ == "__main__":
    unittest.main()
