"""Asynchronous live stitching worker for DeepShark View Studio."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Lock, Thread
import time
from typing import Any

import numpy as np

from .calibration_candidate import CandidatePanoramaProcessor
from .stitch_runtime_controller import RuntimeStitchController, RuntimeStitchResult
from .stitch_runtime_modes import RuntimeStitchConfig
from .stitcher import SurroundStitcher


@dataclass(frozen=True)
class StitchRequest:
    session_id: int
    request_id: int
    frames: dict[str, np.ndarray]


@dataclass(frozen=True)
class StitchResult:
    session_id: int
    request_id: int
    result_id: int
    warped: dict[str, np.ndarray]
    canvas: np.ndarray | None
    elapsed_ms: float
    error: str = ""
    runtime_mode: str = ""
    runtime_status: str = ""
    runtime_warnings: tuple[str, ...] = ()
    runtime_metrics: dict[str, Any] | None = None


class StitchProcessingWorker:
    """Single long-lived worker that processes only the newest pending request."""

    def __init__(self):
        self._condition = Condition()
        self._lock = Lock()
        self._thread = Thread(target=self._run, name="StitchProcessingWorker")
        self._shutdown = False
        self._pending_request: StitchRequest | None = None
        self._latest_result: StitchResult | None = None
        self._stitcher: (
            RuntimeStitchController | CandidatePanoramaProcessor | None
        ) = None
        self._config: dict[str, Any] = {}
        self._max_input_width: int | None = None
        self._use_intrinsics = False
        self._request_id = 0
        self._result_id = 0
        self._active_session_id = 0
        self._started = False

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._thread.start()

    def configure(
        self,
        session_id: int,
        config: dict[str, Any],
        max_input_width: int | None,
        use_intrinsics: bool,
        processor_mode: str = "template",
        candidate_directory: str | None = None,
        runtime_config: RuntimeStitchConfig | None = None,
    ) -> None:
        with self._condition:
            self._active_session_id = session_id
            self._config = dict(config)
            self._max_input_width = max_input_width
            self._use_intrinsics = use_intrinsics
            if processor_mode == "candidate":
                if not candidate_directory:
                    raise ValueError(
                        "Candidate processor requires a candidate directory."
                    )
                self._stitcher = CandidatePanoramaProcessor(
                    candidate_directory
                )
            else:
                stitcher = SurroundStitcher(
                    self._config,
                    max_input_width=max_input_width,
                    use_intrinsics=use_intrinsics,
                )
                self._stitcher = RuntimeStitchController(
                    stitcher,
                    runtime_config or RuntimeStitchConfig(),
                )
            self._pending_request = None
            self._latest_result = None
            self._condition.notify_all()

    def submit_latest(self, session_id: int, frames: dict[str, np.ndarray]) -> int:
        with self._condition:
            if self._shutdown:
                return self._request_id
            if self._stitcher is None or session_id != self._active_session_id:
                return self._request_id
            self._request_id += 1
            self._pending_request = StitchRequest(
                session_id=session_id,
                request_id=self._request_id,
                frames=dict(frames),
            )
            self._condition.notify_all()
            return self._request_id

    def invalidate_session(self, session_id: int) -> None:
        with self._condition:
            self._active_session_id = session_id
            self._pending_request = None
            self._latest_result = None
            self._condition.notify_all()

    def take_latest_result(self) -> StitchResult | None:
        with self._lock:
            return self._latest_result

    def shutdown(self, timeout: float = 2.0) -> bool:
        with self._condition:
            self._shutdown = True
            self._pending_request = None
            self._condition.notify_all()
        if self._started:
            self._thread.join(timeout=max(0.0, timeout))
            return not self._thread.is_alive()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown and self._pending_request is None:
                    self._condition.wait()
                if self._shutdown:
                    return
                request = self._pending_request
                self._pending_request = None
                stitcher = self._stitcher

            if request is None or stitcher is None:
                continue

            start_time = time.perf_counter()
            warped: dict[str, np.ndarray] = {}
            canvas: np.ndarray | None = None
            error = ""
            runtime_mode = ""
            runtime_status = ""
            runtime_warnings: tuple[str, ...] = ()
            runtime_metrics: dict[str, Any] | None = None
            try:
                processed = stitcher.process(request.frames)
                if isinstance(processed, RuntimeStitchResult):
                    warped = processed.warped
                    canvas = processed.canvas
                    runtime_mode = processed.mode.value
                    runtime_status = processed.status
                    runtime_warnings = processed.warnings
                    runtime_metrics = processed.metrics
                else:
                    warped, canvas = processed
            except Exception as exc:
                error = str(exc)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            with self._lock:
                self._result_id += 1
                self._latest_result = StitchResult(
                    session_id=request.session_id,
                    request_id=request.request_id,
                    result_id=self._result_id,
                    warped=warped,
                    canvas=canvas,
                    elapsed_ms=elapsed_ms,
                    error=error,
                    runtime_mode=runtime_mode,
                    runtime_status=runtime_status,
                    runtime_warnings=runtime_warnings,
                    runtime_metrics=runtime_metrics,
                )


class StitchProcessingManager:
    """Facade used by MainWindow to keep live stitching off the GUI thread."""

    def __init__(self):
        self._worker = StitchProcessingWorker()

    def start(self) -> None:
        self._worker.start()

    def configure(
        self,
        session_id: int,
        config: dict[str, Any],
        max_input_width: int | None,
        use_intrinsics: bool,
        processor_mode: str = "template",
        candidate_directory: str | None = None,
        runtime_config: RuntimeStitchConfig | None = None,
    ) -> None:
        self.start()
        self._worker.configure(
            session_id,
            config,
            max_input_width,
            use_intrinsics,
            processor_mode=processor_mode,
            candidate_directory=candidate_directory,
            runtime_config=runtime_config,
        )

    def submit_latest(self, session_id: int, frames: dict[str, np.ndarray]) -> int:
        self.start()
        return self._worker.submit_latest(session_id, frames)

    def invalidate_session(self, session_id: int) -> None:
        self._worker.invalidate_session(session_id)

    def take_latest_result(self) -> StitchResult | None:
        return self._worker.take_latest_result()

    def shutdown(self, timeout: float = 2.0) -> bool:
        return self._worker.shutdown(timeout=timeout)
