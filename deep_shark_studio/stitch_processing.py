"""Asynchronous live stitching worker for DeepShark View Studio."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread
import time
from typing import Any

import numpy as np

from .calibration_candidate import CandidatePanoramaProcessor
from .stitch_runtime_controller import RuntimeStitchController, RuntimeStitchResult
from .stitch_runtime_modes import (
    RuntimeStitchConfig,
    resolve_effective_runtime_status,
)
from .stitcher import SurroundStitcher


@dataclass(frozen=True)
class StitchRequest:
    session_id: int
    request_id: int
    configuration_id: int
    processor_mode: str
    runtime_mode: str
    runtime_status: str
    projection_source: str
    frames: dict[str, np.ndarray]


@dataclass(frozen=True)
class StitchResult:
    session_id: int
    request_id: int
    result_id: int
    warped: dict[str, np.ndarray]
    canvas: np.ndarray | None
    elapsed_ms: float
    configuration_id: int = 0
    processor_mode: str = ""
    projection_source: str = ""
    error: str = ""
    runtime_mode: str = ""
    runtime_status: str = ""
    runtime_warnings: tuple[str, ...] = ()
    runtime_metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class StitchProcessingState:
    """Observable worker configuration, lifecycle state, and telemetry.

    Counters are lifetime totals for accepted requests. ``superseded`` is the
    subset of ``dropped`` replaced while still pending; ``dropped`` also counts
    pending or in-flight work invalidated by reconfigure, session invalidation,
    or shutdown. Successful reconfigure and session invalidation reset rolling
    timing/FPS samples, which otherwise describe only successful results from
    the current configuration.
    """

    session_id: int
    configuration_id: int
    processor_mode: str
    runtime_mode: str
    runtime_status: str
    projection_source: str
    configured: bool
    shutdown_requested: bool
    thread_started: bool
    thread_alive: bool
    thread_daemon: bool
    submitted: int = 0
    superseded: int = 0
    dropped: int = 0
    completed: int = 0
    failed: int = 0
    busy: bool = False
    rolling_elapsed_p50_ms: float = 0.0
    rolling_elapsed_p95_ms: float = 0.0
    rolling_result_fps: float = 0.0


class StitchProcessingWorker:
    """Single long-lived worker that processes only the newest pending request."""

    _TELEMETRY_SAMPLE_LIMIT = 120
    _RESULT_FPS_WINDOW_SECONDS = 5.0

    def __init__(self):
        self._condition = Condition()
        self._thread = Thread(
            target=self._run,
            name="StitchProcessingWorker",
            daemon=True,
        )
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
        self._configuration_id = 0
        self._processor_mode = ""
        self._runtime_mode = ""
        self._runtime_status = ""
        self._projection_source = ""
        self._started = False
        self._submitted = 0
        self._superseded = 0
        self._dropped = 0
        self._completed = 0
        self._failed = 0
        self._busy = False
        self._rolling_elapsed_ms: deque[float] = deque(
            maxlen=self._TELEMETRY_SAMPLE_LIMIT,
        )
        self._rolling_result_times: deque[float] = deque(
            maxlen=self._TELEMETRY_SAMPLE_LIMIT,
        )

    def start(self) -> None:
        with self._condition:
            if self._started or self._shutdown:
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
        candidate_use_opencl: bool = False,
        runtime_config: RuntimeStitchConfig | None = None,
    ) -> None:
        if processor_mode not in {"template", "candidate"}:
            raise ValueError(f"Unknown stitch processor mode: {processor_mode}")
        config_copy = dict(config)
        normalized_runtime_config = (
            runtime_config or RuntimeStitchConfig()
        ).normalized()
        if processor_mode == "candidate":
            if not candidate_directory:
                raise ValueError(
                    "Candidate processor requires a candidate directory."
                )
            processor: RuntimeStitchController | CandidatePanoramaProcessor = (
                CandidatePanoramaProcessor(
                    candidate_directory,
                    use_opencl=candidate_use_opencl,
                )
            )
            effective_runtime_config = normalized_runtime_config
            runtime_mode = "far_field"
            projection_source = "b2_candidate_projection"
        else:
            stitcher = SurroundStitcher(
                config_copy,
                max_input_width=max_input_width,
                use_intrinsics=use_intrinsics,
            )
            processor = RuntimeStitchController(
                stitcher,
                normalized_runtime_config,
            )
            effective_runtime_config = getattr(
                processor,
                "config",
                normalized_runtime_config,
            ).normalized()
            runtime_mode = (
                "near_field"
                if effective_runtime_config.mode.value == "near_field"
                else "far_field"
            )
            projection_source = effective_runtime_config.projection_source.value
            far_field_candidate = getattr(
                processor,
                "far_field_layout_candidate",
                None,
            )
            if (
                effective_runtime_config.use_far_field_custom_layout
                and far_field_candidate is not None
            ):
                projection_source = str(far_field_candidate.projection_source)
        runtime_status = resolve_effective_runtime_status(
            effective_runtime_config,
            processor_mode=processor_mode,
        )

        with self._condition:
            if self._shutdown:
                raise RuntimeError("Stitch processing worker is shut down.")
            self._active_session_id = session_id
            self._configuration_id += 1
            self._config = config_copy
            self._max_input_width = max_input_width
            self._use_intrinsics = use_intrinsics
            self._stitcher = processor
            self._processor_mode = processor_mode
            self._runtime_mode = runtime_mode
            self._runtime_status = runtime_status
            self._projection_source = projection_source
            if self._pending_request is not None:
                self._dropped += 1
            self._pending_request = None
            self._latest_result = None
            self._reset_rolling_telemetry_locked()
            self._condition.notify_all()

    def submit_latest(self, session_id: int, frames: dict[str, np.ndarray]) -> int:
        with self._condition:
            if self._shutdown:
                return self._request_id
            if self._stitcher is None or session_id != self._active_session_id:
                return self._request_id
            self._request_id += 1
            self._submitted += 1
            if self._pending_request is not None:
                self._superseded += 1
                self._dropped += 1
            self._pending_request = StitchRequest(
                session_id=session_id,
                request_id=self._request_id,
                configuration_id=self._configuration_id,
                processor_mode=self._processor_mode,
                runtime_mode=self._runtime_mode,
                runtime_status=self._runtime_status,
                projection_source=self._projection_source,
                frames=dict(frames),
            )
            self._condition.notify_all()
            return self._request_id

    def invalidate_session(self, session_id: int) -> None:
        with self._condition:
            self._active_session_id = session_id
            self._configuration_id += 1
            self._stitcher = None
            self._processor_mode = ""
            self._runtime_mode = ""
            self._runtime_status = ""
            self._projection_source = ""
            if self._pending_request is not None:
                self._dropped += 1
            self._pending_request = None
            self._latest_result = None
            self._reset_rolling_telemetry_locked()
            self._condition.notify_all()

    def take_latest_result(self) -> StitchResult | None:
        with self._condition:
            return self._latest_result

    def state(self) -> StitchProcessingState:
        with self._condition:
            elapsed_samples = tuple(self._rolling_elapsed_ms)
            rolling_elapsed_p50_ms = (
                float(np.percentile(elapsed_samples, 50.0))
                if elapsed_samples
                else 0.0
            )
            rolling_elapsed_p95_ms = (
                float(np.percentile(elapsed_samples, 95.0))
                if elapsed_samples
                else 0.0
            )
            return StitchProcessingState(
                session_id=self._active_session_id,
                configuration_id=self._configuration_id,
                processor_mode=self._processor_mode,
                runtime_mode=self._runtime_mode,
                runtime_status=self._runtime_status,
                projection_source=self._projection_source,
                configured=self._stitcher is not None,
                shutdown_requested=self._shutdown,
                thread_started=self._started,
                thread_alive=self._thread.is_alive(),
                thread_daemon=self._thread.daemon,
                submitted=self._submitted,
                superseded=self._superseded,
                dropped=self._dropped,
                completed=self._completed,
                failed=self._failed,
                busy=self._busy,
                rolling_elapsed_p50_ms=rolling_elapsed_p50_ms,
                rolling_elapsed_p95_ms=rolling_elapsed_p95_ms,
                rolling_result_fps=self._rolling_result_fps_locked(),
            )

    def shutdown(self, timeout: float = 2.0) -> bool:
        with self._condition:
            if not self._shutdown:
                self._configuration_id += 1
            self._shutdown = True
            self._stitcher = None
            self._processor_mode = ""
            self._runtime_mode = ""
            self._runtime_status = ""
            self._projection_source = ""
            if self._pending_request is not None:
                self._dropped += 1
            self._pending_request = None
            self._latest_result = None
            self._condition.notify_all()
        if self._started:
            self._thread.join(timeout=max(0.0, timeout))
            return not self._thread.is_alive()
        return True

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._shutdown and self._pending_request is None:
                        self._condition.wait()
                    if self._shutdown:
                        return
                    request = self._pending_request
                    self._pending_request = None
                    stitcher = self._stitcher
                    self._busy = request is not None and stitcher is not None

                if request is None or stitcher is None:
                    with self._condition:
                        self._busy = False
                        self._condition.notify_all()
                    continue

                start_time = time.perf_counter()
                warped: dict[str, np.ndarray] = {}
                canvas: np.ndarray | None = None
                error = ""
                runtime_warnings: tuple[str, ...] = ()
                runtime_metrics: dict[str, Any] | None = None
                try:
                    processed = stitcher.process(request.frames)
                    if isinstance(processed, RuntimeStitchResult):
                        warped = processed.warped
                        canvas = processed.canvas
                        runtime_warnings = processed.warnings
                        runtime_metrics = processed.metrics
                    else:
                        warped, canvas = processed
                        if request.processor_mode == "candidate":
                            timing = getattr(stitcher, "last_timings", {})
                            runtime_warnings = (
                                "B-2 candidate view is experimental and read-only; it does not write calibration.yaml.",
                            )
                            runtime_metrics = {
                                "runtime_mode": "b2_candidate_view",
                                "timing": timing if isinstance(timing, dict) else {},
                            }
                except Exception as exc:
                    error = str(exc)
                if not error:
                    runtime_metrics = dict(runtime_metrics or {})
                    runtime_metrics["runtime_mode"] = request.runtime_status
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                with self._condition:
                    self._busy = False
                    if (
                        self._shutdown
                        or request.session_id != self._active_session_id
                        or request.configuration_id != self._configuration_id
                        or stitcher is not self._stitcher
                    ):
                        self._dropped += 1
                        self._condition.notify_all()
                        continue
                    self._result_id += 1
                    self._latest_result = StitchResult(
                        session_id=request.session_id,
                        request_id=request.request_id,
                        result_id=self._result_id,
                        warped=warped,
                        canvas=canvas,
                        elapsed_ms=elapsed_ms,
                        configuration_id=request.configuration_id,
                        processor_mode=request.processor_mode,
                        projection_source=request.projection_source,
                        error=error,
                        runtime_mode=request.runtime_mode,
                        runtime_status=request.runtime_status,
                        runtime_warnings=runtime_warnings,
                        runtime_metrics=runtime_metrics,
                    )
                    if error:
                        self._failed += 1
                    else:
                        self._completed += 1
                        self._rolling_elapsed_ms.append(elapsed_ms)
                        self._rolling_result_times.append(time.monotonic())
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._stitcher = None
                self._processor_mode = ""
                self._runtime_mode = ""
                self._runtime_status = ""
                self._projection_source = ""
                self._pending_request = None
                self._busy = False
                self._condition.notify_all()

    def _reset_rolling_telemetry_locked(self) -> None:
        """Reset generation-scoped samples while preserving lifetime totals."""

        self._rolling_elapsed_ms.clear()
        self._rolling_result_times.clear()

    def _rolling_result_fps_locked(self) -> float:
        """Return successful result throughput over the recent rolling window."""

        cutoff = time.monotonic() - self._RESULT_FPS_WINDOW_SECONDS
        while self._rolling_result_times and self._rolling_result_times[0] < cutoff:
            self._rolling_result_times.popleft()
        if len(self._rolling_result_times) < 2:
            return 0.0
        elapsed = self._rolling_result_times[-1] - self._rolling_result_times[0]
        if elapsed <= 0.0:
            return 0.0
        return (len(self._rolling_result_times) - 1) / elapsed


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
        candidate_use_opencl: bool = False,
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
            candidate_use_opencl=candidate_use_opencl,
            runtime_config=runtime_config,
        )

    def submit_latest(self, session_id: int, frames: dict[str, np.ndarray]) -> int:
        self.start()
        return self._worker.submit_latest(session_id, frames)

    def invalidate_session(self, session_id: int) -> None:
        self._worker.invalidate_session(session_id)

    def take_latest_result(self) -> StitchResult | None:
        return self._worker.take_latest_result()

    def state(self) -> StitchProcessingState:
        return self._worker.state()

    def shutdown(self, timeout: float = 2.0) -> bool:
        return self._worker.shutdown(timeout=timeout)
