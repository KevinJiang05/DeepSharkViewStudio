from __future__ import annotations

from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from deep_shark_studio import stitch_processing
from deep_shark_studio.stitch_processing import StitchProcessingManager
from deep_shark_studio.stitch_runtime_controller import RuntimeStitchResult
from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)


def wait_for_result(
    manager: StitchProcessingManager,
    timeout: float = 1.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = manager.take_latest_result()
        if result is not None:
            return result
        time.sleep(0.005)
    return None


class ImmediateProcessor:
    def __init__(self, value: int = 1):
        self.value = value

    def process(self, _frames):
        return RuntimeStitchResult(
            warped={},
            canvas=np.full((1, 1, 3), self.value, dtype=np.uint8),
            mode=StitchRuntimeMode.FAR_FIELD,
            status="far_field",
            metrics={
                "runtime_mode": "far_field",
                "timing": {"synthetic_ms": 1.0},
            },
        )


class RaisingProcessor:
    def process(self, _frames):
        raise RuntimeError("synthetic processing failure")


class BlockingProcessor:
    def __init__(self, value: int = 1):
        self.value = value
        self.started = Event()
        self.release = Event()

    def process(self, _frames):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test did not release blocked processor")
        return RuntimeStitchResult(
            warped={},
            canvas=np.full((1, 1, 3), self.value, dtype=np.uint8),
            mode=StitchRuntimeMode.FAR_FIELD,
            status="far_field",
        )


class StitchProcessingStateTests(unittest.TestCase):
    def test_configure_exposes_normalized_effective_runtime_state(self) -> None:
        manager = StitchProcessingManager()
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    side_effect=lambda _stitcher, _config: ImmediateProcessor(),
                ),
                patch.object(
                    stitch_processing,
                    "CandidatePanoramaProcessor",
                    return_value=ImmediateProcessor(),
                ),
            ):
                cases = (
                    (
                        "template",
                        RuntimeStitchConfig(),
                        None,
                        "far_field",
                        "far_field_default",
                        ProjectionSource.CURRENT_PERSPECTIVE.value,
                    ),
                    (
                        "candidate",
                        RuntimeStitchConfig(),
                        "synthetic-candidate",
                        "far_field",
                        "b2_candidate_view",
                        "b2_candidate_projection",
                    ),
                    (
                        "template",
                        RuntimeStitchConfig(
                            use_far_field_custom_layout=True,
                            far_field_layout_candidate_path="synthetic-far-layout.json",
                        ),
                        None,
                        "far_field",
                        "far_field_custom",
                        ProjectionSource.CURRENT_PERSPECTIVE.value,
                    ),
                    (
                        "template",
                        RuntimeStitchConfig(
                            mode=StitchRuntimeMode.NEAR_FIELD,
                            projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
                            layout_candidate_path="synthetic-near-layout.json",
                        ),
                        None,
                        "near_field",
                        "near_field_current_perspective",
                        ProjectionSource.CURRENT_PERSPECTIVE.value,
                    ),
                    (
                        "template",
                        RuntimeStitchConfig(
                            mode=StitchRuntimeMode.NEAR_FIELD,
                            projection_source=(
                                ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE
                            ),
                            layout_candidate_path="synthetic-near-layout.json",
                            projection_intrinsics_source_path="synthetic-intrinsics.json",
                        ),
                        None,
                        "near_field",
                        "near_field_fisheye_rectilinear",
                        ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
                    ),
                    (
                        "template",
                        RuntimeStitchConfig(mode=StitchRuntimeMode.AUTO),
                        None,
                        "far_field",
                        "auto_fallback_far_field",
                        ProjectionSource.CURRENT_PERSPECTIVE.value,
                    ),
                )
                previous_generation = 0
                for session_id, case in enumerate(cases, start=1):
                    (
                        processor_mode,
                        runtime_config,
                        candidate_directory,
                        runtime_mode,
                        runtime_status,
                        projection_source,
                    ) = case
                    manager.configure(
                        session_id,
                        {},
                        None,
                        False,
                        processor_mode=processor_mode,
                        candidate_directory=candidate_directory,
                        runtime_config=runtime_config,
                    )
                    state = manager.state()
                    self.assertTrue(state.configured)
                    self.assertEqual(session_id, state.session_id)
                    self.assertGreater(state.configuration_id, previous_generation)
                    self.assertEqual(processor_mode, state.processor_mode)
                    self.assertEqual(runtime_mode, state.runtime_mode)
                    self.assertEqual(runtime_status, state.runtime_status)
                    self.assertEqual(projection_source, state.projection_source)
                    self.assertTrue(state.thread_started)
                    self.assertTrue(state.thread_alive)
                    self.assertTrue(state.thread_daemon)
                    previous_generation = state.configuration_id
        finally:
            self.assertTrue(manager.shutdown())

    def test_far_custom_state_exposes_candidate_projection_source(self) -> None:
        manager = StitchProcessingManager()
        runtime_config = RuntimeStitchConfig(
            use_far_field_custom_layout=True,
            far_field_layout_candidate_path="synthetic-far-layout.json",
        )
        processor = ImmediateProcessor()
        processor.config = runtime_config
        processor.far_field_layout_candidate = SimpleNamespace(
            projection_source="b2_far_field_candidate",
        )
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    return_value=processor,
                ),
            ):
                manager.configure(
                    7,
                    {},
                    None,
                    False,
                    runtime_config=runtime_config,
                )

            state = manager.state()
            self.assertEqual("far_field_custom", state.runtime_status)
            self.assertEqual(
                "b2_far_field_candidate",
                state.projection_source,
            )
        finally:
            self.assertTrue(manager.shutdown())

    def test_failed_configure_keeps_previous_configuration_atomic(self) -> None:
        manager = StitchProcessingManager()
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    return_value=ImmediateProcessor(),
                ),
            ):
                manager.configure(10, {}, None, False)
            before = manager.state()

            with patch.object(
                stitch_processing,
                "CandidatePanoramaProcessor",
                side_effect=ValueError("bad candidate"),
            ):
                with self.assertRaisesRegex(ValueError, "bad candidate"):
                    manager.configure(
                        11,
                        {},
                        None,
                        False,
                        processor_mode="candidate",
                        candidate_directory="bad-candidate",
                    )

            after = manager.state()
            self.assertEqual(before, after)
        finally:
            self.assertTrue(manager.shutdown())

    def test_invalidate_detaches_processor_and_rejects_submission(self) -> None:
        manager = StitchProcessingManager()
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    return_value=ImmediateProcessor(),
                ),
            ):
                manager.configure(20, {}, None, False)
            configured = manager.state()

            manager.invalidate_session(21)

            invalidated = manager.state()
            self.assertEqual(21, invalidated.session_id)
            self.assertGreater(
                invalidated.configuration_id,
                configured.configuration_id,
            )
            self.assertFalse(invalidated.configured)
            self.assertEqual("", invalidated.processor_mode)
            request_id = manager.submit_latest(
                21,
                {"front": np.zeros((1, 1, 3), dtype=np.uint8)},
            )
            self.assertEqual(0, request_id)
            self.assertIsNone(manager.take_latest_result())
        finally:
            self.assertTrue(manager.shutdown())


class StitchProcessingGenerationTests(unittest.TestCase):
    def test_same_session_reconfigure_discards_inflight_old_result(self) -> None:
        old_processor = BlockingProcessor(value=1)
        new_processor = ImmediateProcessor(value=2)
        manager = StitchProcessingManager()
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    side_effect=(old_processor, new_processor),
                ),
            ):
                manager.configure(30, {}, None, False)
                old_generation = manager.state().configuration_id
                manager.submit_latest(
                    30,
                    {"front": np.zeros((1, 1, 3), dtype=np.uint8)},
                )
                self.assertTrue(old_processor.started.wait(timeout=1.0))

                manager.configure(30, {}, None, False)
                new_generation = manager.state().configuration_id
                self.assertGreater(new_generation, old_generation)
                old_processor.release.set()
                time.sleep(0.05)
                self.assertIsNone(manager.take_latest_result())

                manager.submit_latest(
                    30,
                    {"front": np.ones((1, 1, 3), dtype=np.uint8)},
                )
                result = wait_for_result(manager)
                self.assertIsNotNone(result)
                self.assertEqual(new_generation, result.configuration_id)
                self.assertEqual(2, int(result.canvas[0, 0, 0]))
                self.assertEqual("template", result.processor_mode)
                self.assertEqual("far_field_default", result.runtime_status)
                self.assertEqual(
                    "far_field_default",
                    result.runtime_metrics["runtime_mode"],
                )
                self.assertEqual(
                    {"synthetic_ms": 1.0},
                    result.runtime_metrics["timing"],
                )
        finally:
            old_processor.release.set()
            self.assertTrue(manager.shutdown())

    def test_processing_error_keeps_effective_configuration_metadata(self) -> None:
        manager = StitchProcessingManager()
        runtime_config = RuntimeStitchConfig(
            mode=StitchRuntimeMode.NEAR_FIELD,
            projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
            layout_candidate_path="synthetic-near-layout.json",
            projection_intrinsics_source_path="synthetic-intrinsics.json",
        )
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    return_value=RaisingProcessor(),
                ),
            ):
                manager.configure(
                    40,
                    {},
                    None,
                    False,
                    runtime_config=runtime_config,
                )
                configured = manager.state()
                manager.submit_latest(
                    40,
                    {"front": np.zeros((1, 1, 3), dtype=np.uint8)},
                )
                result = wait_for_result(manager)

            self.assertIsNotNone(result)
            self.assertIn("synthetic processing failure", result.error)
            self.assertEqual(configured.configuration_id, result.configuration_id)
            self.assertEqual("template", result.processor_mode)
            self.assertEqual("near_field", result.runtime_mode)
            self.assertEqual(
                "near_field_fisheye_rectilinear",
                result.runtime_status,
            )
            self.assertEqual(
                ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
                result.projection_source,
            )
        finally:
            self.assertTrue(manager.shutdown())


class StitchProcessingShutdownTests(unittest.TestCase):
    def test_shutdown_timeout_is_observable_and_daemon_thread_can_later_exit(self) -> None:
        processor = BlockingProcessor()
        manager = StitchProcessingManager()
        try:
            with (
                patch.object(stitch_processing, "SurroundStitcher", return_value=object()),
                patch.object(
                    stitch_processing,
                    "RuntimeStitchController",
                    return_value=processor,
                ),
            ):
                manager.configure(50, {}, None, False)
                manager.submit_latest(
                    50,
                    {"front": np.zeros((1, 1, 3), dtype=np.uint8)},
                )
                self.assertTrue(processor.started.wait(timeout=1.0))

                started_at = time.monotonic()
                self.assertFalse(manager.shutdown(timeout=0.01))
                elapsed = time.monotonic() - started_at
                self.assertLess(elapsed, 0.5)

                timed_out = manager.state()
                self.assertTrue(timed_out.shutdown_requested)
                self.assertTrue(timed_out.thread_alive)
                self.assertTrue(timed_out.thread_daemon)
                self.assertFalse(timed_out.configured)

                processor.release.set()
                self.assertTrue(manager.shutdown(timeout=1.0))
                stopped = manager.state()
                self.assertTrue(stopped.shutdown_requested)
                self.assertFalse(stopped.thread_alive)
                self.assertFalse(stopped.configured)
        finally:
            processor.release.set()
            manager.shutdown(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
