from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.benchmark_stage_c import (
    array_sha256,
    black_pixel_ratio,
    BenchmarkSpec,
    ModeRunner,
    build_parser,
    current_rss_bytes,
    duration_statistics,
    generate_synthetic_frames,
    percentile,
    run_benchmark_case,
    snapshot_files,
    validate_output_path,
    verify_snapshot,
)


class StageCBenchmarkTests(unittest.TestCase):
    def test_canvas_fingerprint_is_content_sensitive_and_reports_black_pixels(self) -> None:
        first = np.zeros((2, 3, 3), dtype=np.uint8)
        second = first.copy()
        second[0, 0] = (1, 2, 3)

        self.assertNotEqual(array_sha256(first), array_sha256(second))
        self.assertEqual(1.0, black_pixel_ratio(first))
        self.assertAlmostEqual(5.0 / 6.0, black_pixel_ratio(second))

    def test_default_arguments_are_synthetic_five_mode_read_only_shape(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(2, args.warmup)
        self.assertEqual(10, args.samples)
        self.assertEqual(
            [
                "far_default",
                "b2_view",
                "far_custom",
                "near_current",
                "near_fisheye",
            ],
            args.modes,
        )
        self.assertEqual("-", args.json_output)
        self.assertIsNone(args.markdown_output)

    def test_percentile_and_duration_statistics_are_deterministic(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]

        self.assertEqual(25.0, percentile(values, 50.0))
        self.assertAlmostEqual(38.5, percentile(values, 95.0))
        stats = duration_statistics(values)
        self.assertEqual(25.0, stats["mean_ms"])
        self.assertEqual(25.0, stats["p50_ms"])
        self.assertAlmostEqual(38.5, stats["p95_ms"])
        self.assertEqual(40.0, stats["result_fps"])

    def test_standard_library_rss_probe_returns_a_positive_value(self) -> None:
        rss = current_rss_bytes()

        self.assertIsNotNone(rss)
        self.assertGreater(rss or 0, 0)

    def test_synthetic_frames_are_repeatable_and_include_valid_black_content(self) -> None:
        first = generate_synthetic_frames((64, 48), seed=123)
        second = generate_synthetic_frames((64, 48), seed=123)

        self.assertEqual({"front_left", "front", "front_right"}, set(first))
        for camera in first:
            self.assertEqual((48, 64, 3), first[camera].shape)
            self.assertTrue(np.array_equal(first[camera], second[camera]))
            self.assertGreater(np.count_nonzero(np.all(first[camera] == 0, axis=2)), 0)

    def test_runner_applies_warmup_then_collects_samples_and_rss(self) -> None:
        executions: list[int] = []

        def factory() -> ModeRunner:
            def execute() -> np.ndarray:
                executions.append(len(executions) + 1)
                return np.full((2, 2), executions[-1], dtype=np.uint8)

            return ModeRunner(
                execute=execute,
                describe=lambda value: {"last_value": int(value[0, 0])},
            )

        rss_values = iter([100, 110, 120, 130, 140])
        result = run_benchmark_case(
            BenchmarkSpec("fake", factory),
            warmup=2,
            samples=2,
            rss_reader=lambda: next(rss_values),
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(4, len(executions))
        self.assertEqual(4, result["result_metadata"]["last_value"])
        self.assertEqual(2, len(result["warmup_ms"]))
        self.assertEqual(2, len(result["samples_ms"]))
        self.assertEqual(10, result["rss"]["setup_delta_bytes"])
        self.assertEqual(30, result["rss"]["sample_delta_bytes"])
        self.assertEqual(130, result["rss"]["peak_sample_bytes"])

    def test_output_rejects_config_candidate_roots_and_reserved_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "configs"
            protected.mkdir()

            with self.assertRaises(ValueError):
                validate_output_path(str(protected / "bench.json"), [protected])
            with self.assertRaises(ValueError):
                validate_output_path(str(root / "candidate.yaml"), [protected])
            validate_output_path(str(root / "reports" / "bench.json"), [protected])
            validate_output_path("-", [protected])

    def test_snapshot_guard_is_content_based_and_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.yaml"
            path.write_text("value: 1\n", encoding="utf-8")
            before = snapshot_files([path])

            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                before[str(path.resolve())],
            )
            verify_snapshot(before)
            path.write_text("value: 2\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_snapshot(before)


if __name__ == "__main__":
    unittest.main()
