"""Reproducible, read-only Stage C benchmark for the five runtime views.

The default input is a deterministic synthetic three-camera frame set.  The
tool reuses the production stitcher, projection providers, runtime controller,
and compositors, but it never starts Qt, FFmpeg, QGC, or a camera stream.  It
also snapshots every selected runtime input and verifies that the benchmark did
not change it.

Examples::

    python tools/benchmark_stage_c.py --warmup 2 --samples 10
    python tools/benchmark_stage_c.py --json-output reports/stage_c.json \
        --markdown-output reports/stage_c.md
    python tools/benchmark_stage_c.py --modes far_default near_current
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass, replace
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_shark_studio.calibration_candidate import (  # noqa: E402
    CandidatePanoramaProcessor,
    latest_calibration_candidate,
)
from deep_shark_studio.config import file_revision, load_yaml  # noqa: E402
from deep_shark_studio.projection.intrinsics_runtime_loader import (  # noqa: E402
    FisheyeIntrinsicsRuntimeSource,
    load_fisheye_intrinsics_source,
)
from deep_shark_studio.projection.runtime_providers import (  # noqa: E402
    B2_FAR_FIELD_PROJECTION_SOURCE,
    FisheyeRectilinearParams,
    FisheyeRectilinearProjectionProvider,
)
from deep_shark_studio.seam.far_field_layout_candidate_runtime import (  # noqa: E402
    FarFieldLayoutRuntimeCandidate,
    load_far_field_layout_candidate,
)
from deep_shark_studio.seam.layout_candidate_runtime import (  # noqa: E402
    LayoutRuntimeCandidate,
    LayoutRuntimeProjection,
    load_layout_candidate_for_runtime,
)
from deep_shark_studio.stitch_runtime_controller import (  # noqa: E402
    RuntimeStitchController,
    RuntimeStitchResult,
)
from deep_shark_studio.stitch_runtime_modes import (  # noqa: E402
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)
from deep_shark_studio.stitcher import SurroundStitcher  # noqa: E402
from deep_shark_studio.stitching.camera_layout_adjust import (  # noqa: E402
    CameraAdjustParams,
)


CAMERAS = ("front_left", "front", "front_right")
MODE_ORDER = (
    "far_default",
    "b2_view",
    "far_custom",
    "near_current",
    "near_fisheye",
)


@dataclass(frozen=True)
class BenchmarkInputs:
    calibration_path: Path
    cameras_path: Path
    calibration: dict[str, Any]
    max_input_width: int | None
    frame_size: tuple[int, int]
    b2_candidate_directory: Path | None
    near_candidate: LayoutRuntimeCandidate | None
    fisheye_source: FisheyeIntrinsicsRuntimeSource | None
    far_candidate: FarFieldLayoutRuntimeCandidate | None
    far_candidate_is_synthetic: bool
    near_fisheye_candidate_is_synthetic: bool


@dataclass(frozen=True)
class ModeRunner:
    execute: Callable[[], Any]
    describe: Callable[[Any], dict[str, Any]]
    close: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    factory: Callable[[], ModeRunner] | None
    unavailable_reason: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the five DeepShark runtime views with deterministic "
            "synthetic frames. No GUI, stream, FFmpeg, QGC, config write, or "
            "candidate write is performed."
        )
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--samples", type=positive_int, default=10)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODE_ORDER,
        default=list(MODE_ORDER),
        help="Runtime views to benchmark (default: all five).",
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "calibration.yaml",
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cameras.yaml",
    )
    parser.add_argument("--b2-candidate", type=Path)
    parser.add_argument("--near-candidate", type=Path)
    parser.add_argument("--fisheye-source", type=Path)
    parser.add_argument("--far-candidate", type=Path)
    parser.add_argument(
        "--max-input-width",
        type=positive_int,
        help="Override cameras.yaml performance.max_input_width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260713,
        help="Synthetic pattern seed (default: 20260713).",
    )
    parser.add_argument(
        "--b2-opencl",
        action="store_true",
        help=(
            "Request OpenCL for the standalone B-2 View only. Far Custom "
            "still needs CPU per-camera B-2 projections."
        ),
    )
    parser.add_argument(
        "--json-output",
        default="-",
        help="JSON destination, or '-' for stdout (default: '-').",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional Markdown destination, or '-' for stdout.",
    )
    return parser


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def duration_statistics(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("at least one benchmark sample is required")
    values = [float(value) for value in samples_ms]
    total = sum(values)
    mean = total / len(values)
    return {
        "minimum_ms": min(values),
        "mean_ms": mean,
        "p50_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "maximum_ms": max(values),
        "result_fps": (len(values) * 1000.0 / total) if total > 0.0 else 0.0,
    }


def current_rss_bytes() -> int | None:
    """Return current process RSS using only the Python standard library."""
    if os.name == "nt":
        return _windows_rss_bytes()
    proc_statm = Path("/proc/self/statm")
    if proc_statm.exists():
        try:
            resident_pages = int(proc_statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return maximum if sys.platform == "darwin" else maximum * 1024
    except (ImportError, OSError, ValueError):
        return None


def rss_measurement_source() -> str:
    if os.name == "nt":
        return "GetProcessMemoryInfo.WorkingSetSize"
    if Path("/proc/self/statm").exists():
        return "/proc/self/statm resident pages"
    return "resource.getrusage peak RSS fallback"


def _windows_rss_bytes() -> int | None:
    try:
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory_info = kernel32.K32GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = get_memory_info(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        return int(counters.WorkingSetSize) if ok else None
    except (AttributeError, OSError, ValueError):
        return None


def generate_synthetic_frames(
    frame_size: tuple[int, int],
    seed: int,
) -> dict[str, np.ndarray]:
    """Create deterministic full-resolution frames with texture and black data."""
    width, height = (int(frame_size[0]), int(frame_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("synthetic frame dimensions must be positive")
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    frames: dict[str, np.ndarray] = {}
    for index, camera in enumerate(CAMERAS):
        offset = (int(seed) + index * 71) & 0xFF
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = ((x + offset) & 0xFF).astype(np.uint8)
        frame[:, :, 1] = ((y * 3 + offset * 2) & 0xFF).astype(np.uint8)
        frame[:, :, 2] = ((x // 2 + y // 3 + offset * 5) & 0xFF).astype(np.uint8)
        # A valid black region ensures geometry masks, rather than pixel colour,
        # are exercised by the current perspective chain.
        black_width = max(8, width // 16)
        frame[height // 4 : height // 2, index * black_width : (index + 1) * black_width] = 0
        frames[camera] = frame
    return frames


def snapshot_files(paths: Iterable[Path]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for raw_path in sorted({Path(path).resolve() for path in paths}, key=str):
        if not raw_path.exists() or not raw_path.is_file():
            continue
        digest = hashlib.sha256()
        with raw_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        snapshot[str(raw_path)] = digest.hexdigest()
    return snapshot


def verify_snapshot(snapshot: Mapping[str, str]) -> None:
    current = snapshot_files(Path(path) for path in snapshot)
    if dict(snapshot) != current:
        before_paths = set(snapshot)
        after_paths = set(current)
        changed = sorted(
            path
            for path in before_paths & after_paths
            if snapshot[path] != current[path]
        )
        missing = sorted(before_paths - after_paths)
        details = changed + [f"missing: {path}" for path in missing]
        raise RuntimeError(
            "Read-only benchmark guard detected modified runtime inputs: "
            + ", ".join(details)
        )


def validate_output_path(path: str | None, protected_roots: Iterable[Path]) -> None:
    if path in (None, "-"):
        return
    output = Path(path).resolve()
    if output.name.lower() in {"candidate.yaml", "report.yaml"}:
        raise ValueError("Benchmark output may not be named candidate.yaml or report.yaml.")
    for root in protected_roots:
        resolved_root = Path(root).resolve()
        if output == resolved_root or resolved_root in output.parents:
            raise ValueError(
                f"Benchmark output is inside a protected config/candidate root: {output}"
            )


def discover_near_candidate(path: Path | None) -> LayoutRuntimeCandidate | None:
    if path is not None:
        return load_layout_candidate_for_runtime(path)
    root = PROJECT_ROOT / "projects" / "layout_candidates"
    if not root.exists():
        return None
    candidates = sorted(
        root.glob("**/candidate.yaml"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate_path in candidates:
        try:
            candidate = load_layout_candidate_for_runtime(candidate_path)
        except (OSError, TypeError, ValueError):
            continue
        if candidate.projection.source == ProjectionSource.CURRENT_PERSPECTIVE:
            return candidate
    return None


def load_benchmark_inputs(args: argparse.Namespace) -> BenchmarkInputs:
    calibration_path = Path(args.calibration_config).resolve()
    cameras_path = Path(args.cameras_config).resolve()
    calibration = load_yaml(calibration_path)
    cameras_config = load_yaml(cameras_path)
    performance = cameras_config.get("performance", {})
    configured_width = performance.get("max_input_width")
    max_input_width = (
        int(args.max_input_width)
        if args.max_input_width is not None
        else (int(configured_width) if configured_width else None)
    )

    b2_directory = (
        _candidate_directory(Path(args.b2_candidate).resolve())
        if args.b2_candidate is not None
        else latest_calibration_candidate()
    )
    near_candidate = discover_near_candidate(args.near_candidate)
    fisheye_path = (
        Path(args.fisheye_source).resolve()
        if args.fisheye_source is not None
        else b2_directory
    )
    fisheye_source = (
        load_fisheye_intrinsics_source(fisheye_path)
        if fisheye_path is not None
        else None
    )

    far_candidate: FarFieldLayoutRuntimeCandidate | None = None
    far_candidate_is_synthetic = False
    if args.far_candidate is not None:
        far_candidate = load_far_field_layout_candidate(args.far_candidate)
        if b2_directory is None and far_candidate.b2_candidate_directory is not None:
            b2_directory = far_candidate.b2_candidate_directory
    elif b2_directory is not None:
        b2_data = load_yaml(b2_directory / "candidate.yaml")
        panorama = b2_data.get("virtual_panorama", {})
        canvas_size = panorama.get("canvas_size")
        if not isinstance(canvas_size, (list, tuple)) or len(canvas_size) != 2:
            raise ValueError("B-2 candidate virtual_panorama.canvas_size is missing.")
        far_candidate = _synthetic_far_candidate(
            b2_directory,
            (int(canvas_size[0]), int(canvas_size[1])),
        )
        far_candidate_is_synthetic = True

    frame_size = _resolve_frame_size(calibration, b2_directory, fisheye_source)
    return BenchmarkInputs(
        calibration_path=calibration_path,
        cameras_path=cameras_path,
        calibration=calibration,
        max_input_width=max_input_width,
        frame_size=frame_size,
        b2_candidate_directory=b2_directory,
        near_candidate=near_candidate,
        fisheye_source=fisheye_source,
        far_candidate=far_candidate,
        far_candidate_is_synthetic=far_candidate_is_synthetic,
        near_fisheye_candidate_is_synthetic=(
            near_candidate is not None and fisheye_source is not None
        ),
    )


def _candidate_directory(path: Path) -> Path:
    return path.parent if path.name == "candidate.yaml" else path


def _resolve_frame_size(
    calibration: Mapping[str, Any],
    b2_directory: Path | None,
    fisheye_source: FisheyeIntrinsicsRuntimeSource | None,
) -> tuple[int, int]:
    required_sizes: list[tuple[str, tuple[int, int]]] = []
    if b2_directory is not None:
        candidate = load_yaml(b2_directory / "candidate.yaml")
        resolution = candidate.get("resolution")
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            raise ValueError("B-2 candidate resolution is missing.")
        required_sizes.append(("B-2", (int(resolution[0]), int(resolution[1]))))
    if fisheye_source is not None:
        fisheye_sizes = {model.image_size for model in fisheye_source.cameras.values()}
        if len(fisheye_sizes) != 1:
            raise ValueError(f"Fisheye camera resolutions do not match: {fisheye_sizes}")
        required_sizes.append(("Near Fisheye", next(iter(fisheye_sizes))))
    profile = _active_profile(calibration)
    reference = profile.get("source_reference_size")
    if isinstance(reference, (list, tuple)) and len(reference) == 2:
        required_sizes.append(
            ("current perspective", (int(reference[0]), int(reference[1])))
        )
    if not required_sizes:
        return (1920, 1080)
    unique = {size for _label, size in required_sizes}
    if len(unique) != 1:
        detail = ", ".join(f"{label}={size[0]}x{size[1]}" for label, size in required_sizes)
        raise ValueError(f"Selected runtime inputs require incompatible frame sizes: {detail}")
    return required_sizes[0][1]


def _active_profile(calibration: Mapping[str, Any]) -> dict[str, Any]:
    topology = str(calibration.get("stitch_topology", ""))
    profiles = calibration.get("topologies")
    if topology and isinstance(profiles, Mapping):
        profile = profiles.get(topology)
        if isinstance(profile, dict):
            return profile
    return dict(calibration)


def _synthetic_far_candidate(
    b2_directory: Path,
    output_size: tuple[int, int],
) -> FarFieldLayoutRuntimeCandidate:
    identity = {
        camera: CameraAdjustParams(x_offset_px=0, y_offset_px=0, scale=1.0)
        for camera in CAMERAS
    }
    width, height = output_size
    return FarFieldLayoutRuntimeCandidate(
        path=Path("<in-memory-stage-c-far-custom>"),
        schema_version=1,
        profile_id="triple_front_panorama",
        camera_adjust=identity,
        output_width_px=int(width),
        output_height_px=int(height),
        feather_width_override_px=None,
        calibration_hash=None,
        current_calibration_hash=None,
        warnings=("Synthetic identity layout used by benchmark only.",),
        raw={},
        projection_source=B2_FAR_FIELD_PROJECTION_SOURCE,
        b2_candidate_directory=b2_directory,
        blend_mode="b2_weight_selection",
    )


def _fisheye_near_candidate(
    candidate: LayoutRuntimeCandidate,
    source: FisheyeIntrinsicsRuntimeSource,
) -> LayoutRuntimeCandidate:
    projection = LayoutRuntimeProjection(
        source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
        intrinsics_source_path=source.path,
        balance=0.6,
        fov_scale=1.0,
        projection_pipeline="fisheye_rectilinear_then_template_perspective_warp",
        raw={"benchmark_in_memory_adapter": True},
    )
    return replace(
        candidate,
        schema_version=3,
        projection=projection,
        warnings=tuple(candidate.warnings)
        + ("Synthetic projection binding used by benchmark only.",),
    )


def build_specs(
    inputs: BenchmarkInputs,
    frames: Mapping[str, np.ndarray],
    modes: Sequence[str],
    *,
    b2_opencl: bool,
) -> list[BenchmarkSpec]:
    factories: dict[str, Callable[[], ModeRunner] | None] = {
        "far_default": lambda: _far_default_runner(inputs, frames),
        "b2_view": (
            (lambda: _b2_runner(inputs.b2_candidate_directory, frames, b2_opencl))
            if inputs.b2_candidate_directory is not None
            else None
        ),
        "far_custom": (
            (lambda: _far_custom_runner(inputs, frames))
            if inputs.far_candidate is not None
            and inputs.far_candidate.b2_candidate_directory is not None
            else None
        ),
        "near_current": (
            (lambda: _near_current_runner(inputs, frames))
            if inputs.near_candidate is not None
            else None
        ),
        "near_fisheye": (
            (lambda: _near_fisheye_runner(inputs, frames))
            if inputs.near_candidate is not None and inputs.fisheye_source is not None
            else None
        ),
    }
    reasons = {
        "b2_view": "No complete B-2 calibration candidate was found or provided.",
        "far_custom": "Far Custom requires a B-2 candidate and a Far layout candidate.",
        "near_current": "No runtime-safe Near layout candidate was found or provided.",
        "near_fisheye": "Near Fisheye requires Near layout and fisheye intrinsics sources.",
    }
    return [
        BenchmarkSpec(
            name=mode,
            factory=factories[mode],
            unavailable_reason=(reasons.get(mode, "") if factories[mode] is None else ""),
        )
        for mode in modes
    ]


def _new_stitcher(inputs: BenchmarkInputs) -> SurroundStitcher:
    return SurroundStitcher(
        inputs.calibration,
        max_input_width=inputs.max_input_width,
        use_intrinsics=False,
    )


def _far_default_runner(
    inputs: BenchmarkInputs,
    frames: Mapping[str, np.ndarray],
) -> ModeRunner:
    controller = RuntimeStitchController(
        _new_stitcher(inputs),
        RuntimeStitchConfig(mode=StitchRuntimeMode.FAR_FIELD),
    )
    return ModeRunner(
        execute=lambda: controller.process(frames),
        describe=lambda result: _runtime_result_metadata(
            result,
            backend={"engine": "OpenCV Mat", "backend": "cpu"},
        ),
    )


def _b2_runner(
    candidate_directory: Path | None,
    frames: Mapping[str, np.ndarray],
    use_opencl: bool,
) -> ModeRunner:
    if candidate_directory is None:
        raise ValueError("B-2 candidate is required")
    previous_opencl = bool(cv2.ocl.useOpenCL())
    processor = CandidatePanoramaProcessor(candidate_directory, use_opencl=use_opencl)

    def describe(result: tuple[dict[str, np.ndarray], np.ndarray]) -> dict[str, Any]:
        warped, canvas = result
        return {
            "effective_status": "b2_candidate_view",
            "canvas_shape": list(canvas.shape),
            "canvas_sha256": array_sha256(canvas),
            "canvas_black_pixel_ratio": black_pixel_ratio(canvas),
            "warped_cameras": sorted(warped),
            "backend": processor.backend_status(),
            "runtime_timing": dict(processor.last_timings),
        }

    def close() -> None:
        try:
            cv2.ocl.setUseOpenCL(previous_opencl)
        except cv2.error:
            pass

    return ModeRunner(
        execute=lambda: processor.process(dict(frames)),
        describe=describe,
        close=close,
    )


def _far_custom_runner(
    inputs: BenchmarkInputs,
    frames: Mapping[str, np.ndarray],
) -> ModeRunner:
    candidate = inputs.far_candidate
    if candidate is None:
        raise ValueError("Far layout candidate is required")
    controller = RuntimeStitchController(
        _new_stitcher(inputs),
        RuntimeStitchConfig(
            mode=StitchRuntimeMode.FAR_FIELD,
            use_far_field_custom_layout=True,
            far_field_layout_candidate_path=candidate.path,
        ),
        far_field_layout_candidate=candidate,
    )
    # Make candidate map/weight construction part of setup rather than the
    # first timed warm-up. The same controller path remains responsible for it.
    provider = controller._far_field_custom_projection_provider()

    def describe(result: RuntimeStitchResult) -> dict[str, Any]:
        processor = getattr(provider, "processor", None)
        backend = (
            processor.backend_status()
            if processor is not None and hasattr(processor, "backend_status")
            else {"engine": "OpenCV Mat", "backend": "cpu"}
        )
        metadata = _runtime_result_metadata(result, backend=backend)
        metadata["layout_source"] = (
            "synthetic_identity_in_memory"
            if inputs.far_candidate_is_synthetic
            else "persisted_candidate_read_only"
        )
        return metadata

    return ModeRunner(
        execute=lambda: controller.process(frames),
        describe=describe,
    )


def _near_current_runner(
    inputs: BenchmarkInputs,
    frames: Mapping[str, np.ndarray],
) -> ModeRunner:
    if inputs.near_candidate is None:
        raise ValueError("Near candidate is required")
    controller = RuntimeStitchController(
        _new_stitcher(inputs),
        RuntimeStitchConfig(
            mode=StitchRuntimeMode.NEAR_FIELD,
            projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
            layout_candidate_path=inputs.near_candidate.path,
        ),
        layout_candidate=inputs.near_candidate,
    )
    return ModeRunner(
        execute=lambda: controller.process(frames),
        describe=lambda result: _runtime_result_metadata(
            result,
            backend={"engine": "OpenCV Mat", "backend": "cpu"},
        ),
    )


def _near_fisheye_runner(
    inputs: BenchmarkInputs,
    frames: Mapping[str, np.ndarray],
) -> ModeRunner:
    if inputs.near_candidate is None or inputs.fisheye_source is None:
        raise ValueError("Near candidate and fisheye source are required")
    candidate = _fisheye_near_candidate(inputs.near_candidate, inputs.fisheye_source)
    stitcher = _new_stitcher(inputs)
    provider = FisheyeRectilinearProjectionProvider(
        stitcher,
        inputs.fisheye_source,
        FisheyeRectilinearParams(balance=0.6, fov_scale=1.0),
    )
    controller = RuntimeStitchController(
        stitcher,
        RuntimeStitchConfig(
            mode=StitchRuntimeMode.NEAR_FIELD,
            projection_source=ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE,
            layout_candidate_path=candidate.path,
            projection_intrinsics_source_path=inputs.fisheye_source.path,
            fisheye_balance=0.6,
            fisheye_fov_scale=1.0,
        ),
        layout_candidate=candidate,
        projection_provider=provider,
    )

    def describe(result: RuntimeStitchResult) -> dict[str, Any]:
        metadata = _runtime_result_metadata(
            result,
            backend={"engine": "OpenCV Mat", "backend": "cpu"},
        )
        metadata["projection_binding"] = "synthetic_in_memory"
        metadata["fisheye_map_build_count"] = int(provider.map_build_count)
        return metadata

    return ModeRunner(execute=lambda: controller.process(frames), describe=describe)


def _runtime_result_metadata(
    result: RuntimeStitchResult,
    *,
    backend: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    projection = metrics.get("projection", {})
    canvas = result.canvas
    return {
        "effective_status": result.status,
        "canvas_shape": list(canvas.shape) if canvas is not None else None,
        "canvas_sha256": array_sha256(canvas) if canvas is not None else None,
        "canvas_black_pixel_ratio": (
            black_pixel_ratio(canvas) if canvas is not None else None
        ),
        "warped_cameras": sorted(result.warped),
        "backend": dict(backend),
        "runtime_timing": dict(metrics.get("timing", {})),
        "projection_metadata": dict(projection) if isinstance(projection, Mapping) else {},
        "composition_mode": metrics.get("composition_mode"),
        "warnings": list(result.warnings),
    }


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def black_pixel_ratio(image: np.ndarray) -> float:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("canvas must have shape HxWx3")
    return float(np.count_nonzero(np.all(value == 0, axis=2)) / max(1, value.shape[0] * value.shape[1]))


def run_benchmark_case(
    spec: BenchmarkSpec,
    *,
    warmup: int,
    samples: int,
    rss_reader: Callable[[], int | None] = current_rss_bytes,
) -> dict[str, Any]:
    if spec.factory is None:
        return {
            "mode": spec.name,
            "status": "unavailable",
            "reason": spec.unavailable_reason,
        }
    rss_before_setup = rss_reader()
    setup_start = time.perf_counter()
    runner: ModeRunner | None = None
    try:
        runner = spec.factory()
        setup_ms = (time.perf_counter() - setup_start) * 1000.0
        rss_after_setup = rss_reader()
        warmup_ms: list[float] = []
        last_result: Any = None
        for _index in range(warmup):
            start = time.perf_counter()
            last_result = runner.execute()
            warmup_ms.append((time.perf_counter() - start) * 1000.0)
        sample_ms: list[float] = []
        rss_samples: list[int] = []
        for _index in range(samples):
            start = time.perf_counter()
            last_result = runner.execute()
            sample_ms.append((time.perf_counter() - start) * 1000.0)
            rss = rss_reader()
            if rss is not None:
                rss_samples.append(int(rss))
        if last_result is None:
            raise RuntimeError("benchmark produced no result")
        rss_after_samples = rss_reader()
        result = {
            "mode": spec.name,
            "status": "success",
            "setup_ms": setup_ms,
            "warmup_ms": warmup_ms,
            "samples_ms": sample_ms,
            "statistics": duration_statistics(sample_ms),
            "rss": {
                "before_setup_bytes": rss_before_setup,
                "after_setup_bytes": rss_after_setup,
                "after_samples_bytes": rss_after_samples,
                "peak_sample_bytes": max(rss_samples) if rss_samples else None,
                "setup_delta_bytes": _optional_delta(rss_after_setup, rss_before_setup),
                "sample_delta_bytes": _optional_delta(rss_after_samples, rss_after_setup),
            },
            "result_metadata": runner.describe(last_result),
        }
        return _json_safe(result)
    except Exception as exc:
        return {
            "mode": spec.name,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if runner is not None:
            runner.close()
        gc.collect()


def _optional_delta(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return int(after) - int(before)


def runtime_source_files(inputs: BenchmarkInputs) -> list[Path]:
    files = [inputs.calibration_path, inputs.cameras_path]
    if inputs.b2_candidate_directory is not None:
        files.extend(
            path
            for path in inputs.b2_candidate_directory.rglob("*")
            if path.is_file()
        )
    if inputs.near_candidate is not None and inputs.near_candidate.path.exists():
        files.append(inputs.near_candidate.path)
    if inputs.fisheye_source is not None and inputs.fisheye_source.path.exists():
        files.append(inputs.fisheye_source.path)
    if (
        inputs.far_candidate is not None
        and not inputs.far_candidate_is_synthetic
        and inputs.far_candidate.path.exists()
    ):
        files.append(inputs.far_candidate.path)
    return sorted({path.resolve() for path in files}, key=str)


def protected_output_roots(inputs: BenchmarkInputs) -> list[Path]:
    roots = [PROJECT_ROOT / "configs", PROJECT_ROOT / "projects"]
    for candidate_path in (
        inputs.b2_candidate_directory,
        inputs.near_candidate.path.parent if inputs.near_candidate is not None else None,
        inputs.fisheye_source.path.parent if inputs.fisheye_source is not None else None,
        (
            inputs.far_candidate.path.parent
            if inputs.far_candidate is not None and not inputs.far_candidate_is_synthetic
            else None
        ),
    ):
        if candidate_path is not None:
            roots.append(Path(candidate_path))
    return roots


def build_report(
    args: argparse.Namespace,
    inputs: BenchmarkInputs,
    results: Sequence[dict[str, Any]],
    protected_snapshot: Mapping[str, str],
) -> dict[str, Any]:
    return _json_safe(
        {
            "schema_version": 1,
            "tool": "DeepSharkStageCSyntheticBenchmark",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "input": {
                "kind": "deterministic_synthetic_frames",
                "frame_size": list(inputs.frame_size),
                "cameras": list(CAMERAS),
                "seed": int(args.seed),
                "calibration_path": str(inputs.calibration_path),
                "calibration_revision": file_revision(inputs.calibration_path),
                "cameras_path": str(inputs.cameras_path),
                "cameras_revision": file_revision(inputs.cameras_path),
                "b2_candidate_directory": (
                    str(inputs.b2_candidate_directory)
                    if inputs.b2_candidate_directory is not None
                    else None
                ),
                "near_candidate_path": (
                    str(inputs.near_candidate.path)
                    if inputs.near_candidate is not None
                    else None
                ),
                "fisheye_source_path": (
                    str(inputs.fisheye_source.path)
                    if inputs.fisheye_source is not None
                    else None
                ),
                "far_layout_source": (
                    "synthetic_identity_in_memory"
                    if inputs.far_candidate_is_synthetic
                    else (
                        str(inputs.far_candidate.path)
                        if inputs.far_candidate is not None
                        else None
                    )
                ),
                "near_fisheye_projection_binding": (
                    "synthetic_in_memory"
                    if inputs.near_fisheye_candidate_is_synthetic
                    else None
                ),
            },
            "parameters": {
                "modes": list(args.modes),
                "warmup": int(args.warmup),
                "samples": int(args.samples),
                "max_input_width": inputs.max_input_width,
                "b2_opencl_requested": bool(args.b2_opencl),
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "opencv_threads": int(cv2.getNumThreads()),
                "opencv_optimized": bool(cv2.useOptimized()),
                "opencl_available": bool(cv2.ocl.haveOpenCL()),
                "opencl_active_before_benchmark": bool(cv2.ocl.useOpenCL()),
                "logical_cpu_count": os.cpu_count(),
                "rss_measurement_source": rss_measurement_source(),
            },
            "read_only_guard": {
                "verified": True,
                "method": "SHA-256 before/after selected runtime inputs",
                "protected_file_count": len(protected_snapshot),
            },
            "results": list(results),
        }
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# DeepShark Stage C Synthetic Benchmark",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        "",
        (
            "Input: deterministic synthetic `"
            + "x".join(str(value) for value in report["input"]["frame_size"])
            + "` frames; no GUI, stream, FFmpeg, or QGC process was started."
        ),
        "",
        "| Mode | Status | Setup ms | p50 ms | p95 ms | Result FPS | Peak RSS MiB | Backend |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("results", []):
        status = str(item.get("status", ""))
        if status != "success":
            reason = item.get("reason") or item.get("error") or ""
            lines.append(
                f"| {item.get('mode', '')} | {status}: {reason} | — | — | — | — | — | — |"
            )
            continue
        stats = item["statistics"]
        rss = item.get("rss", {})
        peak_rss = rss.get("peak_sample_bytes")
        backend = item.get("result_metadata", {}).get("backend", {})
        backend_name = backend.get("backend") or backend.get("engine") or "unknown"
        lines.append(
            "| {mode} | success | {setup:.2f} | {p50:.2f} | {p95:.2f} | "
            "{fps:.2f} | {rss} | {backend} |".format(
                mode=item.get("mode", ""),
                setup=float(item.get("setup_ms", 0.0)),
                p50=float(stats["p50_ms"]),
                p95=float(stats["p95_ms"]),
                fps=float(stats["result_fps"]),
                rss=(f"{peak_rss / (1024 * 1024):.1f}" if peak_rss is not None else "—"),
                backend=backend_name,
            )
        )
    lines.extend(
        [
            "",
            "## Measurement contract",
            "",
            f"- Warm-up iterations: {report['parameters']['warmup']}",
            f"- Timed samples: {report['parameters']['samples']}",
            "- Result FPS is timed samples divided by their total end-to-end runtime.",
            "- RSS is process working-set/RSS, not GPU memory.",
            (
                "- For mode-isolated RSS, invoke one mode per process; when several "
                "modes run together, later rows can include OpenCV allocator retention "
                "from earlier rows."
            ),
            "- The SHA-256 read-only guard passed for all selected runtime inputs.",
            "- Far Custom uses the production B-2 per-camera projection and B-2 weight-selection compositor.",
            "- When no persisted Far layout exists, an identity layout object is created only in memory.",
            "- Near Fisheye reuses the persisted Near layout with an in-memory fisheye projection binding.",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_output(content: str, destination: str | None) -> None:
    if destination is None:
        return
    if destination == "-":
        print(content, end="" if content.endswith("\n") else "\n")
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_benchmark_inputs(args)
    protected_roots = protected_output_roots(inputs)
    validate_output_path(args.json_output, protected_roots)
    validate_output_path(args.markdown_output, protected_roots)
    source_files = runtime_source_files(inputs)
    before = snapshot_files(source_files)
    frames = generate_synthetic_frames(inputs.frame_size, args.seed)
    specs = build_specs(
        inputs,
        frames,
        args.modes,
        b2_opencl=bool(args.b2_opencl),
    )
    results = [
        run_benchmark_case(spec, warmup=args.warmup, samples=args.samples)
        for spec in specs
    ]
    verify_snapshot(before)
    report = build_report(args, inputs, results, before)
    json_content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_markdown(report)
    emit_output(json_content, args.json_output)
    if args.markdown_output == "-" and args.json_output == "-":
        print("---")
    emit_output(markdown_content, args.markdown_output)
    return 0 if all(item.get("status") == "success" for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
