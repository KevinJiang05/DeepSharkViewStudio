"""Relocatable startup preflight for DeepShark View Studio.

This module intentionally imports only Python's standard library at module load
time.  Runtime dependencies and local configuration are checked before the GUI
module (and therefore Qt/OpenCV) is imported.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"

_RUNTIME_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("yaml", "PyYAML"),
    ("PySide6", "PySide6"),
)

_LOCAL_CONFIG_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("cameras.yaml", "cameras.example.yaml"),
    ("network.yaml", "network.example.yaml"),
)


class DependencyPreflightError(RuntimeError):
    """Raised when a required runtime dependency is unavailable."""


class ConfigPreflightError(RuntimeError):
    """Raised when the local runtime configuration is missing or invalid."""


def check_runtime_dependencies() -> None:
    """Report missing runtime modules without importing GUI code.

    ``find_spec`` keeps this check lightweight and, importantly, lets a clean
    checkout explain how to install dependencies even when PyYAML or Qt is not
    installed yet.
    """

    missing: list[str] = []
    for module_name, distribution_name in _RUNTIME_DEPENDENCIES:
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(distribution_name)

    if missing:
        names = ", ".join(missing)
        raise DependencyPreflightError(
            f"Missing runtime dependencies: {names}. "
            "From the project directory, run "
            "'python -m pip install -r requirements.txt', then retry."
        )


def _load_yaml_mapping(path: Path, *, display_name: str | None = None) -> Mapping[str, Any]:
    """Load one YAML file and reject parse errors and non-mapping roots."""

    label = display_name or path.name
    try:
        yaml = importlib.import_module("yaml")
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyPreflightError(
            "PyYAML is required to validate configs. Install requirements.txt first."
        ) from exc

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, UnicodeError) as exc:
        raise ConfigPreflightError(f"Cannot read {label}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigPreflightError(f"Invalid YAML in {label}: {exc}") from exc

    if not isinstance(value, Mapping):
        raise ConfigPreflightError(f"{label} YAML root must be a mapping.")
    return value


def _require_type(
    config_name: str,
    config: Mapping[str, Any],
    key: str,
    expected_type: type,
) -> None:
    value = config.get(key)
    if not isinstance(value, expected_type):
        expected_name = "mapping" if expected_type is dict else expected_type.__name__
        raise ConfigPreflightError(
            f"{config_name} must contain {key!r} as a {expected_name}."
        )


def _validate_config(path: Path, *, display_name: str | None = None) -> None:
    """Validate the structural fields needed before constructing MainWindow."""

    name = display_name or path.name
    config = _load_yaml_mapping(path, display_name=name)
    if name == "calibration.yaml":
        _require_type(name, config, "canvas", dict)
        _validate_canvas_mapping(name, config["canvas"])
        _require_type(name, config, "topologies", dict)
        topology = config.get("stitch_topology")
        if not isinstance(topology, str) or not topology.strip():
            raise ConfigPreflightError(
                "calibration.yaml must contain 'stitch_topology' as a non-empty string."
            )
        profile = config["topologies"].get(topology)
        if not isinstance(profile, dict):
            raise ConfigPreflightError(
                f"calibration.yaml selected topology is missing: {topology}."
            )
        _require_type("calibration.yaml selected topology", profile, "canvas", dict)
        _validate_canvas_mapping(
            "calibration.yaml selected topology",
            profile["canvas"],
        )
        active_cameras = profile.get("active_cameras")
        if not isinstance(active_cameras, list) or not active_cameras or not all(
            isinstance(camera, str) and camera for camera in active_cameras
        ):
            raise ConfigPreflightError(
                "calibration.yaml selected topology must contain active_cameras as a non-empty string list."
            )
    elif name == "cameras.yaml":
        _require_type(name, config, "camera_order", list)
        _require_type(name, config, "cameras", dict)
        _require_type(name, config, "performance", dict)
        active_count = config.get("active_camera_count")
        if type(active_count) is not int or active_count < 0:
            raise ConfigPreflightError(
                "cameras.yaml active_camera_count must be a non-negative integer."
            )
        if not all(
            isinstance(camera, str) and camera
            for camera in config["camera_order"]
        ):
            raise ConfigPreflightError(
                "cameras.yaml camera_order must contain only non-empty strings."
            )
        for camera, camera_config in config["cameras"].items():
            if not isinstance(camera, str) or not isinstance(camera_config, dict):
                raise ConfigPreflightError(
                    "cameras.yaml cameras entries must map string IDs to mappings."
                )
        performance = config["performance"]
        for key, minimum in (
            ("process_fps", 1),
            ("preview_fps", 1),
            ("max_input_width", 0),
        ):
            value = performance.get(key)
            if value is not None and (
                type(value) is not int or value < minimum
            ):
                raise ConfigPreflightError(
                    f"cameras.yaml performance.{key} must be an integer >= {minimum}."
                )
    elif name == "network.yaml":
        _require_type(name, config, "output", dict)
        _require_type(name, config, "udp_legacy", dict)


def _validate_canvas_mapping(label: str, canvas: Mapping[str, Any]) -> None:
    width = canvas.get("width")
    height = canvas.get("height")
    if (
        type(width) is not int
        or width <= 0
        or type(height) is not int
        or height <= 0
    ):
        raise ConfigPreflightError(
            f"{label} canvas width and height must be positive integers."
        )


def _stage_config_copy(template_path: Path, target_path: Path) -> Path:
    """Create and fsync a same-directory staging file for one config."""

    try:
        payload = template_path.read_bytes()
    except OSError as exc:
        raise ConfigPreflightError(
            f"Cannot read configuration template {template_path.name}: {exc}"
        ) from exc

    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            staged_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return staged_path
    except OSError as exc:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)
        raise ConfigPreflightError(
            f"Cannot stage {target_path.name} for initialization: {exc}"
        ) from exc


def _initialize_missing_configs(
    config_dir: Path,
    missing: Sequence[tuple[str, str]],
    *,
    post_publish_validate: Callable[[], None],
) -> None:
    """Publish all missing local configs or roll the whole initialization back.

    Staging files live beside their targets. ``os.link`` publishes each fully
    written file atomically and refuses to overwrite a target that appeared
    concurrently. Publication and final validation share one transaction
    boundary. If either fails, targets are removed only while they still refer
    to this call's staged file; a concurrently replaced target is preserved and
    reported as a rollback conflict.
    """

    staged: list[tuple[Path, Path]] = []
    created_targets: list[tuple[Path, Path]] = []
    try:
        for target_name, template_name in missing:
            target_path = config_dir / target_name
            template_path = config_dir / template_name
            staged_path = _stage_config_copy(template_path, target_path)
            staged.append((staged_path, target_path))
            # Validate the exact staged bytes, closing the gap between checking
            # a template and copying it if another process edits the template.
            _validate_config(staged_path, display_name=target_name)

        for staged_path, target_path in staged:
            os.link(staged_path, target_path)
            created_targets.append((staged_path, target_path))

        post_publish_validate()
    except Exception as exc:
        rollback_errors: list[str] = []
        rollback_conflicts: list[str] = []
        for staged_path, target_path in reversed(created_targets):
            try:
                if not os.path.lexists(target_path):
                    continue
                try:
                    still_ours = os.path.samefile(staged_path, target_path)
                except OSError as identity_exc:
                    rollback_errors.append(
                        f"cannot verify {target_path.name}: {identity_exc}"
                    )
                    continue
                if not still_ours:
                    rollback_conflicts.append(
                        f"{target_path.name} was replaced concurrently and was preserved"
                    )
                    continue
                target_path.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target_path.name}: {rollback_exc}")

        details: list[str] = []
        if rollback_conflicts:
            details.append(f"Rollback conflicts: {'; '.join(rollback_conflicts)}.")
        if rollback_errors:
            details.append(f"Rollback errors: {'; '.join(rollback_errors)}.")
        detail_text = f" {' '.join(details)}" if details else ""
        raise ConfigPreflightError(
            "Configuration initialization failed; no existing config was "
            f"overwritten: {exc}.{detail_text}"
        ) from exc
    finally:
        for staged_path, _target_path in staged:
            staged_path.unlink(missing_ok=True)


def preflight_configs(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    initialize: bool = False,
) -> None:
    """Validate runtime configs, optionally creating only missing local files.

    The default path is strictly read-only. Explicit initialization copies
    ``cameras.example.yaml`` and ``network.example.yaml`` only when their local
    counterparts are absent. It never creates or replaces calibration data.
    """

    directory = Path(config_dir)
    if not directory.is_dir():
        raise ConfigPreflightError(
            f"Config directory does not exist: {directory}. Restore the tracked configs directory."
        )

    calibration_path = directory / "calibration.yaml"
    if not calibration_path.is_file():
        raise ConfigPreflightError(
            "Missing calibration.yaml. It is device-specific and will not be generated; "
            "restore or provide a validated calibration file."
        )

    # Validate calibration and every existing local file before staging any
    # change, so malformed user data can never be replaced as a side effect.
    _validate_config(calibration_path)
    missing: list[tuple[str, str]] = []
    for target_name, template_name in _LOCAL_CONFIG_TEMPLATES:
        target_path = directory / target_name
        if target_path.exists():
            if not target_path.is_file():
                raise ConfigPreflightError(f"{target_name} is not a regular file.")
            _validate_config(target_path)
        else:
            missing.append((target_name, template_name))

    if missing and not initialize:
        names = ", ".join(name for name, _template in missing)
        raise ConfigPreflightError(
            f"Missing local configs: {names}. Run app.py with "
            "--initialize-configs to copy the tracked safe example files, review them, then retry."
        )

    if missing:
        # Parse and structurally validate every source before publishing any of
        # them. Use the target name so schema validation is applied to examples.
        for target_name, template_name in missing:
            template_path = directory / template_name
            if not template_path.is_file():
                raise ConfigPreflightError(
                    f"Missing tracked template {template_name}; cannot initialize {target_name}."
                )
            _validate_config(template_path, display_name=target_name)
        def validate_published_set() -> None:
            # Validate the final paths while staging identities are still
            # available for a safe rollback.
            for name in ("calibration.yaml", "cameras.yaml", "network.yaml"):
                _validate_config(directory / name)

        _initialize_missing_configs(
            directory,
            missing,
            post_publish_validate=validate_published_set,
        )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepShark View Studio after a safe startup preflight."
    )
    parser.add_argument(
        "--initialize-configs",
        action="store_true",
        help=(
            "copy missing cameras.yaml/network.yaml from tracked examples; "
            "never overwrite existing files or generate calibration.yaml"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate dependencies/configs and exit without importing the GUI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run dependency/config preflight, then lazily enter the Qt application."""

    incoming_args = list(sys.argv[1:] if argv is None else argv)
    options, qt_args = _argument_parser().parse_known_args(incoming_args)

    try:
        check_runtime_dependencies()
        preflight_configs(initialize=options.initialize_configs)
    except (DependencyPreflightError, ConfigPreflightError) as exc:
        print(f"Startup preflight failed: {exc}", file=sys.stderr)
        return 2

    if options.preflight_only:
        print("Startup preflight passed.")
        return 0

    original_argv = list(sys.argv)
    program_name = original_argv[0] if original_argv else "app.py"
    sys.argv[:] = [program_name, *qt_args]
    try:
        try:
            gui_module = importlib.import_module("deep_shark_studio.gui.main_window")
        except (ImportError, OSError) as exc:
            print(
                "GUI dependencies were found but could not be loaded. "
                "Reinstall requirements.txt and check native runtime libraries: "
                f"{exc}",
                file=sys.stderr,
            )
            return 2
        return int(gui_module.main())
    finally:
        sys.argv[:] = original_argv


__all__ = [
    "ConfigPreflightError",
    "DependencyPreflightError",
    "check_runtime_dependencies",
    "main",
    "preflight_configs",
]
