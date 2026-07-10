"""Portable project package export, strict validation, and activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np

from deep_shark_studio.calibration_candidate import load_calibration_candidate
from deep_shark_studio.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    file_revision,
    load_yaml,
    save_yaml,
)
from deep_shark_studio.projection.intrinsics_runtime_loader import (
    load_fisheye_intrinsics_source,
)
from deep_shark_studio.seam.far_field_layout_candidate_runtime import (
    load_far_field_layout_candidate,
)
from deep_shark_studio.seam.layout_candidate_runtime import (
    load_layout_candidate_for_runtime,
)

from .dependency_collector import (
    candidate_yaml_path,
    copy_b2_candidate_directory,
    copy_candidate_directory,
    rewrite_yaml_file_paths,
)
from .path_resolver import (
    ProjectPathError,
    find_absolute_path_strings,
    is_absolute_path_string,
    make_relative_to_package,
    resolve_project_path,
)


PROJECT_PACKAGE_SCHEMA_VERSION = 1
PROJECT_PACKAGE_TYPE = "deep_shark_project"
MANIFEST_NAME = "project.dcsvs.yaml"
ACTIVE_PROJECT_NAME = "active_project.yaml"
ACTIVE_PROJECT_FORMAT = "DeepSharkActiveProject"
ACTIVE_PROJECT_SCHEMA_VERSION = 1
RUNTIME_CAMERAS = ("front_left", "front", "front_right")
ACTIVE_FILES = (
    "calibration.yaml",
    "cameras.yaml",
    "network.yaml",
    ACTIVE_PROJECT_NAME,
)


class ProjectPackageExportError(RuntimeError):
    """Raised when a complete, valid package cannot be published."""


class ProjectPackageActivationError(RuntimeError):
    """Raised when activation fails and rollback also needs attention."""


class ActiveProjectStateError(RuntimeError):
    """Raised when persisted active-package state is missing or unsafe."""


@dataclass(frozen=True)
class ProjectPackageExportResult:
    package_root: Path
    manifest_path: Path
    report_path: Path
    warnings: tuple[str, ...]
    copied_files: tuple[Path, ...]


@dataclass(frozen=True)
class ProjectPackageValidationResult:
    package_root: Path
    manifest_path: Path
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    report_path: Path | None = None


@dataclass(frozen=True)
class ProjectPackageActivationResult:
    package_root: Path
    manifest_path: Path
    backup_path: Path | None
    active_state_path: Path
    activated_configs: tuple[Path, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ActiveProjectRuntimeDefaults:
    manifest_path: Path
    package_root: Path
    default_stitch_mode: str
    live_stitch_strategy: str
    use_far_field_custom: bool
    far_field_layout_candidate: Path | None
    near_field_layout_candidate: Path | None
    b2_candidate: Path | None
    fisheye_intrinsics_source: Path | None
    near_field_projection_source: str


def export_project_package(
    output_root: str | Path,
    project_name: str = "DeepShark_Project",
    *,
    calibration_config_path: str | Path | None = None,
    cameras_config_path: str | Path | None = None,
    network_config_path: str | Path | None = None,
    far_field_candidate_path: str | Path | None = None,
    near_field_candidate_path: str | Path | None = None,
    b2_candidate_path: str | Path | None = None,
    fisheye_intrinsics_path: str | Path | None = None,
    default_stitch_mode: str = "far_field",
    live_stitch_strategy: str = "template",
    use_far_field_custom: bool = False,
    near_field_projection_source: str = "current_perspective",
    include_preview_assets: bool = True,
    include_debug_assets: bool = False,
) -> ProjectPackageExportResult:
    """Build in a hidden same-volume staging directory, then publish once."""

    output_directory = Path(output_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    final_root = _unique_package_root(output_directory, project_name)
    staging_root = Path(
        tempfile.mkdtemp(
            dir=output_directory,
            prefix=f".{_safe_project_name(project_name)}.staging-",
        )
    )
    published = False
    try:
        staged_result = _build_project_package(
            staging_root,
            project_name=project_name,
            calibration_config_path=calibration_config_path,
            cameras_config_path=cameras_config_path,
            network_config_path=network_config_path,
            far_field_candidate_path=far_field_candidate_path,
            near_field_candidate_path=near_field_candidate_path,
            b2_candidate_path=b2_candidate_path,
            fisheye_intrinsics_path=fisheye_intrinsics_path,
            default_stitch_mode=default_stitch_mode,
            live_stitch_strategy=live_stitch_strategy,
            use_far_field_custom=use_far_field_custom,
            near_field_projection_source=near_field_projection_source,
            include_preview_assets=include_preview_assets,
            include_debug_assets=include_debug_assets,
        )
        validation = validate_project_package(staging_root, write_report=False)
        if not validation.valid:
            raise ProjectPackageExportError(
                "Exported package failed strict self-validation: "
                + "; ".join(validation.errors)
            )
        if final_root.exists():
            raise FileExistsError(f"Package destination already exists: {final_root}")
        resolved_stage = staging_root.resolve()
        report_relative = staged_result.report_path.resolve().relative_to(
            resolved_stage
        )
        copied_relatives = tuple(
            copied_path.resolve().relative_to(resolved_stage)
            for copied_path in staged_result.copied_files
        )
        os.replace(staging_root, final_root)
        published = True
        return ProjectPackageExportResult(
            package_root=final_root,
            manifest_path=final_root / MANIFEST_NAME,
            report_path=final_root / report_relative,
            warnings=staged_result.warnings,
            copied_files=tuple(
                final_root / relative_path for relative_path in copied_relatives
            ),
        )
    finally:
        if not published and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def _build_project_package(
    package_root: Path,
    *,
    project_name: str,
    calibration_config_path: str | Path | None,
    cameras_config_path: str | Path | None,
    network_config_path: str | Path | None,
    far_field_candidate_path: str | Path | None,
    near_field_candidate_path: str | Path | None,
    b2_candidate_path: str | Path | None,
    fisheye_intrinsics_path: str | Path | None,
    default_stitch_mode: str,
    live_stitch_strategy: str,
    use_far_field_custom: bool,
    near_field_projection_source: str,
    include_preview_assets: bool,
    include_debug_assets: bool,
) -> ProjectPackageExportResult:
    warnings: list[str] = []
    copied_files: list[Path] = []
    config_sources = {
        "calibration": Path(calibration_config_path or CONFIG_DIR / "calibration.yaml"),
        "cameras": Path(cameras_config_path or CONFIG_DIR / "cameras.yaml"),
        "network": Path(network_config_path or CONFIG_DIR / "network.yaml"),
    }
    config_manifest: dict[str, str] = {}
    for key, source in config_sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required config is missing: {source}")
        source_data = _load_yaml_mapping_or_raise(source, f"{key} config")
        absolute_config_paths = find_absolute_path_strings(source_data)
        if absolute_config_paths:
            raise ProjectPackageExportError(
                f"Config {source.name} contains machine-specific absolute paths: "
                f"{absolute_config_paths}"
            )
        destination = package_root / "configs" / f"{key}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_files.append(destination)
        config_manifest[key] = make_relative_to_package(package_root, destination)

    inferred_b2 = _infer_b2_candidate_path(far_field_candidate_path)
    if b2_candidate_path is None and inferred_b2 is not None:
        b2_candidate_path = inferred_b2
    elif b2_candidate_path is not None and inferred_b2 is not None:
        if candidate_yaml_path(b2_candidate_path).resolve() != candidate_yaml_path(
            inferred_b2
        ).resolve():
            raise ProjectPackageExportError(
                "Far-field candidate and explicit B-2 dependency refer to different candidates."
            )
    inferred_fisheye = _infer_fisheye_intrinsics_path(near_field_candidate_path)
    if fisheye_intrinsics_path is None and inferred_fisheye is not None:
        fisheye_intrinsics_path = inferred_fisheye
    elif fisheye_intrinsics_path is not None and inferred_fisheye is not None:
        if candidate_yaml_path(fisheye_intrinsics_path).resolve() != candidate_yaml_path(
            inferred_fisheye
        ).resolve():
            raise ProjectPackageExportError(
                "Near-field candidate and explicit fisheye dependency refer to different candidates."
            )

    candidate_manifest: dict[str, dict[str, Any]] = {}
    known_rewrites: dict[str, str] = {}

    b2_dest: Path | None = None
    if b2_candidate_path is not None:
        b2_source = candidate_yaml_path(b2_candidate_path)
        if not b2_source.is_file():
            raise FileNotFoundError(f"B-2 candidate is missing: {b2_source}")
        b2_dest = package_root / "candidates" / "b2_candidate"
        copied_files.extend(
            copy_b2_candidate_directory(
                b2_source,
                b2_dest,
                include_preview_assets=include_preview_assets,
            )
        )
        known_rewrites[str(b2_source.parent.resolve())] = "candidates/b2_candidate"
        known_rewrites[str(b2_source.resolve())] = "candidates/b2_candidate/candidate.yaml"
        candidate_manifest["b2_candidate"] = {
            "path": "candidates/b2_candidate/candidate.yaml",
            "source_type": "b2_calibration_candidate",
        }

    fisheye_dest: Path | None = None
    if fisheye_intrinsics_path is not None:
        fisheye_source = candidate_yaml_path(fisheye_intrinsics_path)
        if not fisheye_source.is_file():
            raise FileNotFoundError(
                f"Fisheye intrinsics source is missing: {fisheye_source}"
            )
        fisheye_dest = package_root / "candidates" / "fisheye_intrinsics"
        copied_files.extend(
            copy_candidate_directory(
                fisheye_source,
                fisheye_dest,
                include_preview_assets=include_preview_assets,
                include_debug_assets=include_debug_assets,
            )
        )
        known_rewrites[str(fisheye_source.parent.resolve())] = (
            "candidates/fisheye_intrinsics"
        )
        known_rewrites[str(fisheye_source.resolve())] = (
            "candidates/fisheye_intrinsics/candidate.yaml"
        )
        candidate_manifest["fisheye_intrinsics"] = {
            "path": "candidates/fisheye_intrinsics/candidate.yaml",
            "source_type": "opencv_fisheye_intrinsics",
        }

    if far_field_candidate_path is not None:
        source = candidate_yaml_path(far_field_candidate_path)
        if not source.is_file():
            raise FileNotFoundError(f"Far-field candidate is missing: {source}")
        destination = package_root / "candidates" / "far_field_layout"
        copied_files.extend(
            copy_candidate_directory(
                source,
                destination,
                include_preview_assets=include_preview_assets,
                include_debug_assets=include_debug_assets,
            )
        )
        known_rewrites[str(source.parent.resolve())] = "candidates/far_field_layout"
        if b2_dest is not None:
            _rewrite_far_field_candidate(
                destination / "candidate.yaml",
                "../b2_candidate",
            )
        candidate_manifest["far_field_layout"] = {
            "path": "candidates/far_field_layout/candidate.yaml",
            "candidate_type": "far_field_layout",
        }

    if near_field_candidate_path is not None:
        source = candidate_yaml_path(near_field_candidate_path)
        if not source.is_file():
            raise FileNotFoundError(f"Near-field candidate is missing: {source}")
        destination = package_root / "candidates" / "near_field_layout"
        copied_files.extend(
            copy_candidate_directory(
                source,
                destination,
                include_preview_assets=include_preview_assets,
                include_debug_assets=include_debug_assets,
            )
        )
        known_rewrites[str(source.parent.resolve())] = "candidates/near_field_layout"
        if fisheye_dest is not None:
            _rewrite_near_field_candidate(
                destination / "candidate.yaml",
                "../fisheye_intrinsics/candidate.yaml",
            )
        candidate_manifest["near_field_layout"] = {
            "path": "candidates/near_field_layout/candidate.yaml",
            "candidate_type": "front_priority_layout",
        }

    for yaml_file in sorted(package_root.rglob("*.yaml")):
        rewrite_yaml_file_paths(yaml_file, known_rewrites, warnings)
    for yaml_file in sorted(package_root.rglob("*.yml")):
        rewrite_yaml_file_paths(yaml_file, known_rewrites, warnings)

    runtime_defaults = {
        "default_stitch_mode": default_stitch_mode,
        "live_stitch_strategy": live_stitch_strategy,
        "use_far_field_custom": use_far_field_custom,
        "far_field_layout_candidate": _candidate_path_or_none(
            candidate_manifest, "far_field_layout"
        ),
        "near_field_layout_candidate": _candidate_path_or_none(
            candidate_manifest, "near_field_layout"
        ),
        "b2_candidate": _candidate_path_or_none(
            candidate_manifest, "b2_candidate"
        ),
        "near_field_projection_source": near_field_projection_source,
        "fisheye_intrinsics_source": _candidate_path_or_none(
            candidate_manifest, "fisheye_intrinsics"
        ),
    }
    manifest = {
        "schema_version": PROJECT_PACKAGE_SCHEMA_VERSION,
        "package_type": PROJECT_PACKAGE_TYPE,
        "project": {
            "name": project_name,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "exported_by": "DeepSharkViewStudio",
        },
        "configs": config_manifest,
        "runtime_defaults": runtime_defaults,
        "candidates": candidate_manifest,
        "paths": {"use_relative_paths": True},
    }
    manifest_path = package_root / MANIFEST_NAME
    save_yaml(manifest_path, manifest)
    copied_files.append(manifest_path)

    readme_path = package_root / "README_使用说明.md"
    readme_path.write_text(_readme_text(project_name, manifest), encoding="utf-8")
    copied_files.append(readme_path)
    report_path = package_root / "export_report.yaml"
    save_yaml(
        report_path,
        {
            "schema_version": 1,
            "package_root": ".",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "warnings": warnings,
            "copied_files": [
                make_relative_to_package(package_root, path)
                for path in sorted(set(copied_files))
                if path.exists()
            ],
        },
    )
    copied_files.append(report_path)
    return ProjectPackageExportResult(
        package_root=package_root,
        manifest_path=manifest_path,
        report_path=report_path,
        warnings=tuple(warnings),
        copied_files=tuple(copied_files),
    )


def validate_project_package(
    path: str | Path,
    *,
    write_report: bool = False,
) -> ProjectPackageValidationResult:
    """Validate without changing the package unless a report is requested."""

    try:
        return _validate_project_package(path, write_report=write_report)
    except Exception as exc:
        try:
            manifest_path = _manifest_path(path)
            package_root = manifest_path.parent
        except Exception:
            package_root = Path.cwd()
            manifest_path = package_root / MANIFEST_NAME
        return ProjectPackageValidationResult(
            package_root=package_root,
            manifest_path=manifest_path,
            valid=False,
            errors=(f"Project package validation failed safely: {type(exc).__name__}: {exc}",),
            warnings=(),
            report_path=None,
        )


def _validate_project_package(
    path: str | Path,
    *,
    write_report: bool,
) -> ProjectPackageValidationResult:
    manifest_path = _manifest_path(path)
    package_root = manifest_path.parent.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.is_file():
        return ProjectPackageValidationResult(
            package_root=package_root,
            manifest_path=manifest_path,
            valid=False,
            errors=(f"Manifest not found: {manifest_path}",),
            warnings=(),
            report_path=None,
        )

    yaml_documents = _scan_package_yaml(package_root, errors)
    manifest = yaml_documents.get(manifest_path.resolve())
    if not isinstance(manifest, dict):
        if not any("manifest" in error.lower() for error in errors):
            errors.append("Manifest YAML root must be a mapping.")
        return _validation_result(
            package_root,
            manifest_path,
            errors,
            warnings,
            write_report,
        )

    if type(manifest.get("schema_version")) is not int or (
        manifest.get("schema_version") != PROJECT_PACKAGE_SCHEMA_VERSION
    ):
        errors.append(
            f"Unsupported schema_version: {manifest.get('schema_version')}"
        )
    if manifest.get("package_type") != PROJECT_PACKAGE_TYPE:
        errors.append(f"Unsupported package_type: {manifest.get('package_type')}")
    if not isinstance(manifest.get("project"), dict):
        errors.append("manifest.project must be a mapping.")

    paths_block = manifest.get("paths")
    if not isinstance(paths_block, dict):
        errors.append("manifest.paths must be a mapping.")
    elif paths_block.get("use_relative_paths") is not True:
        errors.append("manifest.paths.use_relative_paths must be true.")

    absolute_manifest_paths = find_absolute_path_strings(manifest)
    if absolute_manifest_paths:
        errors.append(f"Manifest contains absolute paths: {absolute_manifest_paths}")

    configs = manifest.get("configs")
    if not isinstance(configs, dict):
        errors.append("manifest.configs is required and must be a mapping.")
        configs = {}
    config_paths: dict[str, Path] = {}
    config_data: dict[str, dict[str, Any]] = {}
    for key in ("calibration", "cameras", "network"):
        resolved = _validate_relative_file(
            package_root,
            configs.get(key),
            f"configs.{key}",
            errors,
        )
        if resolved is None:
            continue
        config_paths[key] = resolved
        data = yaml_documents.get(resolved.resolve())
        if isinstance(data, dict):
            config_data[key] = data
        else:
            errors.append(f"configs.{key} YAML root must be a mapping.")

    if "calibration" in config_data:
        _validate_calibration_config(config_data["calibration"], errors)
    if "cameras" in config_data:
        _validate_cameras_config(config_data["cameras"], errors)
    if "network" in config_data:
        _validate_network_config(config_data["network"], errors)

    candidates = manifest.get("candidates")
    if candidates is None:
        candidates = {}
    if not isinstance(candidates, dict):
        errors.append("manifest.candidates must be a mapping.")
        candidates = {}

    candidate_paths: dict[str, Path] = {}
    candidate_data: dict[str, dict[str, Any]] = {}
    metadata_contracts = {
        "far_field_layout": ("candidate_type", "far_field_layout"),
        "near_field_layout": ("candidate_type", "front_priority_layout"),
        "b2_candidate": ("source_type", "b2_calibration_candidate"),
        "fisheye_intrinsics": ("source_type", "opencv_fisheye_intrinsics"),
    }
    for key, (metadata_key, expected_metadata) in metadata_contracts.items():
        item = candidates.get(key)
        if item is None:
            warnings.append(f"Optional candidate missing: {key}")
            continue
        if not isinstance(item, dict):
            errors.append(f"candidates.{key} must be a mapping.")
            continue
        if item.get(metadata_key) != expected_metadata:
            errors.append(
                f"candidates.{key}.{metadata_key} must be {expected_metadata}."
            )
        resolved = _validate_relative_file(
            package_root,
            item.get("path"),
            f"candidates.{key}.path",
            errors,
        )
        if resolved is None:
            continue
        candidate_paths[key] = resolved
        data = yaml_documents.get(resolved.resolve())
        if isinstance(data, dict):
            candidate_data[key] = data
        else:
            errors.append(f"candidates.{key} YAML root must be a mapping.")

    profile_id = "triple_front_panorama"
    calibration = config_data.get("calibration")
    if isinstance(calibration, dict) and isinstance(
        calibration.get("stitch_topology"), str
    ):
        profile_id = calibration["stitch_topology"]
    calibration_path = config_paths.get("calibration")

    b2_runtime = _validate_b2_candidate(
        package_root,
        candidate_paths.get("b2_candidate"),
        candidate_data.get("b2_candidate"),
        profile_id,
        errors,
    )
    _validate_fisheye_candidate(
        candidate_paths.get("fisheye_intrinsics"),
        errors,
    )
    far_runtime = _validate_far_candidate(
        package_root,
        candidate_paths.get("far_field_layout"),
        candidate_data.get("far_field_layout"),
        candidate_paths.get("b2_candidate"),
        profile_id,
        calibration_path,
        errors,
    )
    near_runtime = _validate_near_candidate(
        package_root,
        candidate_paths.get("near_field_layout"),
        candidate_data.get("near_field_layout"),
        candidate_paths.get("fisheye_intrinsics"),
        profile_id,
        calibration_path,
        errors,
    )

    defaults = manifest.get("runtime_defaults")
    if not isinstance(defaults, dict):
        errors.append("manifest.runtime_defaults must be a mapping.")
        defaults = {}
    _validate_runtime_defaults(
        package_root,
        defaults,
        candidate_paths,
        far_runtime,
        near_runtime,
        b2_runtime,
        errors,
    )

    return _validation_result(
        package_root,
        manifest_path,
        errors,
        warnings,
        write_report,
    )


def activate_project_package(
    path: str | Path,
    *,
    active_config_dir: str | Path | None = None,
    backup_root: str | Path | None = None,
) -> ProjectPackageActivationResult:
    """Back up four active files, stage replacements, and roll back on failure."""

    validation = validate_project_package(path, write_report=False)
    if not validation.valid:
        raise ValueError("Project package is not valid: " + "; ".join(validation.errors))
    manifest_revision = file_revision(validation.manifest_path)
    manifest = _load_yaml_mapping_or_raise(validation.manifest_path, "manifest")
    config_dir = Path(active_config_dir or CONFIG_DIR)
    config_dir.mkdir(parents=True, exist_ok=True)
    selected_backup_root = (
        Path(backup_root)
        if backup_root is not None
        else PROJECT_ROOT / "projects" / "backups" / "project_packages"
        if active_config_dir is None
        else config_dir.parent / "backups"
    )

    originals: dict[str, bytes | None] = {}
    for name in ACTIVE_FILES:
        target = config_dir / name
        originals[name] = target.read_bytes() if target.is_file() else None

    backup_path = _backup_config_directory(config_dir, selected_backup_root)
    activation_stage = Path(
        tempfile.mkdtemp(dir=config_dir, prefix=".activation-stage-")
    )
    activated: list[Path] = []
    try:
        configs = manifest["configs"]
        for key, filename in (
            ("calibration", "calibration.yaml"),
            ("cameras", "cameras.yaml"),
            ("network", "network.yaml"),
        ):
            source = resolve_project_path(validation.package_root, configs[key])
            staged_config = activation_stage / filename
            shutil.copy2(source, staged_config)
            if file_revision(source) != file_revision(staged_config):
                raise OSError(f"Staged config copy verification failed: {filename}")

        staged_errors: list[str] = []
        _validate_calibration_config(
            _load_yaml_mapping_or_raise(
                activation_stage / "calibration.yaml", "staged calibration"
            ),
            staged_errors,
        )
        _validate_cameras_config(
            _load_yaml_mapping_or_raise(
                activation_stage / "cameras.yaml", "staged cameras"
            ),
            staged_errors,
        )
        _validate_network_config(
            _load_yaml_mapping_or_raise(
                activation_stage / "network.yaml", "staged network"
            ),
            staged_errors,
        )
        if staged_errors:
            raise ValueError(
                "Staged package configs are invalid: " + "; ".join(staged_errors)
            )
        if file_revision(validation.manifest_path) != manifest_revision:
            raise ValueError("Project package manifest changed during activation.")
        final_validation = validate_project_package(
            validation.manifest_path,
            write_report=False,
        )
        if not final_validation.valid:
            raise ValueError(
                "Project package changed during activation: "
                + "; ".join(final_validation.errors)
            )

        state_path = activation_stage / ACTIVE_PROJECT_NAME
        save_yaml(
            state_path,
            {
                "format": ACTIVE_PROJECT_FORMAT,
                "schema_version": ACTIVE_PROJECT_SCHEMA_VERSION,
                "manifest_path": str(validation.manifest_path.resolve()),
                "manifest_sha256": manifest_revision,
            },
        )

        try:
            for filename in ACTIVE_FILES:
                source = activation_stage / filename
                destination = config_dir / filename
                os.replace(source, destination)
                if filename != ACTIVE_PROJECT_NAME:
                    activated.append(destination)
        except Exception as publish_error:
            rollback_errors = _rollback_active_files(config_dir, originals)
            if rollback_errors:
                raise ProjectPackageActivationError(
                    f"Activation failed: {publish_error}. Rollback errors: "
                    + "; ".join(rollback_errors)
                ) from publish_error
            raise
    finally:
        if activation_stage.exists():
            shutil.rmtree(activation_stage, ignore_errors=True)

    active_state_path = config_dir / ACTIVE_PROJECT_NAME
    return ProjectPackageActivationResult(
        package_root=validation.package_root,
        manifest_path=validation.manifest_path,
        backup_path=backup_path,
        active_state_path=active_state_path,
        activated_configs=tuple(activated),
        warnings=validation.warnings,
    )


def load_active_project_runtime_defaults(
    *,
    active_config_dir: str | Path = CONFIG_DIR,
) -> ActiveProjectRuntimeDefaults | None:
    """Resolve the active package after restart without mutating package data."""

    state_path = Path(active_config_dir) / ACTIVE_PROJECT_NAME
    if not state_path.exists():
        return None
    try:
        state = _load_yaml_mapping_or_raise(state_path, ACTIVE_PROJECT_NAME)
        if state.get("format") != ACTIVE_PROJECT_FORMAT:
            raise ActiveProjectStateError("Unsupported active project format.")
        if state.get("schema_version") != ACTIVE_PROJECT_SCHEMA_VERSION:
            raise ActiveProjectStateError("Unsupported active project schema_version.")
        raw_manifest_path = state.get("manifest_path")
        if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip():
            raise ActiveProjectStateError("active_project.yaml has no manifest_path.")
        manifest_path = Path(raw_manifest_path)
        if not manifest_path.is_absolute():
            raise ActiveProjectStateError("active project manifest_path must be absolute.")
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise ActiveProjectStateError(
                f"Active project manifest is missing: {manifest_path}"
            )
        expected_revision = state.get("manifest_sha256")
        if not isinstance(expected_revision, str) or (
            file_revision(manifest_path) != expected_revision
        ):
            raise ActiveProjectStateError(
                "Active project manifest hash does not match activation state."
            )
        validation = validate_project_package(manifest_path, write_report=False)
        if not validation.valid:
            raise ActiveProjectStateError(
                "Active project package no longer validates: "
                + "; ".join(validation.errors)
            )
        manifest = _load_yaml_mapping_or_raise(manifest_path, "manifest")
        defaults = manifest["runtime_defaults"]
        candidates = manifest.get("candidates") or {}

        def runtime_path(default_key: str, candidate_key: str) -> Path | None:
            raw = defaults.get(default_key)
            if raw is None and candidate_key == "b2_candidate":
                item = candidates.get(candidate_key)
                raw = item.get("path") if isinstance(item, dict) else None
            if raw is None:
                return None
            return resolve_project_path(validation.package_root, raw)

        return ActiveProjectRuntimeDefaults(
            manifest_path=manifest_path,
            package_root=validation.package_root,
            default_stitch_mode=str(defaults["default_stitch_mode"]),
            live_stitch_strategy=str(
                defaults.get("live_stitch_strategy", "template")
            ),
            use_far_field_custom=bool(defaults["use_far_field_custom"]),
            far_field_layout_candidate=runtime_path(
                "far_field_layout_candidate", "far_field_layout"
            ),
            near_field_layout_candidate=runtime_path(
                "near_field_layout_candidate", "near_field_layout"
            ),
            b2_candidate=runtime_path("b2_candidate", "b2_candidate"),
            fisheye_intrinsics_source=runtime_path(
                "fisheye_intrinsics_source", "fisheye_intrinsics"
            ),
            near_field_projection_source=str(
                defaults["near_field_projection_source"]
            ),
        )
    except ActiveProjectStateError:
        raise
    except Exception as exc:
        raise ActiveProjectStateError(
            f"Cannot restore active project state: {type(exc).__name__}: {exc}"
        ) from exc


def _scan_package_yaml(
    package_root: Path,
    errors: list[str],
) -> dict[Path, dict[str, Any]]:
    documents: dict[Path, dict[str, Any]] = {}
    yaml_files = set(package_root.rglob("*.yaml")) | set(
        package_root.rglob("*.yml")
    )
    for yaml_path in sorted(yaml_files, key=lambda item: item.as_posix()):
        try:
            resolved = yaml_path.resolve()
            resolved.relative_to(package_root)
        except (OSError, RuntimeError, ValueError):
            errors.append(
                f"YAML path escapes package root: {yaml_path.relative_to(package_root)}"
            )
            continue
        if not yaml_path.is_file():
            errors.append(
                f"Package YAML is not a regular file: {yaml_path.relative_to(package_root)}"
            )
            continue
        try:
            data = load_yaml(yaml_path)
        except Exception as exc:
            errors.append(
                f"Invalid YAML {yaml_path.relative_to(package_root).as_posix()}: {exc}"
            )
            continue
        if not isinstance(data, dict):
            errors.append(
                f"YAML root must be a mapping: {yaml_path.relative_to(package_root).as_posix()}"
            )
            continue
        absolute = find_absolute_path_strings(data)
        if absolute:
            errors.append(
                "YAML contains absolute paths: "
                f"{yaml_path.relative_to(package_root).as_posix()}: {absolute}"
            )
        documents[resolved] = data
    return documents


def _validation_result(
    package_root: Path,
    manifest_path: Path,
    errors: list[str],
    warnings: list[str],
    write_report: bool,
) -> ProjectPackageValidationResult:
    report_path: Path | None = None
    if write_report and package_root.is_dir():
        candidate_report_path = package_root / "project_validation_report.yaml"
        try:
            save_yaml(
                candidate_report_path,
                {
                    "schema_version": 1,
                    "valid": not errors,
                    "errors": errors,
                    "warnings": warnings,
                    "manifest": MANIFEST_NAME,
                },
            )
            report_path = candidate_report_path
        except Exception as exc:
            warnings.append(f"Validation report could not be written: {exc}")
    return ProjectPackageValidationResult(
        package_root=package_root,
        manifest_path=manifest_path,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        report_path=report_path,
    )


def _validate_calibration_config(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    _validate_canvas(data.get("canvas"), "configs.calibration canvas", errors)
    topology = data.get("stitch_topology")
    if not isinstance(topology, str) or not topology.strip():
        errors.append("configs.calibration.stitch_topology must be a non-empty string.")
        return
    topologies = data.get("topologies")
    if not isinstance(topologies, dict):
        errors.append("configs.calibration.topologies must be a mapping.")
        return
    profile = topologies.get(topology)
    if not isinstance(profile, dict):
        errors.append(
            f"configs.calibration selected topology is missing: {topology}"
        )
        return
    _validate_canvas(profile.get("canvas"), "configs.calibration topology canvas", errors)
    active_cameras = profile.get("active_cameras")
    if not isinstance(active_cameras, list) or not active_cameras or not all(
        isinstance(camera, str) and camera for camera in active_cameras
    ):
        errors.append(
            "configs.calibration selected topology active_cameras must be a non-empty string list."
        )


def _validate_canvas(value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a mapping.")
        return
    width = value.get("width")
    height = value.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        errors.append(f"{label} width and height must be positive integers.")


def _validate_cameras_config(data: dict[str, Any], errors: list[str]) -> None:
    count = data.get("active_camera_count")
    if type(count) is not int or count < 0:
        errors.append("configs.cameras.active_camera_count must be a non-negative integer.")
    order = data.get("camera_order")
    if not isinstance(order, list) or not all(
        isinstance(camera, str) and camera for camera in order
    ):
        errors.append("configs.cameras.camera_order must be a string list.")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        errors.append("configs.cameras.cameras must be a mapping.")
    elif not all(
        isinstance(camera, str) and isinstance(camera_config, dict)
        for camera, camera_config in cameras.items()
    ):
        errors.append(
            "configs.cameras.cameras entries must map string IDs to mappings."
        )
    performance = data.get("performance")
    if not isinstance(performance, dict):
        errors.append("configs.cameras.performance must be a mapping.")
    else:
        for key, minimum in (
            ("process_fps", 1),
            ("preview_fps", 1),
            ("max_input_width", 0),
        ):
            value = performance.get(key)
            if value is not None and (
                type(value) is not int or value < minimum
            ):
                errors.append(
                    f"configs.cameras.performance.{key} must be an integer >= {minimum}."
                )


def _validate_network_config(data: dict[str, Any], errors: list[str]) -> None:
    for key in ("output", "udp_legacy"):
        if not isinstance(data.get(key), dict):
            errors.append(f"configs.network.{key} must be a mapping.")


def _validate_far_candidate(
    package_root: Path,
    path: Path | None,
    data: dict[str, Any] | None,
    b2_path: Path | None,
    profile_id: str,
    calibration_path: Path | None,
    errors: list[str],
) -> Any | None:
    if path is None or data is None or calibration_path is None:
        return None
    projection = data.get("projection")
    projection_source = (
        projection.get("source") if isinstance(projection, dict) else None
    )
    resolved_b2: Path | None = None
    if projection_source == "b2_far_field_candidate":
        raw = projection.get("candidate_directory")
        resolved_b2 = _resolve_embedded_path(
            package_root,
            path.parent,
            raw,
            "Far-field projection candidate_directory",
            errors,
        )
        if resolved_b2 is None:
            return None
    try:
        candidate = load_far_field_layout_candidate(
            path,
            expected_profile_id=profile_id,
            formal_calibration_path=calibration_path,
        )
    except Exception as exc:
        errors.append(f"Far-field candidate is not runtime-safe: {exc}")
        return None
    if candidate.projection_source != "b2_far_field_candidate":
        errors.append(
            "Packaged Far-field layout candidates must use B-2 per-camera projection."
        )
        return candidate
    if candidate.projection_source == "b2_far_field_candidate":
        if b2_path is None:
            errors.append("Far-field B-2 projection requires candidates.b2_candidate.")
        elif resolved_b2 != b2_path.parent.resolve():
            errors.append(
                "Far-field projection dependency does not match the declared B-2 candidate."
            )
        blend = data.get("far_field_blend")
        if not isinstance(blend, dict) or blend.get("mode") != "b2_weight_selection":
            errors.append(
                "Far-field B-2 projection requires far_field_blend.mode=b2_weight_selection."
            )
    return candidate


def _validate_near_candidate(
    package_root: Path,
    path: Path | None,
    data: dict[str, Any] | None,
    fisheye_path: Path | None,
    profile_id: str,
    calibration_path: Path | None,
    errors: list[str],
) -> Any | None:
    if path is None or data is None or calibration_path is None:
        return None
    projection = data.get("projection")
    projection_source = (
        projection.get("source") if isinstance(projection, dict) else None
    )
    resolved_fisheye: Path | None = None
    if projection_source in {
        "fisheye_rectilinear",
        "fisheye_rectilinear_candidate",
    }:
        raw = projection.get("intrinsics_source_path")
        resolved_fisheye = _resolve_embedded_path(
            package_root,
            path.parent,
            raw,
            "Near-field intrinsics_source_path",
            errors,
        )
        if resolved_fisheye is None:
            return None
    try:
        candidate = load_layout_candidate_for_runtime(
            path,
            expected_profile_id=profile_id,
            formal_calibration_path=calibration_path,
        )
    except Exception as exc:
        errors.append(f"Near-field candidate is not runtime-safe: {exc}")
        return None
    if candidate.projection.source.value == "fisheye_rectilinear_candidate":
        if fisheye_path is None:
            errors.append(
                "Near-field fisheye projection requires candidates.fisheye_intrinsics."
            )
        elif resolved_fisheye != fisheye_path.resolve():
            errors.append(
                "Near-field intrinsics dependency does not match the declared fisheye candidate."
            )
    return candidate


def _validate_b2_candidate(
    package_root: Path,
    path: Path | None,
    data: dict[str, Any] | None,
    profile_id: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if path is None or data is None:
        return None
    report_path = path.parent / "report.yaml"
    if not _package_file_is_safe(
        package_root,
        report_path,
        "B-2 report.yaml",
        errors,
    ):
        return None
    try:
        candidate, _report = load_calibration_candidate(path.parent)
    except Exception as exc:
        errors.append(f"B-2 candidate is not runtime-safe: {exc}")
        return None
    if candidate.get("topology") not in (None, profile_id):
        errors.append("B-2 candidate topology does not match calibration topology.")
    rig = candidate.get("rig")
    if not isinstance(rig, dict) or rig.get("complete") is not True:
        errors.append("B-2 candidate rig.complete must be true.")
        return candidate
    transforms = rig.get("transforms")
    if not isinstance(transforms, dict):
        errors.append("B-2 candidate rig.transforms must be a mapping.")
    else:
        for camera in RUNTIME_CAMERAS:
            transform = transforms.get(camera)
            rotation = (
                transform.get("rotation_camera_to_front")
                if isinstance(transform, dict)
                else None
            )
            try:
                matrix = np.asarray(rotation, dtype=np.float64)
            except Exception as exc:
                errors.append(f"B-2 transform for {camera} is not numeric: {exc}")
                continue
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                errors.append(
                    f"B-2 transform for {camera} must be a finite 3x3 matrix."
                )

    panorama = candidate.get("virtual_panorama")
    if not isinstance(panorama, dict):
        errors.append("B-2 virtual_panorama must be a mapping.")
        return candidate
    canvas_size = panorama.get("canvas_size")
    if (
        not isinstance(canvas_size, (list, tuple))
        or len(canvas_size) != 2
        or type(canvas_size[0]) is not int
        or type(canvas_size[1]) is not int
        or canvas_size[0] <= 0
        or canvas_size[1] <= 0
        or canvas_size[0] > 5000
        or canvas_size[1] > 3000
    ):
        errors.append(
            "B-2 virtual_panorama.canvas_size must contain bounded positive integers."
        )
        return candidate
    expected_shape = (int(canvas_size[1]), int(canvas_size[0]))
    files = panorama.get("files")
    remaps = files.get("remaps") if isinstance(files, dict) else None
    if not isinstance(remaps, dict):
        errors.append("B-2 virtual_panorama.files.remaps must be a mapping.")
        return candidate
    for camera in RUNTIME_CAMERAS:
        raw = remaps.get(camera)
        if not isinstance(raw, str) or not raw:
            errors.append(f"B-2 remap declaration missing for {camera}.")
            continue
        try:
            remap_path = resolve_project_path(path.parent, raw)
        except ProjectPathError as exc:
            errors.append(f"B-2 remap path for {camera}: {exc}")
            continue
        try:
            remap_path.resolve().relative_to(package_root)
        except (OSError, RuntimeError, ValueError):
            errors.append(f"B-2 remap path for {camera} escapes package root.")
            continue
        _validate_remap_npz(remap_path, camera, expected_shape, errors)
    return candidate


def _validate_remap_npz(
    path: Path,
    camera: str,
    expected_shape: tuple[int, int],
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"B-2 remap NPZ missing for {camera}: {path.name}")
        return
    expected_archive_names = {"map_x.npy", "map_y.npy", "valid_mask.npy"}
    maximum_uncompressed_bytes = (
        expected_shape[0] * expected_shape[1] * 9 + 4096
    )
    try:
        with zipfile.ZipFile(path) as zip_archive:
            members = zip_archive.infolist()
            member_names = {member.filename for member in members}
            if member_names != expected_archive_names:
                errors.append(
                    f"B-2 remap NPZ entries for {camera} are invalid: {sorted(member_names)}."
                )
                return
            if sum(member.file_size for member in members) > maximum_uncompressed_bytes:
                errors.append(
                    f"B-2 remap NPZ for {camera} exceeds its declared canvas size."
                )
                return
            header_contracts = {
                "map_x.npy": {np.dtype(np.float32)},
                "map_y.npy": {np.dtype(np.float32)},
                "valid_mask.npy": {np.dtype(np.bool_), np.dtype(np.uint8)},
            }
            for member_name, allowed_dtypes in header_contracts.items():
                shape, dtype = _read_npy_header(zip_archive, member_name)
                if shape != expected_shape:
                    errors.append(
                        f"B-2 remap NPY header shape for {camera}/{member_name} "
                        f"must be {expected_shape}, got {shape}."
                    )
                    return
                if dtype not in allowed_dtypes:
                    errors.append(
                        f"B-2 remap NPY header dtype for {camera}/{member_name} is invalid: {dtype}."
                    )
                    return
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"B-2 remap NPZ for {camera} is unreadable or truncated: {exc}")
        return
    except Exception as exc:
        errors.append(f"B-2 remap NPY header for {camera} is invalid: {exc}")
        return
    try:
        with path.open("rb") as stream:
            with np.load(stream, allow_pickle=False) as archive:
                expected_keys = {"map_x", "map_y", "valid_mask"}
                actual_keys = set(archive.files)
                if actual_keys != expected_keys:
                    errors.append(
                        f"B-2 remap NPZ keys for {camera} must be {sorted(expected_keys)}, got {sorted(actual_keys)}."
                    )
                    return
                map_x = np.asarray(archive["map_x"])
                map_y = np.asarray(archive["map_y"])
                valid_mask = np.asarray(archive["valid_mask"])
    except Exception as exc:
        errors.append(f"B-2 remap NPZ for {camera} is unreadable or truncated: {exc}")
        return
    if map_x.dtype != np.float32 or map_y.dtype != np.float32:
        errors.append(f"B-2 remap map dtype for {camera} must be float32.")
    if valid_mask.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        errors.append(f"B-2 remap mask dtype for {camera} must be bool or uint8.")
    if (
        map_x.shape != expected_shape
        or map_y.shape != expected_shape
        or valid_mask.shape != expected_shape
    ):
        errors.append(
            f"B-2 remap shape for {camera} must equal canvas shape {expected_shape}."
        )
    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        errors.append(f"B-2 remap maps for {camera} must contain only finite values.")


def _read_npy_header(
    archive: zipfile.ZipFile,
    member_name: str,
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    with archive.open(member_name, "r") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=4096,
            )
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=4096,
            )
        else:
            raise ValueError(f"Unsupported NPY format version: {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _validate_fisheye_candidate(
    path: Path | None,
    errors: list[str],
) -> Any | None:
    if path is None:
        return None
    try:
        source = load_fisheye_intrinsics_source(path)
    except Exception as exc:
        errors.append(f"Fisheye intrinsics source is not runtime-safe: {exc}")
        return None
    for camera in RUNTIME_CAMERAS:
        intrinsics = source.cameras.get(camera)
        if intrinsics is None:
            errors.append(f"Fisheye intrinsics missing runtime camera {camera}.")
            continue
        if not np.isfinite(intrinsics.camera_matrix).all() or not np.isfinite(
            intrinsics.distortion
        ).all():
            errors.append(f"Fisheye intrinsics for {camera} must be finite.")
        if intrinsics.image_size[0] <= 0 or intrinsics.image_size[1] <= 0:
            errors.append(f"Fisheye resolution for {camera} must be positive.")
    return source


def _validate_runtime_defaults(
    package_root: Path,
    defaults: dict[str, Any],
    candidate_paths: dict[str, Path],
    far_runtime: Any | None,
    near_runtime: Any | None,
    b2_runtime: Any | None,
    errors: list[str],
) -> None:
    mode = defaults.get("default_stitch_mode")
    if type(mode) is not str or mode not in {"far_field", "near_field", "auto"}:
        errors.append(
            "runtime_defaults.default_stitch_mode must be far_field, near_field, or auto."
        )
    custom = defaults.get("use_far_field_custom")
    if type(custom) is not bool:
        errors.append("runtime_defaults.use_far_field_custom must be a boolean.")
        custom = False
    projection = defaults.get("near_field_projection_source")
    if type(projection) is not str or projection not in {
        "current_perspective",
        "fisheye_rectilinear_candidate",
    }:
        errors.append(
            "runtime_defaults.near_field_projection_source is unsupported."
        )
    strategy = defaults.get("live_stitch_strategy", "template")
    if type(strategy) is not str or strategy not in {"template", "candidate"}:
        errors.append(
            "runtime_defaults.live_stitch_strategy must be template or candidate."
        )

    resolved_defaults: dict[str, Path | None] = {}
    references = {
        "far_field_layout_candidate": "far_field_layout",
        "near_field_layout_candidate": "near_field_layout",
        "b2_candidate": "b2_candidate",
        "fisheye_intrinsics_source": "fisheye_intrinsics",
    }
    for default_key, candidate_key in references.items():
        raw = defaults.get(default_key)
        if raw is None and default_key == "b2_candidate":
            resolved_defaults[default_key] = candidate_paths.get(candidate_key)
            continue
        if raw is None:
            resolved_defaults[default_key] = None
            continue
        if not isinstance(raw, str):
            errors.append(f"runtime_defaults.{default_key} must be a relative path or null.")
            resolved_defaults[default_key] = None
            continue
        resolved = _validate_relative_file(
            package_root,
            raw,
            f"runtime_defaults.{default_key}",
            errors,
        )
        resolved_defaults[default_key] = resolved
        declared = candidate_paths.get(candidate_key)
        if resolved is not None and (
            declared is None or resolved.resolve() != declared.resolve()
        ):
            errors.append(
                f"runtime_defaults.{default_key} does not match candidates.{candidate_key}.path."
            )

    far_path = resolved_defaults.get("far_field_layout_candidate")
    near_path = resolved_defaults.get("near_field_layout_candidate")
    b2_path = resolved_defaults.get("b2_candidate")
    fisheye_path = resolved_defaults.get("fisheye_intrinsics_source")
    if mode == "near_field" and near_path is None:
        errors.append("Near-field default requires a Near-field layout candidate.")
    if custom:
        if far_path is None:
            errors.append("Far-field Custom requires a Far-field layout candidate.")
        if b2_path is None or b2_runtime is None:
            errors.append("Far-field Custom requires a valid B-2 candidate.")
        if far_runtime is None or getattr(
            far_runtime, "projection_source", None
        ) != "b2_far_field_candidate":
            errors.append("Far-field Custom must use B-2 per-camera projection.")
    if near_runtime is not None and type(projection) is str and (
        near_runtime.projection.source.value != projection
    ):
        errors.append(
            "runtime_defaults Near-field projection source does not match its candidate."
        )
    if projection == "fisheye_rectilinear_candidate" and fisheye_path is None:
        errors.append("Fisheye projection requires a fisheye intrinsics source.")
    if strategy == "candidate" and (
        mode != "far_field" or custom or b2_path is None
    ):
        errors.append(
            "B-2 candidate strategy is only valid for Far-field Default with a B-2 candidate."
        )


def _validate_relative_file(
    package_root: Path,
    relative: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(relative, str):
        errors.append(f"{label} must be a relative path string.")
        return None
    try:
        path = resolve_project_path(package_root, relative)
    except ProjectPathError as exc:
        errors.append(f"{label}: {exc}")
        return None
    if not path.is_file():
        errors.append(f"{label} missing or not a regular file: {relative}")
        return None
    return path


def _package_file_is_safe(
    package_root: Path,
    path: Path,
    label: str,
    errors: list[str],
) -> bool:
    try:
        resolved = path.resolve()
        resolved.relative_to(package_root.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"{label} escapes package root: {exc}")
        return False
    if not path.is_file():
        errors.append(f"{label} is missing or not a regular file.")
        return False
    return True


def _resolve_embedded_path(
    package_root: Path,
    base_directory: Path,
    raw: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        errors.append(f"{label} must be a relative path string.")
        return None
    if is_absolute_path_string(raw):
        errors.append(f"{label} must be relative, not absolute: {raw}")
        return None
    try:
        base_relative = make_relative_to_package(package_root, base_directory)
        return resolve_project_path(package_root, f"{base_relative}/{raw}")
    except ProjectPathError as exc:
        errors.append(f"{label}: {exc}")
        return None


def _manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / MANIFEST_NAME if candidate.is_dir() else candidate


def _safe_project_name(project_name: str) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(project_name)
    ).strip("_")
    return safe_name or "DeepShark_Project"


def _unique_package_root(output_root: Path, project_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _safe_project_name(project_name)
    while True:
        root = output_root / (
            f"DeepShark_Project_{safe_name}_{timestamp}_{uuid4().hex[:8]}"
        )
        if not root.exists():
            return root


def _infer_b2_candidate_path(
    far_field_candidate_path: str | Path | None,
) -> Path | None:
    if far_field_candidate_path is None:
        return None
    candidate = candidate_yaml_path(far_field_candidate_path)
    if not candidate.is_file():
        return None
    data = load_yaml(candidate)
    projection = data.get("projection") if isinstance(data, dict) else None
    if not isinstance(projection, dict):
        return None
    raw = projection.get("candidate_directory") or projection.get(
        "b2_candidate_directory"
    )
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (candidate.parent / path).resolve()


def _infer_fisheye_intrinsics_path(
    near_field_candidate_path: str | Path | None,
) -> Path | None:
    if near_field_candidate_path is None:
        return None
    candidate = candidate_yaml_path(near_field_candidate_path)
    if not candidate.is_file():
        return None
    data = load_yaml(candidate)
    projection = data.get("projection") if isinstance(data, dict) else None
    if not isinstance(projection, dict):
        return None
    raw = projection.get("intrinsics_source_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (candidate.parent / path).resolve()


def _rewrite_far_field_candidate(
    candidate_path: Path,
    b2_relative_directory: str,
) -> None:
    data = _load_yaml_mapping_or_raise(candidate_path, "Far-field candidate")
    projection = data.get("projection")
    if isinstance(projection, dict) and str(projection.get("source")) == (
        "b2_far_field_candidate"
    ):
        projection["candidate_directory"] = b2_relative_directory
        projection.pop("b2_candidate_directory", None)
        save_yaml(candidate_path, data)


def _rewrite_near_field_candidate(
    candidate_path: Path,
    fisheye_relative_path: str,
) -> None:
    data = _load_yaml_mapping_or_raise(candidate_path, "Near-field candidate")
    projection = data.get("projection")
    if isinstance(projection, dict) and str(projection.get("source")) in {
        "fisheye_rectilinear",
        "fisheye_rectilinear_candidate",
    }:
        projection["intrinsics_source_path"] = fisheye_relative_path
        save_yaml(candidate_path, data)


def _candidate_path_or_none(
    candidates: dict[str, dict[str, Any]],
    key: str,
) -> str | None:
    value = candidates.get(key)
    path = value.get("path") if isinstance(value, dict) else None
    return path if isinstance(path, str) else None


def _backup_config_directory(config_dir: Path, backup_root: Path) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=backup_root, prefix=".backup-stage-"))
    published = False
    try:
        present: dict[str, str | None] = {}
        for name in ACTIVE_FILES:
            source = config_dir / name
            if source.is_file():
                shutil.copy2(source, staging / name)
                present[name] = file_revision(source)
            else:
                present[name] = None
        save_yaml(
            staging / "backup_manifest.yaml",
            {
                "format": "DeepSharkProjectActivationBackup",
                "schema_version": 1,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "files": present,
            },
        )
        destination = backup_root / (
            f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
        os.replace(staging, destination)
        published = True
        return destination
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _rollback_active_files(
    config_dir: Path,
    originals: dict[str, bytes | None],
) -> list[str]:
    errors: list[str] = []
    for name in reversed(ACTIVE_FILES):
        target = config_dir / name
        payload = originals.get(name)
        try:
            if payload is None:
                target.unlink(missing_ok=True)
            else:
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=config_dir,
                        prefix=f".{name}.rollback-",
                        suffix=".tmp",
                        delete=False,
                    ) as stream:
                        temporary = Path(stream.name)
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                    temporary = None
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return errors


def _load_yaml_mapping_or_raise(path: Path, label: str) -> dict[str, Any]:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{label} YAML root must be a mapping.")
    return data


def _readme_text(project_name: str, manifest: dict[str, Any]) -> str:
    candidates = manifest.get("candidates", {})
    candidate_lines = "\n".join(
        f"- {key}: {value.get('path')}"
        for key, value in candidates.items()
        if isinstance(value, dict)
    )
    if not candidate_lines:
        candidate_lines = "- 未包含候选文件。"
    return (
        f"# {project_name}\n\n"
        f"导出时间：{manifest.get('project', {}).get('created_at', '')}\n\n"
        "## 导入方式\n\n"
        "在 DeepSharkViewStudio 的项目管理中选择导入项目包，选择本目录的 "
        "project.dcsvs.yaml。先执行只读校验，确认后再激活。\n\n"
        "## 包含内容\n\n"
        "- configs/calibration.yaml\n"
        "- configs/cameras.yaml\n"
        "- configs/network.yaml\n"
        f"{candidate_lines}\n\n"
        "所有清单和候选依赖均使用包内相对路径。Validate 默认只读；激活前会备份当前配置。\n"
    )
