"""Portable project package export, validation, and activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

from deep_shark_studio.config import CONFIG_DIR, PROJECT_ROOT, load_yaml, save_yaml
from deep_shark_studio.project import backup_configs
from deep_shark_studio.projection.intrinsics_runtime_loader import (
    FisheyeIntrinsicsRuntimeError,
    load_fisheye_intrinsics_source,
)

from .dependency_collector import (
    candidate_yaml_path,
    collect_b2_candidate_dependencies,
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
    activated_configs: tuple[Path, ...]
    warnings: tuple[str, ...]


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
    use_far_field_custom: bool = False,
    near_field_projection_source: str = "current_perspective",
    include_preview_assets: bool = True,
    include_debug_assets: bool = False,
) -> ProjectPackageExportResult:
    """Export a directory package with relative manifest paths."""
    output_directory = Path(output_root)
    package_root = _unique_package_root(output_directory, project_name)
    package_root.mkdir(parents=True, exist_ok=False)
    warnings: list[str] = []
    copied_files: list[Path] = []

    config_sources = {
        "calibration": Path(calibration_config_path or CONFIG_DIR / "calibration.yaml"),
        "cameras": Path(cameras_config_path or CONFIG_DIR / "cameras.yaml"),
        "network": Path(network_config_path or CONFIG_DIR / "network.yaml"),
    }
    config_manifest: dict[str, str] = {}
    for key, source in config_sources.items():
        destination = package_root / "configs" / source.name
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_files.append(destination)
            config_manifest[key] = make_relative_to_package(package_root, destination)
        else:
            warnings.append(f"Missing config: {source}")

    inferred_b2 = _infer_b2_candidate_path(far_field_candidate_path)
    if b2_candidate_path is None and inferred_b2 is not None:
        b2_candidate_path = inferred_b2
    inferred_fisheye = _infer_fisheye_intrinsics_path(near_field_candidate_path)
    if fisheye_intrinsics_path is None and inferred_fisheye is not None:
        fisheye_intrinsics_path = inferred_fisheye

    candidate_manifest: dict[str, dict[str, Any]] = {}
    known_rewrites: dict[str, str] = {}

    b2_dest: Path | None = None
    if b2_candidate_path is not None:
        b2_source = candidate_yaml_path(b2_candidate_path)
        if b2_source.exists():
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
        else:
            warnings.append(f"Missing B-2 candidate: {b2_source}")

    fisheye_dest: Path | None = None
    if fisheye_intrinsics_path is not None:
        fisheye_source = candidate_yaml_path(fisheye_intrinsics_path)
        if fisheye_source.exists():
            fisheye_dest = package_root / "candidates" / "fisheye_intrinsics"
            copied_files.extend(
                copy_candidate_directory(
                    fisheye_source,
                    fisheye_dest,
                    include_preview_assets=include_preview_assets,
                    include_debug_assets=include_debug_assets,
                )
            )
            known_rewrites[str(fisheye_source.parent.resolve())] = "candidates/fisheye_intrinsics"
            known_rewrites[str(fisheye_source.resolve())] = "candidates/fisheye_intrinsics/candidate.yaml"
            candidate_manifest["fisheye_intrinsics"] = {
                "path": "candidates/fisheye_intrinsics/candidate.yaml",
                "source_type": "opencv_fisheye_intrinsics",
            }
        else:
            warnings.append(f"Missing fisheye intrinsics source: {fisheye_source}")

    if far_field_candidate_path is not None:
        source = candidate_yaml_path(far_field_candidate_path)
        if source.exists():
            dest = package_root / "candidates" / "far_field_layout"
            copied_files.extend(
                copy_candidate_directory(
                    source,
                    dest,
                    include_preview_assets=include_preview_assets,
                    include_debug_assets=include_debug_assets,
                )
            )
            if b2_dest is not None:
                known_rewrites[str(source.parent.resolve())] = "candidates/far_field_layout"
                _rewrite_far_field_candidate(dest / "candidate.yaml", "../b2_candidate")
            candidate_manifest["far_field_layout"] = {
                "path": "candidates/far_field_layout/candidate.yaml",
                "candidate_type": "far_field_layout",
            }
        else:
            warnings.append(f"Missing Far-field Layout Candidate: {source}")

    if near_field_candidate_path is not None:
        source = candidate_yaml_path(near_field_candidate_path)
        if source.exists():
            dest = package_root / "candidates" / "near_field_layout"
            copied_files.extend(
                copy_candidate_directory(
                    source,
                    dest,
                    include_preview_assets=include_preview_assets,
                    include_debug_assets=include_debug_assets,
                )
            )
            if fisheye_dest is not None:
                known_rewrites[str(source.parent.resolve())] = "candidates/near_field_layout"
                _rewrite_near_field_candidate(dest / "candidate.yaml", "../fisheye_intrinsics/candidate.yaml")
            candidate_manifest["near_field_layout"] = {
                "path": "candidates/near_field_layout/candidate.yaml",
                "candidate_type": "front_priority_layout",
            }
        else:
            warnings.append(f"Missing Near-field Layout Candidate: {source}")

    for yaml_file in package_root.rglob("*.yaml"):
        rewrite_yaml_file_paths(yaml_file, known_rewrites, warnings)
    for yaml_file in package_root.rglob("*.yml"):
        rewrite_yaml_file_paths(yaml_file, known_rewrites, warnings)

    runtime_defaults = {
        "default_stitch_mode": default_stitch_mode,
        "use_far_field_custom": bool(use_far_field_custom),
        "far_field_layout_candidate": _candidate_path_or_none(candidate_manifest, "far_field_layout"),
        "near_field_layout_candidate": _candidate_path_or_none(candidate_manifest, "near_field_layout"),
        "near_field_projection_source": near_field_projection_source,
        "fisheye_intrinsics_source": _candidate_path_or_none(candidate_manifest, "fisheye_intrinsics"),
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
    readme_path = package_root / "README_使用说明.md"
    readme_path.write_text(_readme_text(project_name, manifest), encoding="utf-8")
    copied_files.append(readme_path)
    report = {
        "schema_version": 1,
        "package_root": ".",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "warnings": warnings,
        "copied_files": [
            make_relative_to_package(package_root, path)
            for path in sorted(set(copied_files))
            if path.exists()
        ],
    }
    report_path = package_root / "export_report.yaml"
    save_yaml(report_path, report)
    rewrite_yaml_file_paths(report_path, known_rewrites, warnings)
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
    write_report: bool = True,
) -> ProjectPackageValidationResult:
    manifest_path = _manifest_path(path)
    package_root = manifest_path.parent
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest_path.exists():
        return ProjectPackageValidationResult(
            package_root=package_root,
            manifest_path=manifest_path,
            valid=False,
            errors=(f"Manifest not found: {manifest_path}",),
            warnings=(),
        )
    try:
        manifest = load_yaml(manifest_path)
    except Exception as exc:
        return ProjectPackageValidationResult(
            package_root=package_root,
            manifest_path=manifest_path,
            valid=False,
            errors=(f"Manifest is not readable YAML: {exc}",),
            warnings=(),
        )

    if manifest.get("schema_version") != PROJECT_PACKAGE_SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version: {manifest.get('schema_version')}")
    if manifest.get("package_type") != PROJECT_PACKAGE_TYPE:
        errors.append(f"Unsupported package_type: {manifest.get('package_type')}")
    absolute_manifest_paths = find_absolute_path_strings(manifest)
    if absolute_manifest_paths:
        errors.append(f"Manifest contains absolute paths: {absolute_manifest_paths}")

    configs = manifest.get("configs")
    if not isinstance(configs, dict):
        errors.append("manifest.configs is required.")
    else:
        for key in ("calibration", "cameras", "network"):
            _validate_relative_file(package_root, configs.get(key), f"configs.{key}", errors)

    candidates = manifest.get("candidates") or {}
    if not isinstance(candidates, dict):
        errors.append("manifest.candidates must be a mapping.")
        candidates = {}
    _validate_candidate(package_root, candidates, "far_field_layout", "far_field_layout", errors, warnings)
    _validate_candidate(package_root, candidates, "near_field_layout", "front_priority_layout", errors, warnings)
    _validate_b2_candidate(package_root, candidates, errors, warnings)
    _validate_fisheye_candidate(package_root, candidates, errors, warnings)

    for yaml_file in package_root.rglob("*.yaml"):
        try:
            data = load_yaml(yaml_file)
        except Exception as exc:
            warnings.append(f"Could not read YAML {yaml_file}: {exc}")
            continue
        absolute = find_absolute_path_strings(data)
        if absolute:
            errors.append(f"YAML contains absolute paths: {yaml_file.relative_to(package_root).as_posix()}: {absolute}")

    report_path = package_root / "project_validation_report.yaml" if write_report else None
    result = ProjectPackageValidationResult(
        package_root=package_root,
        manifest_path=manifest_path,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        report_path=report_path,
    )
    if write_report and report_path is not None:
        save_yaml(
            report_path,
            {
                "schema_version": 1,
                "valid": result.valid,
                "errors": list(result.errors),
                "warnings": list(result.warnings),
                "manifest": make_relative_to_package(package_root, manifest_path),
            },
        )
    return result


def activate_project_package(
    path: str | Path,
    *,
    active_config_dir: str | Path | None = None,
    backup_root: str | Path | None = None,
) -> ProjectPackageActivationResult:
    """Activate package configs after caller confirmation."""
    validation = validate_project_package(path, write_report=True)
    if not validation.valid:
        raise ValueError("Project package is not valid: " + "; ".join(validation.errors))
    manifest = load_yaml(validation.manifest_path)
    config_dir = Path(active_config_dir or CONFIG_DIR)
    backup_path: Path | None = None
    if backup_root is None and active_config_dir is None:
        backup_path = backup_configs()
    elif backup_root is not None:
        backup_path = _backup_config_directory(config_dir, Path(backup_root))
    activated: list[Path] = []
    for key, filename in (
        ("calibration", "calibration.yaml"),
        ("cameras", "cameras.yaml"),
        ("network", "network.yaml"),
    ):
        relative = manifest["configs"][key]
        source = resolve_project_path(validation.package_root, relative)
        destination = config_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        activated.append(destination)
    return ProjectPackageActivationResult(
        package_root=validation.package_root,
        manifest_path=validation.manifest_path,
        backup_path=backup_path,
        activated_configs=tuple(activated),
        warnings=validation.warnings,
    )


def _manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / MANIFEST_NAME
    return candidate


def _unique_package_root(output_root: Path, project_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in project_name).strip("_")
    if not safe_name:
        safe_name = "DeepShark_Project"
    root = output_root / f"DeepShark_Project_{safe_name}_{timestamp}"
    suffix = 1
    while root.exists():
        root = output_root / f"DeepShark_Project_{safe_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    return root


def _infer_b2_candidate_path(far_field_candidate_path: str | Path | None) -> Path | None:
    if far_field_candidate_path is None:
        return None
    candidate = candidate_yaml_path(far_field_candidate_path)
    if not candidate.exists():
        return None
    data = load_yaml(candidate)
    projection = data.get("projection") if isinstance(data, dict) else None
    if not isinstance(projection, dict):
        return None
    raw = projection.get("candidate_directory") or projection.get("b2_candidate_directory")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (candidate.parent / path).resolve()


def _infer_fisheye_intrinsics_path(near_field_candidate_path: str | Path | None) -> Path | None:
    if near_field_candidate_path is None:
        return None
    candidate = candidate_yaml_path(near_field_candidate_path)
    if not candidate.exists():
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


def _rewrite_far_field_candidate(candidate_path: Path, b2_relative_directory: str) -> None:
    data = load_yaml(candidate_path)
    projection = data.get("projection")
    if isinstance(projection, dict) and str(projection.get("source")) == "b2_far_field_candidate":
        projection["candidate_directory"] = b2_relative_directory
        projection.pop("b2_candidate_directory", None)
        save_yaml(candidate_path, data)


def _rewrite_near_field_candidate(candidate_path: Path, fisheye_relative_path: str) -> None:
    data = load_yaml(candidate_path)
    projection = data.get("projection")
    if isinstance(projection, dict) and str(projection.get("source")) in {
        "fisheye_rectilinear",
        "fisheye_rectilinear_candidate",
    }:
        projection["intrinsics_source_path"] = fisheye_relative_path
        save_yaml(candidate_path, data)


def _candidate_path_or_none(candidates: dict[str, dict[str, Any]], key: str) -> str | None:
    value = candidates.get(key)
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str):
            return path
    return None


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
    if not path.exists():
        errors.append(f"{label} missing: {relative}")
        return None
    return path


def _validate_candidate(
    package_root: Path,
    candidates: dict[str, Any],
    key: str,
    expected_type: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    item = candidates.get(key)
    if item is None:
        warnings.append(f"Optional candidate missing: {key}")
        return
    if not isinstance(item, dict):
        errors.append(f"candidates.{key} must be a mapping.")
        return
    path = _validate_relative_file(package_root, item.get("path"), f"candidates.{key}.path", errors)
    if path is None:
        return
    data = load_yaml(path)
    if data.get("candidate_type") != expected_type:
        errors.append(f"{key} candidate_type must be {expected_type}.")


def _validate_b2_candidate(
    package_root: Path,
    candidates: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    item = candidates.get("b2_candidate")
    if item is None:
        warnings.append("Optional candidate missing: b2_candidate")
        return
    if not isinstance(item, dict):
        errors.append("candidates.b2_candidate must be a mapping.")
        return
    path = _validate_relative_file(package_root, item.get("path"), "candidates.b2_candidate.path", errors)
    if path is None:
        return
    for dependency in collect_b2_candidate_dependencies(path, include_preview_assets=False):
        if dependency.required and not dependency.source.exists():
            errors.append(f"B-2 runtime dependency missing: {dependency.relative_path}")


def _validate_fisheye_candidate(
    package_root: Path,
    candidates: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    item = candidates.get("fisheye_intrinsics")
    if item is None:
        warnings.append("Optional candidate missing: fisheye_intrinsics")
        return
    if not isinstance(item, dict):
        errors.append("candidates.fisheye_intrinsics must be a mapping.")
        return
    path = _validate_relative_file(package_root, item.get("path"), "candidates.fisheye_intrinsics.path", errors)
    if path is None:
        return
    try:
        load_fisheye_intrinsics_source(path)
    except FisheyeIntrinsicsRuntimeError as exc:
        errors.append(f"Fisheye intrinsics source is not usable: {exc}")


def _backup_config_directory(config_dir: Path, backup_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = backup_root / f"backup_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("calibration.yaml", "cameras.yaml", "network.yaml"):
        source = config_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    return destination


def _readme_text(project_name: str, manifest: dict[str, Any]) -> str:
    candidates = manifest.get("candidates", {})
    candidate_lines = "\n".join(f"- {key}: {value.get('path')}" for key, value in candidates.items())
    if not candidate_lines:
        candidate_lines = "- 未包含候选文件。"
    return (
        f"# {project_name}\n\n"
        f"导出时间：{manifest.get('project', {}).get('created_at', '')}\n\n"
        "## 如何导入\n\n"
        "在 DeepSharkViewStudio 的“项目管理”中选择“导入项目包”，选择本目录的 "
        "`project.dcsvs.yaml`，校验通过后再确认激活。\n\n"
        "## 包含内容\n\n"
        "- configs/calibration.yaml\n"
        "- configs/cameras.yaml\n"
        "- configs/network.yaml\n"
        f"{candidate_lines}\n\n"
        "## 运行模式说明\n\n"
        "- Far-field Default：原正式 runtime。\n"
        "- Far-field Custom：B-2 per-camera projection + 手动 layout + B-2 weight selection。\n"
        "- Near-field：front-priority layout candidate，可选 Current Perspective 或 Fisheye Rectilinear。\n\n"
        "本项目包使用相对路径。导入校验不会修改当前配置；只有用户确认激活后才会备份并替换当前 configs。\n"
    )
