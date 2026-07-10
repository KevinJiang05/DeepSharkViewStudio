"""Candidate dependency collection for portable project packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from deep_shark_studio.config import load_yaml, save_yaml

from .path_resolver import ProjectPathError, is_absolute_path_string, resolve_project_path


@dataclass(frozen=True)
class PackageDependency:
    source: Path
    relative_path: str
    kind: str
    required: bool = True


def candidate_yaml_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file() or source.suffix.lower() in {".yaml", ".yml"}:
        return source
    return source / "candidate.yaml"


def collect_b2_candidate_dependencies(
    candidate_path: str | Path,
    include_preview_assets: bool = True,
) -> list[PackageDependency]:
    """Collect runtime-required B-2 candidate files.

    Runtime needs candidate.yaml, report.yaml, and virtual_panorama remap npz files.
    Preview artifacts are optional but useful for human review.
    """
    candidate_file = candidate_yaml_path(candidate_path)
    root = candidate_file.parent
    dependencies = [
        PackageDependency(candidate_file, "candidate.yaml", "b2_candidate"),
        PackageDependency(root / "report.yaml", "report.yaml", "b2_report", required=False),
    ]
    data = load_yaml(candidate_file) if candidate_file.exists() else {}
    files = ((data.get("virtual_panorama") or {}).get("files") or {})
    remaps = files.get("remaps") if isinstance(files, dict) else None
    if isinstance(remaps, dict):
        for camera, relative in remaps.items():
            if isinstance(relative, str):
                source, normalized = _safe_dependency_path(root, relative)
                dependencies.append(
                    PackageDependency(
                        source,
                        normalized,
                        f"b2_remap:{camera}",
                    )
                )
    if include_preview_assets and isinstance(files, dict):
        for relative in _iter_relative_strings(files):
            if not relative.startswith("intrinsics/"):
                source, normalized = _safe_dependency_path(root, relative)
                dependencies.append(
                    PackageDependency(source, normalized, "b2_preview", required=False)
                )
    return _dedupe_dependencies(dependencies)


def copy_candidate_directory(
    source: str | Path,
    destination: str | Path,
    include_preview_assets: bool = True,
    include_debug_assets: bool = False,
) -> list[Path]:
    """Copy a candidate directory subset and return copied files."""
    source_candidate = candidate_yaml_path(source)
    source_root = source_candidate.parent
    dest_root = Path(destination)
    copied: list[Path] = []
    for name in ("candidate.yaml", "report.yaml"):
        copied.extend(
            _copy_optional_file(
                source_root / name,
                dest_root / name,
                source_root=source_root,
            )
        )
    if include_preview_assets:
        copied.extend(
            _copy_optional_tree(
                source_root / "previews",
                dest_root / "previews",
                candidate_root=source_root,
            )
        )
        for name in ("preview.png", "preview_with_overlay.png"):
            copied.extend(
                _copy_optional_file(
                    source_root / name,
                    dest_root / name,
                    source_root=source_root,
                )
            )
    if include_debug_assets:
        copied.extend(
            _copy_optional_tree(
                source_root / "debug",
                dest_root / "debug",
                candidate_root=source_root,
            )
        )
    return copied


def copy_b2_candidate_directory(
    source: str | Path,
    destination: str | Path,
    include_preview_assets: bool = True,
) -> list[Path]:
    source_candidate = candidate_yaml_path(source)
    source_root = source_candidate.parent
    dest_root = Path(destination)
    copied: list[Path] = []
    for dependency in collect_b2_candidate_dependencies(
        source_candidate,
        include_preview_assets=include_preview_assets,
    ):
        if not dependency.source.exists():
            continue
        destination_path = resolve_project_path(dest_root, dependency.relative_path)
        copied.extend(
            _copy_optional_file(
                dependency.source,
                destination_path,
                source_root=source_root,
            )
        )
    for name in ("candidate.yaml", "report.yaml"):
        copied.extend(
            _copy_optional_file(
                source_root / name,
                dest_root / name,
                source_root=source_root,
            )
        )
    return _dedupe_paths(copied)


def rewrite_yaml_file_paths(
    yaml_path: str | Path,
    known_rewrites: dict[str, str],
    warnings: list[str],
) -> None:
    """Rewrite known absolute paths and scrub unknown absolute paths in a YAML copy."""
    path = Path(yaml_path)
    if not path.exists():
        return
    data = load_yaml(path)
    rewritten = _rewrite_value(data, known_rewrites, warnings, path)
    if rewritten != data:
        save_yaml(path, rewritten)


def _rewrite_value(
    value: Any,
    known_rewrites: dict[str, str],
    warnings: list[str],
    yaml_path: Path,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_value(item, known_rewrites, warnings, yaml_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_value(item, known_rewrites, warnings, yaml_path) for item in value]
    if isinstance(value, str):
        normalized = _normalize_path_string(value)
        for source, replacement in known_rewrites.items():
            source_norm = _normalize_path_string(source)
            if normalized == source_norm:
                return replacement
            if normalized.startswith(source_norm.rstrip("/") + "/"):
                suffix = normalized[len(source_norm.rstrip("/")) + 1 :]
                return f"{replacement.rstrip('/')}/{suffix}"
        if is_absolute_path_string(value):
            warnings.append(f"Removed absolute path from exported YAML {yaml_path.name}: {value}")
            return "<external_path_removed>"
    return value


def _iter_relative_strings(value: Any) -> list[str]:
    items: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            items.extend(_iter_relative_strings(item))
    elif isinstance(value, list):
        for item in value:
            items.extend(_iter_relative_strings(item))
    elif isinstance(value, str) and value and not is_absolute_path_string(value):
        items.append(value)
    return items


def _safe_dependency_path(root: Path, relative: str) -> tuple[Path, str]:
    """Resolve one dependency inside its candidate directory.

    Rejecting here is intentional: export must never even attempt a read from
    or a write to a traversal target.  Validation catches the same
    ProjectPathError and reports it as a package error.
    """
    if is_absolute_path_string(relative):
        raise ProjectPathError(f"Candidate dependency must be relative: {relative}")
    source = resolve_project_path(root, relative)
    normalized = source.relative_to(root.resolve()).as_posix()
    return source, normalized


def _copy_optional_file(
    source: Path,
    destination: Path,
    *,
    source_root: Path | None = None,
) -> list[Path]:
    if not source.exists() or not source.is_file():
        return []
    if source_root is not None:
        try:
            source.resolve().relative_to(source_root.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProjectPathError(
                f"Candidate source escapes candidate root: {source}"
            ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return [destination]


def _copy_optional_tree(
    source: Path,
    destination: Path,
    *,
    candidate_root: Path | None = None,
) -> list[Path]:
    if not source.exists() or not source.is_dir():
        return []
    copied: list[Path] = []
    source_root = (candidate_root or source).resolve()
    try:
        source.resolve().relative_to(source_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProjectPathError(
            f"Candidate asset tree escapes candidate root: {source}"
        ) from exc
    for child in source.rglob("*"):
        if child.is_file():
            relative = child.relative_to(source)
            copied.extend(
                _copy_optional_file(
                    child,
                    destination / relative,
                    source_root=source_root,
                )
            )
    return copied


def _dedupe_dependencies(items: list[PackageDependency]) -> list[PackageDependency]:
    seen: set[tuple[Path, str]] = set()
    result: list[PackageDependency] = []
    for item in items:
        key = (item.source.resolve(), item.relative_path)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_paths(items: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for item in items:
        key = item.resolve()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _normalize_path_string(value: str) -> str:
    return str(value).replace("\\", "/").rstrip("/")
