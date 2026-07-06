"""Safe package-relative path helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_WINDOWS_ABSOLUTE_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WINDOWS_UNC_RE = re.compile(r"^[\\/]{2}[^\\/]+[\\/][^\\/]+")


class ProjectPathError(ValueError):
    """Raised when a package-relative path is unsafe."""


def is_absolute_path_string(value: str) -> bool:
    text = str(value)
    return (
        Path(text).is_absolute()
        or bool(_WINDOWS_ABSOLUTE_RE.match(text))
        or bool(_WINDOWS_UNC_RE.match(text))
    )


def contains_absolute_path_string(value: Any) -> bool:
    return bool(find_absolute_path_strings(value))


def find_absolute_path_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(find_absolute_path_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(find_absolute_path_strings(item))
    elif isinstance(value, str) and is_absolute_path_string(value):
        found.append(value)
    return found


def resolve_project_path(package_root: str | Path, relative_path: str) -> Path:
    """Resolve a manifest path while rejecting absolute paths and path escape."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ProjectPathError("Package path must be a non-empty relative string.")
    if is_absolute_path_string(relative_path):
        raise ProjectPathError(f"Package path must be relative: {relative_path}")
    root = Path(package_root).resolve()
    target = (root / Path(relative_path.replace("/", "\\"))).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProjectPathError(f"Package path escapes package root: {relative_path}") from exc
    return target


def make_relative_to_package(package_root: str | Path, path: str | Path) -> str:
    """Return a stable POSIX-style relative path inside a package."""
    root = Path(package_root).resolve()
    target = Path(path).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ProjectPathError(f"Path is outside package root: {target}") from exc
    return relative.as_posix()
