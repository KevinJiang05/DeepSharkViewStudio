"""Configuration loading helpers for DeepShark View Studio."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigConflictError(RuntimeError):
    """Raised when a stale window tries to overwrite a newer config file."""


def file_revision(path: str | Path) -> str | None:
    """Return a content-based revision suitable for optimistic write checks."""
    config_path = Path(path)
    if not config_path.exists():
        return None
    return sha256(config_path.read_bytes()).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for blank files."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def save_yaml(
    path: str | Path,
    data: dict[str, Any],
    expected_revision: str | None = None,
) -> str:
    """Atomically save YAML, optionally rejecting stale-writer updates."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_revision is not None:
        current_revision = file_revision(config_path)
        if current_revision != expected_revision:
            raise ConfigConflictError(
                f"{config_path.name} changed on disk; reload before saving."
            )

    serialized = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return file_revision(config_path) or ""


def load_config(name: str) -> dict[str, Any]:
    """Load a named config from the project configs directory."""
    return load_yaml(CONFIG_DIR / name)


def config_revision(name: str) -> str | None:
    """Return the current revision of a named project config."""
    return file_revision(CONFIG_DIR / name)


def save_config(
    name: str,
    data: dict[str, Any],
    expected_revision: str | None = None,
) -> str:
    """Save a named config into the project configs directory."""
    return save_yaml(
        CONFIG_DIR / name,
        data,
        expected_revision=expected_revision,
    )
