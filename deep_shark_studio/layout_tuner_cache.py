"""Provenance key for frozen Layout Tuner projection inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LayoutTunerCacheKey:
    target: str
    projection_source: str
    calibration_revision: str
    profile_revision: str
    projection_revision: str
    balance: float
    fov_scale: float
    frame_signature: str

    def same_selection_as(self, other: "LayoutTunerCacheKey") -> bool:
        """Compare all projection inputs while retaining the frozen frame identity."""
        return (
            self.target == other.target
            and self.projection_source == other.projection_source
            and self.calibration_revision == other.calibration_revision
            and self.profile_revision == other.profile_revision
            and self.projection_revision == other.projection_revision
            and self.balance == other.balance
            and self.fov_scale == other.fov_scale
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame_content_signature(frames: Mapping[str, np.ndarray]) -> str:
    """Hash one frozen multi-camera frame set for candidate provenance."""
    digest = hashlib.sha256()
    for camera in sorted(frames):
        array = np.asarray(frames[camera])
        digest.update(str(camera).encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def mapping_revision(value: Mapping[str, Any]) -> str:
    """Return a stable revision for an in-memory profile/config mapping."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_source_revision(path: str | Path | None) -> str:
    """Hash the runtime-relevant YAML/NPZ files behind a candidate source."""
    if path is None:
        return "none"
    source = Path(path)
    if not source.exists():
        return "missing:" + str(source.resolve())
    files: list[Path]
    if source.is_file():
        files = [source]
    else:
        files = sorted(
            item
            for item in source.rglob("*")
            if item.is_file()
            and (item.name in {"candidate.yaml", "report.yaml"} or item.suffix == ".npz")
        )
    digest = hashlib.sha256()
    root = source.parent if source.is_file() else source
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported revision value: {type(value).__name__}")
