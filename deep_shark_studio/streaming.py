"""Streaming placeholders for future QGC integration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamEndpoint:
    """Describes a future output stream target."""

    protocol: str
    host: str
    port: int
    path: str = ""

    def url(self) -> str:
        suffix = f"/{self.path.lstrip('/')}" if self.path else ""
        return f"{self.protocol}://{self.host}:{self.port}{suffix}"
