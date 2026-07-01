"""Application logging with bounded files and connection-secret redaction."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re


_RTSP_URL = re.compile(r"rtsp://[^\s\"']+", re.IGNORECASE)
_PASSWORD_PARAMETER = re.compile(
    r"(?i)(password|passwd|pwd)=([^&\s\"']*)"
)


def redact_log_message(message: object) -> str:
    """Remove connection details before text reaches the UI or log file."""
    text = str(message)
    text = _RTSP_URL.sub("rtsp://<redacted>", text)
    return _PASSWORD_PARAMETER.sub(r"\1=<redacted>", text)


def create_application_logger(log_path: str | Path) -> logging.Logger:
    """Return one process-wide rotating logger for the requested file."""
    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"deep_shark_studio:{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(
            path,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger
