"""Camera-source abstractions reserved for live video work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoSource:
    name: str
    uri: str | int

    def frames(self) -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(self.uri)
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()
