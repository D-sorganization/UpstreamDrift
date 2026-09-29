"""Frame-accurate video access for playback and scrubbing.

OpenCV decodes both the MJPEG recordings and the H.264 proxies; Qt's media
stack is not used because scrubbing to an exact frame and drawing a pose on
it needs the pixels, not a video surface. Sequential reads are cheap; a
backwards seek re-positions the decoder.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require


class VideoReader:
    """Random-access frames of one file. Use as a context manager."""

    def __init__(self, path: Path) -> None:
        import cv2

        require(path.is_file(), "video must exist", str(path))
        self._cap: Any = cv2.VideoCapture(str(path))
        require(self._cap.isOpened(), "could not open video", str(path))
        self._path = path
        self._next = 0
        self._cached_index: int | None = None
        self._cached_frame: npt.NDArray[np.uint8] | None = None
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def path(self) -> Path:
        return self._path

    def read(self, index: int) -> npt.NDArray[np.uint8] | None:
        """BGR frame ``index`` or ``None`` past the end. Precondition: index >= 0."""
        import cv2

        require(index >= 0, "frame index must be >= 0", index)
        if index == self._cached_index and self._cached_frame is not None:
            return self._cached_frame.copy()
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._next = index + 1
        # Raw and overlay tiles request the same reader/frame. Retain only the
        # most recent decode, and isolate every caller's mutable drawing pixels.
        self._cached_index = index
        self._cached_frame = np.asarray(frame, dtype=np.uint8)
        return self._cached_frame.copy()

    def close(self) -> None:
        self._cached_index = None
        self._cached_frame = None
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def clamp_index(index: int, frame_count: int) -> int:
    """``index`` held inside ``[0, frame_count - 1]`` (0 for an empty file)."""
    if frame_count <= 0:
        return 0
    return min(max(index, 0), frame_count - 1)
