"""Bounded, latest-request video previews with decoder ownership off the UI thread."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from ..player import VideoReader
from .frames import MAX_PIXELS

PREVIEW_EDGE = 1280


@dataclass(frozen=True)
class PreviewFrame:
    index: int
    image: npt.NDArray[np.uint8]
    frame_count: int
    fps: float


class VideoPreview(QObject):
    """Keep one active decode and only the newest pending seek."""

    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: Path, parent: QObject) -> None:
        super().__init__(parent)
        self._path = path
        self._reader: VideoReader | None = None
        self._future: Future[PreviewFrame] | None = None
        self._pending: int | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reference-preview"
        )
        self._timer = QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._poll)
        parent.destroyed.connect(self.close)

    def request(self, index: int) -> None:
        if self._closed:
            raise ValueError("Video preview is closed")
        if index < 0:
            raise ValueError("Choose a nonnegative frame number")
        if self._future is not None:
            self._pending = index
            return
        self._future = self._executor.submit(self._read, index)
        self._timer.start()

    def _read(self, index: int) -> PreviewFrame:
        if self._reader is None:
            self._reader = VideoReader(self._path)
        reader = self._reader
        if reader.frame_count <= 0 or not isfinite(reader.fps) or reader.fps <= 0:
            raise ValueError("The recording needs a known frame count and frame rate")
        if (
            reader.width <= 0
            or reader.height <= 0
            or reader.width * reader.height > MAX_PIXELS
        ):
            raise ValueError("The recording exceeds the supported reference image size")
        if index >= reader.frame_count:
            raise ValueError(f"Choose a frame from 0 to {reader.frame_count - 1}")
        frame = reader.read(index)
        if frame is None:
            raise ValueError(
                "This frame cannot be decoded; choose another frame or restore the recording"
            )
        scale = PREVIEW_EDGE / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        return PreviewFrame(index, frame, reader.frame_count, reader.fps)

    def _poll(self) -> None:
        future = self._future
        if future is None or not future.done():
            return
        self._timer.stop()
        self._future = None
        pending, self._pending = self._pending, None
        try:
            frame = future.result()
        except (ValueError, OSError, cv2.error) as exc:
            self.failed.emit(f"Cannot preview this recording: {exc}")
        else:
            self.ready.emit(frame)
        if pending is not None and not self._closed:
            self.request(pending)

    def _release(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._pending = None
        # Release on the decoder's thread, after an in-flight read. Never join Qt.
        self._executor.submit(self._release)
        self._executor.shutdown(wait=False, cancel_futures=False)
