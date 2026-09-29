"""Verify and decode an archived frame while Qt continues processing input."""

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .frames import load_frame


class FrameLoader(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reference-frame"
        )
        self._future: Future[npt.NDArray[np.uint8]] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        parent.destroyed.connect(self.close)

    def load(self, root: Path, path: str, digest: str, capture_id: str) -> None:
        if self._future is not None:
            raise ValueError("A reference frame is already loading")
        self._future = self._executor.submit(
            load_frame, root, path, digest, capture_id=capture_id
        )
        self._timer.start()

    def _poll(self) -> None:
        future = self._future
        if future is None or not future.done():
            return
        self._timer.stop()
        self._future = None
        try:
            frame = future.result()
        except (ValueError, OSError) as exc:
            self.failed.emit(f"Cannot open reference frame: {exc}")
        else:
            self.ready.emit(frame)

    def close(self) -> None:
        self._timer.stop()
        self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)
