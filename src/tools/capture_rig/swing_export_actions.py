"""Responsive export control for the native swing editor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QProgressDialog, QPushButton, QWidget

from .swing_export import export_swing


class SwingExportWorker(QThread):
    progress = pyqtSignal(int, int)

    def __init__(self, root: Path, view: str, out: Path, parent: QObject) -> None:
        super().__init__(parent)
        self.root, self.view, self.out = root, view, out
        self.message = ""

    def run(self) -> None:
        try:
            export_swing(
                self.root,
                self.view,
                self.out,
                cancelled=self.isInterruptionRequested,
                progress=self.progress.emit,
            )
            self.message = f"Swing video and provenance saved: {self.out}"
        except (OSError, ValueError, cv2.error) as exc:
            self.message = f"Export not completed: {exc}"


class SwingExportActions(QObject):
    """Save first, export in a worker, and finish cancellation before closing."""

    def __init__(
        self,
        parent: QWidget,
        root: Path,
        *,
        view: Callable[[], str],
        save: Callable[[], bool],
        status: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.widget, self.root = parent, root
        self._view, self._save, self._status = view, save, status
        self._worker: SwingExportWorker | None = None
        self._progress: QProgressDialog | None = None
        self._closing = False
        self.button = QPushButton("Export swing…")
        self.button.setToolTip(
            "Save selection, then export its frames and crop to a new video. Originals stay intact."
        )
        self.button.clicked.connect(self.choose_output)

    @property
    def busy(self) -> bool:
        return self._worker is not None

    def choose_output(self) -> None:
        name, _ = QFileDialog.getSaveFileName(
            self.widget,
            "Export selected swing",
            str(self.root / "swing.mp4"),
            "MP4 video (*.mp4);;AVI video (*.avi)",
        )
        if name:
            self.start(Path(name))

    def start(self, out: Path) -> None:
        if self.busy or not self._save():
            return
        self.button.setEnabled(False)
        self._closing = False
        self._progress = QProgressDialog(
            "Preparing source provenance…", "Cancel", 0, 0, self.widget
        )
        self._progress.setWindowTitle("Export swing")
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.canceled.connect(self.cancel)
        self._worker = SwingExportWorker(self.root, self._view(), out, self)
        self._worker.progress.connect(self._update)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _update(self, done: int, total: int) -> None:
        if self._progress:
            self._progress.setLabelText(f"Writing swing frame {done} of {total}")
            self._progress.setRange(0, total)
            self._progress.setValue(done)

    def cancel(self) -> None:
        if self._worker:
            self._worker.requestInterruption()

    def can_close(self) -> bool:
        if not self.busy:
            return True
        self._closing = True
        self.cancel()
        self._status("Cancelling export before closing…")
        return False

    def _finished(self) -> None:
        assert self._worker is not None
        self._status(self._worker.message)
        self._worker.deleteLater()
        self._worker = None
        if self._progress:
            self._progress.close()
            self._progress.deleteLater()
            self._progress = None
        self.button.setEnabled(True)
        if self._closing:
            self.widget.close()
