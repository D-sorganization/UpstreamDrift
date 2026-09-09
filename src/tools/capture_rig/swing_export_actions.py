"""Responsive export control for the native swing editor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QProgressDialog, QPushButton, QWidget

from .swing_export import export_swing
from src.motion_capture.coaching import DrawingLayer

ExportJob = Callable[[Path, Callable[[], bool], Callable[[int, int], None]], None]


@dataclass(frozen=True)
class ExportJobSpec:
    """Snapshot factory and file-dialog presentation for an alternate video export."""

    factory: Callable[[], ExportJob]
    title: str
    filename: str


class SwingExportWorker(QThread):
    progress = pyqtSignal(int, int)

    def __init__(
        self,
        root: Path,
        view: str,
        out: Path,
        parent: QObject,
        drawings: DrawingLayer | None = None,
        job: ExportJob | None = None,
    ) -> None:
        super().__init__(parent)
        self.root, self.view, self.out = root, view, out
        self.message = ""
        self.drawings = drawings
        self.job = job

    def run(self) -> None:
        try:
            if self.job:
                self.job(self.out, self.isInterruptionRequested, self.progress.emit)
            else:
                export_swing(
                    self.root,
                    self.view,
                    self.out,
                    cancelled=self.isInterruptionRequested,
                    progress=self.progress.emit,
                    drawings=self.drawings,
                )
            self.message = f"Video and provenance saved: {self.out}"
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
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
        drawings: Callable[[], DrawingLayer | None] = lambda: None,
        label: str = "Export swing…",
        job: ExportJobSpec | None = None,
    ) -> None:
        super().__init__(parent)
        self.widget, self.root = parent, root
        self._view, self._save, self._status = view, save, status
        self._drawings = drawings
        self._job = job.factory if job else None
        self._title = job.title if job else "Export swing"
        self._filename = job.filename if job else "swing.mp4"
        self._worker: SwingExportWorker | None = None
        self._progress: QProgressDialog | None = None
        self._closing = False
        self.button = QPushButton(label)
        self.button.setToolTip(
            "Save changes, then export the saved frame selection and crop to a new video. Originals stay intact."
        )
        self.button.clicked.connect(self.choose_output)

    @property
    def busy(self) -> bool:
        return self._worker is not None

    def choose_output(self) -> None:
        name, _ = QFileDialog.getSaveFileName(
            self.widget,
            self._title,
            str(self.root / self._filename),
            "MP4 video (*.mp4);;AVI video (*.avi)",
        )
        if name:
            self.start(Path(name))

    def start(self, out: Path) -> None:
        if self.busy or not self._save():
            return
        try:
            job = self._job() if self._job else None
        except (ValueError, OSError) as exc:
            self._status(f"Export not started: {exc}")
            return
        self.button.setEnabled(False)
        self._closing = False
        self._progress = QProgressDialog(
            "Preparing source provenance…", "Cancel", 0, 0, self.widget
        )
        self._progress.setWindowTitle(self._title)
        self._progress.setMinimumDuration(0)
        self._progress.setAutoClose(False)
        self._progress.canceled.connect(self.cancel)
        self._worker = SwingExportWorker(
            self.root,
            self._view(),
            out,
            self,
            self._drawings(),
            job,
        )
        self._worker.progress.connect(self._update)
        self._worker.finished.connect(self._finished)
        self._worker.start()

    def _update(self, done: int, total: int) -> None:
        if self._progress:
            self._progress.setLabelText(f"Writing video frame {done} of {total}")
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
