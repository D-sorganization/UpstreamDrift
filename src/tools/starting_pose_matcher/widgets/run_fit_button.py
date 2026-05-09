"""Run-fit button + QThread worker for the starting-pose matcher.

Slice 2/3 of issue #4707. Encapsulates a "Run fit" button, a Cancel
button, a status label, and a :class:`QThread` worker that calls
``provider_registry.get_provider(engine).fit_swing(target)`` off the
GUI thread. Slice 3 (save-fit JSON) consumes :pyattr:`RunFitButton.last_result`.

Design: DRY (delegates to ``provider_registry``), DbC (validates inputs
before spawning the thread), LoD (callers use the public signals/methods
only).
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.shared.python.motion_matching import provider_registry

__all__ = ["FitWorker", "RunFitButton"]


class FitWorker(QObject):
    """QObject worker that runs ``provider.fit_swing`` off the GUI thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, engine_name: str, target: Any) -> None:
        super().__init__()
        if not isinstance(engine_name, str) or not engine_name:
            raise ValueError("engine_name must be a non-empty string")
        if target is None:
            raise ValueError("target must not be None")
        self._engine_name = engine_name
        self._target = target
        self._cancelled = False

    def request_cancel(self) -> None:
        """Cooperative-cancel flag checked before/after the fit call."""
        self._cancelled = True

    def run(self) -> None:
        """Resolve the provider and run ``fit_swing(target)``."""
        if self._cancelled:
            self.cancelled.emit()
            return
        self.progress.emit(f"Resolving '{self._engine_name}' provider...")
        try:
            provider = provider_registry.get_provider(self._engine_name)
        except KeyError as exc:
            self.failed.emit(str(exc))
            return
        if self._cancelled:
            self.cancelled.emit()
            return
        self.progress.emit(f"Running fit_swing on '{self._engine_name}'...")
        try:
            result = provider.fit_swing(self._target)
        except Exception as exc:  # noqa: BLE001 — surface any engine error
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        if self._cancelled:
            self.cancelled.emit()
            return
        self.finished.emit(result)


class RunFitButton(QWidget):
    """Self-contained Run-fit + Cancel + status widget."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target: Any = None
        self._engine: str = ""
        self._thread: QThread | None = None
        self._worker: FitWorker | None = None
        self.last_result: Any = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run fit")
        self.btn_run.setObjectName("primary")
        self.btn_run.setEnabled(False)
        self.btn_run.setToolTip("Run fit_swing in a background thread.")
        self.btn_run.clicked.connect(self.start_fit)
        row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Cancel the running fit and clean up the thread.")
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)
        self.lbl_status = QLabel("Idle. Load a target and pick an engine.")
        self.lbl_status.setObjectName("status")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    def set_inputs(self, *, target: Any, engine_name: str) -> None:
        """Update the cached target/engine and refresh the run button."""
        self._target = target
        self._engine = engine_name or ""
        ready = self._target is not None and bool(self._engine)
        self.btn_run.setEnabled(ready and self._thread is None)

    def start_fit(self) -> None:
        """Validate inputs and spawn the worker thread."""
        if self._thread is not None:
            raise ValueError("a fit is already running; cancel it first")
        if self._target is None:
            raise ValueError("no target loaded; set_inputs(target=...) first")
        if not self._engine:
            raise ValueError("no engine selected; set_inputs(engine_name=...) first")
        worker = FitWorker(self._engine, self._target)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        self._worker = worker
        self._thread = thread
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText(f"Starting fit on '{self._engine}'...")
        thread.start()

    def cancel(self) -> None:
        """Request worker cancellation and wait briefly for cleanup."""
        if self._worker is not None:
            self._worker.request_cancel()
        if self._thread is not None:
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait(2000)
        self._teardown("Cancelled.")

    def _on_progress(self, text: str) -> None:
        self.lbl_status.setText(text)
        self.progress.emit(text)

    def _on_finished(self, result: Any) -> None:
        self.last_result = result
        self._teardown("Fit complete.")
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self._teardown(f"Fit failed: {message}")
        self.failed.emit(message)

    def _on_cancelled(self) -> None:
        self._teardown("Cancelled.")

    def _teardown(self, status_text: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(status_text)
        ready = self._target is not None and bool(self._engine)
        self.btn_run.setEnabled(ready)
