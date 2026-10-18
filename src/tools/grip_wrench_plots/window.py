"""Tabbed Grip Wrench / NSX window (NSX-43, #12575).

The existing grip wrench view is the first tab; Invariants, Envelopes, Hand
Split and Diagnostics follow and render only the :class:`NSXPlotModel`.  A
preset selector lists the NSX-11 registry presets.  Runs go through a
``QThreadPool`` worker with progress and cancel; the GUI thread only draws.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PyQt6.QtCore import QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.api.services import loop_nullspace_service as service
from src.shared.python.loop_nullspace.plot_model import NSXPlotModel
from src.tools.grip_wrench_plots.gui import GripWrenchPlotWidget
from src.tools.grip_wrench_plots.nsx_job import (
    NSXJob,
    NSXRequest,
    preset_choices,
    request_job,
)
from src.tools.grip_wrench_plots.nsx_tabs import NSXTabs
from src.tools.grip_wrench_plots.nsx_worker import NSXWorker

__all__ = ["GripWrenchTabsWidget", "parse_frames"]

JobFactory = Callable[[NSXRequest], NSXJob]
GRIP_TAB_TITLE = "Grip Wrench"
DEFAULT_FRAMES = "0, 3, 6, 9"


def parse_frames(text: str) -> tuple[int, ...]:
    """Frame indices from ``"0, 3, 6"``.

    Raises:
        ValueError: Empty, non-integer, negative or repeated indices.
    """
    try:
        frames = tuple(int(part) for part in text.replace(",", " ").split())
    except ValueError as exc:
        raise ValueError(f"frames must be integers: {text!r}") from exc
    if not frames:
        raise ValueError("enter at least one frame index")
    if min(frames) < 0 or len(set(frames)) != len(frames):
        raise ValueError("frame indices must be unique and non-negative")
    return frames


class GripWrenchTabsWidget(QWidget):
    """Grip wrench view plus the NSX tabs, with a preset selector and a runner."""

    analysis_finished = pyqtSignal(object)
    analysis_cancelled = pyqtSignal()
    analysis_failed = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        job_factory: JobFactory = request_job,
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._job_factory = job_factory
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._worker: NSXWorker | None = None
        self._model: NSXPlotModel | None = None
        self._build_ui(sources)

    def _build_ui(self, sources: Sequence[str] | None) -> None:
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.preset_combo = QComboBox()
        for choice in preset_choices():
            self.preset_combo.addItem(choice.label, choice.preset_id)
        self.source_combo = QComboBox()
        names = service.registered_sources() if sources is None else tuple(sources)
        self.source_combo.addItems(list(names))
        self.frames_edit = QLineEdit(DEFAULT_FRAMES)
        self.allocation_check = QCheckBox("Allocation")
        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        for label, widget in (
            ("Preset", self.preset_combo),
            ("Source", self.source_combo),
            ("Frames", self.frames_edit),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(widget)
        for control in (self.allocation_check, self.run_button, self.cancel_button):
            controls.addWidget(control)
        root.addLayout(controls)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("")
        root.addWidget(self.progress)
        root.addWidget(self.status_label)
        self.grip_widget = GripWrenchPlotWidget()
        self.tabs = NSXTabs()
        self.tabs.insertTab(0, self.grip_widget, GRIP_TAB_TITLE)
        self.tabs.setCurrentIndex(0)
        root.addWidget(self.tabs, 1)
        self.run_button.clicked.connect(self.run_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self._set_idle_status("")

    # -------------------------------------------------------------- state
    def _set_idle_status(self, text: str) -> None:
        has_source = self.source_combo.count() > 0
        self.run_button.setEnabled(has_source and self._worker is None)
        self.cancel_button.setEnabled(self._worker is not None)
        if text:
            self.status_label.setText(text)
        elif not has_source:
            self.status_label.setText("no trajectory source registered")

    def plot_model(self) -> NSXPlotModel | None:
        """The model the NSX tabs currently show."""
        return self._model

    def set_plot_model(self, model: NSXPlotModel | None) -> None:
        """Render ``model`` in the NSX tabs and show the Invariants tab first."""
        self._model = model
        self.tabs.set_plot_model(model)
        if model is not None:
            self.tabs.setCurrentIndex(1)
            self.status_label.setText(
                f"{model.status}: preset {model.preset}"
                + (f" ({model.reason})" if model.reason else "")
            )

    def selected_preset(self) -> str:
        """Registry id of the chosen preset."""
        return str(self.preset_combo.currentData())

    def is_running(self) -> bool:
        return self._worker is not None

    # ----------------------------------------------------------- running
    def current_request(self) -> NSXRequest:
        """The request the controls describe (``ValueError`` when invalid)."""
        return NSXRequest(
            source_id=self.source_combo.currentText(),
            preset=self.selected_preset(),
            frame_indices=parse_frames(self.frames_edit.text()),
            include_allocation=self.allocation_check.isChecked(),
        )

    def run_analysis(self) -> None:
        """Start a run on the thread pool (no-op while one is running)."""
        if self._worker is not None:
            return
        try:
            job = self._job_factory(self.current_request())
        except ValueError as exc:
            self.status_label.setText(f"cannot run: {exc}")
            return
        worker = NSXWorker(job)
        sig = worker.signals
        sig.progress.connect(self._on_progress)
        sig.finished.connect(self._on_finished)
        sig.failed.connect(self._on_failed)
        sig.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self.progress.setValue(0)
        self._set_idle_status("running")
        self._pool.start(worker)

    def cancel_analysis(self) -> None:
        """Ask the running job to stop at its next stage boundary."""
        worker = self._worker
        if worker is not None:
            worker.cancel()
            self.status_label.setText("cancelling")

    def wait_idle(self, timeout_ms: int = 30000) -> bool:
        """Block until the pool is idle (tests and shutdown only)."""
        return self._pool.waitForDone(timeout_ms)

    def _finish(self, text: str) -> None:
        self._worker = None
        self._set_idle_status(text)

    def _on_progress(self, stage: str, percent: float) -> None:
        self.progress.setValue(int(percent))
        self.status_label.setText(f"running: {stage}")

    def _on_finished(self, model: Any) -> None:
        self.set_plot_model(model)
        self.progress.setValue(100)
        self._finish(self.status_label.text())
        self.analysis_finished.emit(model)

    def _on_failed(self, message: str) -> None:
        self._finish(f"failed: {message}")
        self.analysis_failed.emit(message)

    def _on_cancelled(self) -> None:
        self._finish("cancelled")
        self.analysis_cancelled.emit()

    def cleanup(self) -> None:
        """Cancel any run, wait for the pool and drop references; idempotent."""
        worker = self._worker
        if worker is not None:
            worker.cancel()
        # noqa: gui-thread/ok - shutdown only; the cancelled job ends at a stage boundary.
        self._pool.waitForDone(30000)
        self._worker = None
        self._model = None
        self.grip_widget.cleanup()
