"""PyQt6 front end for the shared putting studies (#14177).

The window only collects inputs and shows rows: :mod:`.core` builds the service
arguments and formats answers, and the studies themselves run in
:mod:`src.shared.python.putting_analytics.service` (the same functions the MCP
tools and the ``/tools/putting-analytics`` API call). Each study runs on a
worker thread because a large rollout count takes a few seconds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from PyQt6 import QtCore, QtWidgets

from src.tools.putting_analytics.core import (
    MODES,
    SERVICE_FUNCTIONS,
    Rows,
    StudyForm,
    result_rows,
    study_args,
)

logger = logging.getLogger(__name__)

SHUTDOWN_WAIT_MS = 5_000
_MODE_LABELS = {
    "optimize": "Best aim and pace",
    "misread_cost": "Cost of misreads",
    "library_study": "Putt-class library",
}
_READS_HINT = "# slope_scale stimp_ratio (1 1 is a perfect read)\n1 1\n0.7 1\n1.3 1"
_CLASSES_HINT = (
    "# distance_ft slope_pct orientation stimp count\n"
    "4 1 uphill 10 4\n6 2 left_to_right 10 3"
)


class StudyWorker(QtCore.QObject):
    """Run one service call off the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, function_name: str, args: Mapping[str, Any]) -> None:
        super().__init__()
        self._function_name = function_name
        self._args = dict(args)

    def run(self) -> None:
        from src.shared.python.putting_analytics import service

        try:
            result = getattr(service, self._function_name)(self._args)
        except ValueError as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


def _spin(low: float, high: float, value: float, decimals: int = 2) -> Any:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setValue(value)
    return box


class MainWidget(QtWidgets.QWidget):
    """Form, run button and result table for the three putting studies."""

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._thread: Any | None = None
        self._worker: StudyWorker | None = None
        self._mode = MODES[0]
        self._closed = False
        self.setWindowTitle("Putting Analytics")
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self._build_form(), 0)
        layout.addWidget(self._build_results(), 1)

    def _build_form(self) -> Any:
        box = QtWidgets.QGroupBox("Study")
        form = QtWidgets.QFormLayout(box)
        self.mode_combo = QtWidgets.QComboBox()
        for mode in MODES:
            self.mode_combo.addItem(_MODE_LABELS[mode], mode)
        form.addRow("Study", self.mode_combo)
        self.distance = _spin(0.5, 100.0, 10.0, 1)
        self.grade = _spin(-10.0, 10.0, 0.0)
        self.aspect = _spin(-180.0, 180.0, 0.0, 1)
        self.stimp = _spin(5.0, 16.0, 10.0, 1)
        self.line_sd = _spin(0.0, 10.0, 1.0)
        self.speed_sd = _spin(0.0, 1.0, 0.08, 3)
        self.objective = QtWidgets.QComboBox()
        self.objective.addItems(["sg", "make_rate"])
        self.rollouts = QtWidgets.QSpinBox()
        self.rollouts.setRange(1, 10_000)
        self.rollouts.setValue(200)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        rows = (
            ("Distance (ft)", self.distance),
            ("Grade (%)", self.grade),
            ("Aspect (deg)", self.aspect),
            ("Stimp", self.stimp),
            ("Start-line SD (deg)", self.line_sd),
            ("Speed SD (fraction)", self.speed_sd),
            ("Objective", self.objective),
            ("Rollouts per action", self.rollouts),
            ("Seed", self.seed),
        )
        for label, widget in rows:
            form.addRow(label, widget)
        self._add_text_fields(form)
        self.run_button = QtWidgets.QPushButton("Run study")
        self.run_button.clicked.connect(self.run_study)
        form.addRow(self.run_button)
        return box

    def _add_text_fields(self, form: Any) -> None:
        self.baseline_distances = QtWidgets.QLineEdit()
        self.baseline_distances.setPlaceholderText("your distances, ft: 2 5 10 20 40")
        self.baseline_putts = QtWidgets.QLineEdit()
        self.baseline_putts.setPlaceholderText("your expected putts at each distance")
        self.aims = QtWidgets.QLineEdit("-4 -2 0 2 4")
        self.paces = QtWidgets.QLineEdit("0.5 1 2 3")
        self.reads = QtWidgets.QPlainTextEdit(_READS_HINT)
        self.classes = QtWidgets.QPlainTextEdit(_CLASSES_HINT)
        form.addRow("Baseline distances", self.baseline_distances)
        form.addRow("Baseline putts", self.baseline_putts)
        form.addRow("Aims (deg)", self.aims)
        form.addRow("Paces past hole (ft)", self.paces)
        form.addRow("Reads (misread study)", self.reads)
        form.addRow("Classes (library study)", self.classes)

    def _build_results(self) -> Any:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.status_label = QtWidgets.QLabel(
            "Enter your own expected-putts curve; no reference data ships."
        )
        self.status_label.setWordWrap(True)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Quantity", "Value"])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table, 1)
        return panel

    def current_mode(self) -> str:
        return str(self.mode_combo.currentData())

    def form_values(self) -> StudyForm:
        """Snapshot the form into the pure-data :class:`StudyForm`."""
        return StudyForm(
            distance_ft=self.distance.value(),
            grade_pct=self.grade.value(),
            aspect_deg=self.aspect.value(),
            stimp=self.stimp.value(),
            line_sd_deg=self.line_sd.value(),
            speed_sd=self.speed_sd.value(),
            objective=self.objective.currentText(),
            n=self.rollouts.value(),
            seed=self.seed.value(),
            baseline_distances=self.baseline_distances.text(),
            baseline_putts=self.baseline_putts.text(),
            aims=self.aims.text(),
            paces=self.paces.text(),
            extra={
                "reads": self.reads.toPlainText(),
                "classes": self.classes.toPlainText(),
            },
        )

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    def run_study(self) -> None:
        """Validate the form and start the selected study on a worker thread."""
        if self.is_running:
            return
        mode = self.current_mode()
        try:
            args = study_args(self.form_values(), mode)
        except ValueError as exc:
            self.status_label.setText(f"Check the inputs: {exc}")
            return
        self._start_worker(mode, args)

    def _start_worker(self, mode: str, args: Mapping[str, Any]) -> None:
        thread = QtCore.QThread(self)
        worker = StudyWorker(SERVICE_FUNCTIONS[mode], args)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Bound methods (not lambdas) so Qt queues the result onto this thread.
        worker.finished.connect(self._show)
        worker.failed.connect(self._fail)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._worker_done)
        self._thread, self._worker, self._mode = thread, worker, mode
        self.run_button.setEnabled(False)
        self.status_label.setText(f"Running: {_MODE_LABELS[mode]}...")
        thread.start()

    def _show(self, result: Mapping[str, Any]) -> None:
        if self._closed:
            return
        self.show_rows(result_rows(self._mode, result))
        self.status_label.setText(f"Done: {_MODE_LABELS[self._mode]}.")

    def _fail(self, message: str) -> None:
        if not self._closed:
            self.status_label.setText(f"The study rejected the inputs: {message}")

    def _worker_done(self) -> None:
        thread = self._thread
        self._thread, self._worker = None, None
        if thread is not None:
            thread.deleteLater()
        if not self._closed:
            self.run_button.setEnabled(True)

    def show_rows(self, rows: Rows) -> None:
        self.table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.table.setItem(index, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(index, 1, QtWidgets.QTableWidgetItem(value))

    def cleanup(self) -> bool:
        """Join the active worker; ``False`` if it outlived the wait."""
        self._closed = True
        thread = self._thread
        if thread is None:
            return True
        thread.quit()
        stopped = bool(thread.wait(SHUTDOWN_WAIT_MS))
        if not stopped:
            logger.warning("Putting study still running at shutdown")
        return stopped

    def delete_when_idle(self) -> None:
        """Defer deletion until a worker that outlived shutdown finishes."""
        if self._thread is None:
            self.deleteLater()
            return
        self._thread.finished.connect(self.deleteLater)

    def is_dirty(self) -> bool:
        return False


__all__ = ["MainWidget", "StudyWorker"]
