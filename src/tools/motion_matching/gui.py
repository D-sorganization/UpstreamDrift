"""Motion Matching tool: match a tour-average capture with one click.

A small PyQt6 form over :mod:`src.tools.motion_matching.pipeline`: choose
the capture (driver or 7-iron), the club, the subject stature and mass and
the length scale factors, build the anthropometric document and run the
ground-support matching (address, full-capture IK, forward-dynamics
tracking with contact). The log streams into the window; when the run
ends the receipt summary and the playback GIF paths are shown. Nothing
here computes; the pipeline scripts do. Epic #10113, child #10106.
"""

from __future__ import annotations

import json
import sys

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.tools.motion_matching import pipeline

WINDOW_TITLE = "Motion Matching"


class MotionMatchingWidget(QWidget):
    """Form, run buttons, log and results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._queue: list[list[str]] = []
        self._request: pipeline.MatchRequest | None = None
        self.capture = QComboBox()
        self.capture.addItems(list(pipeline.CAPTURES))
        self.club = QComboBox()
        self.club.addItems(list(pipeline.CLUBS))
        self.capture.currentTextChanged.connect(self._default_club)
        self.stature = self._spin(1.71, 1.4, 2.2, 0.01)
        self.mass = self._spin(78.0, 40.0, 150.0, 0.5)
        self.trunk = self._spin(1.15, 0.8, 1.4, 0.01)
        self.arm = self._spin(1.10, 0.8, 1.4, 0.01)
        self.shoulder = self._spin(1.0, 0.8, 1.3, 0.01)
        form = QFormLayout()
        form.addRow("Capture", self.capture)
        form.addRow("Club", self.club)
        form.addRow("Stature (m)", self.stature)
        form.addRow("Mass (kg)", self.mass)
        form.addRow("Trunk scale", self.trunk)
        form.addRow("Arm scale", self.arm)
        form.addRow("Shoulder scale", self.shoulder)
        self.run_button = QPushButton("Build document and match")
        self.run_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.results = QLabel("No run yet.")
        self.results.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.log, stretch=1)
        layout.addWidget(self.results)

    @staticmethod
    def _spin(value: float, low: float, high: float, step: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setValue(value)
        return box

    def _default_club(self, capture: str) -> None:
        self.club.setCurrentText(pipeline.CLUB_FOR_CAPTURE.get(capture, "driver"))

    def request(self) -> pipeline.MatchRequest:
        return pipeline.MatchRequest(
            capture=self.capture.currentText(),
            club=self.club.currentText(),
            stature_m=self.stature.value(),
            mass_kg=self.mass.value(),
            trunk_scale=self.trunk.value(),
            arm_scale=self.arm.value(),
            shoulder_scale=self.shoulder.value(),
        )

    def start(self) -> None:
        if self._process is not None:
            return
        self._request = self.request()
        self._queue = [
            pipeline.build_command(self._request),
            pipeline.match_command(self._request),
        ]
        self.log.clear()
        self.results.setText("Running...")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._next()

    def _next(self) -> None:
        if not self._queue:
            self._finish()
            return
        command = self._queue.pop(0)
        self.log.appendPlainText("$ " + " ".join(command))
        process = QProcess(self)
        process.setWorkingDirectory(str(pipeline.REPO_ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(
                process.readAllStandardOutput().data().decode(errors="replace")
            )
        )
        process.finished.connect(self._step_finished)
        self._process = process
        process.start(command[0], command[1:])

    def _step_finished(self, code: int, _status: object) -> None:
        self._process = None
        if code != 0:
            self.results.setText(f"A step failed with exit code {code}; see the log.")
            self._reset_buttons()
            return
        self._next()

    def _finish(self) -> None:
        assert self._request is not None
        try:
            summary = pipeline.read_summary(self._request.output_dir)
            gifs = ", ".join(
                str(p) for p in pipeline.artefacts(self._request.output_dir)
            )
            self.results.setText(
                json.dumps(summary, indent=1) + f"\nPlayback: {gifs or 'none'}"
            )
        except ValueError as exc:
            self.results.setText(str(exc))
        self._reset_buttons()

    def stop(self) -> None:
        self._queue.clear()
        if self._process is not None:
            self._process.kill()

    def _reset_buttons(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)


def get_dockable_ui() -> QMainWindow:
    """Main window for docking in the launcher."""
    window = QMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.setCentralWidget(MotionMatchingWidget())
    window.resize(720, 640)
    return window


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    window = get_dockable_ui()
    window.show()
    return app.exec()


__all__ = ["MotionMatchingWidget", "get_dockable_ui", "main"]
