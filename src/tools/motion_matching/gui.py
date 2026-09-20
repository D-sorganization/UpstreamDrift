"""Motion Matching tool: match a tour-average capture with one click.

A tabbed PyQt6 launcher tile over :mod:`src.tools.motion_matching.pipeline`:
1. Matching tab: choose capture, club, anthropometrics and stages (free/bound wrists,
   closure fit, ZMP filter, contact-aware shooting fit), build document and run matching.
2. Downswing experiment tab: configure contact/controller overrides, run downswing
   experiments and inspect root timeline errors and marker RMS to 1.5s.
3. MJX tab: export MJX differentiable optimization package and validate optimized
   references in the shared-law plant.

Nothing here computes; pipeline scripts do. Epic #10162, child #10158 (HO-4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.tools.motion_matching import pipeline

WINDOW_TITLE = "Motion Matching"


class RunWorker(QObject):
    """Executes a queue of external CLI commands sequentially via QProcess."""

    started = pyqtSignal()
    output_received = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._queue: list[list[str]] = []
        self._cwd: Path = pipeline.REPO_ROOT

    def is_running(self) -> bool:
        return self._process is not None

    def start(self, commands: list[list[str]], cwd: Path | None = None) -> None:
        if self._process is not None:
            return
        if not commands:
            self.finished.emit(0)
            return
        self._queue = list(commands)
        self._cwd = cwd or pipeline.REPO_ROOT
        self.started.emit()
        self._next()

    def _next(self) -> None:
        if not self._queue:
            self._process = None
            self.finished.emit(0)
            return
        command = self._queue.pop(0)
        self.output_received.emit("$ " + " ".join(command) + "\n")
        process = QProcess(self)
        process.setWorkingDirectory(str(self._cwd))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda: self.output_received.emit(
                process.readAllStandardOutput().data().decode(errors="replace")
            )
        )
        process.finished.connect(self._on_step_finished)
        self._process = process
        process.start(command[0], command[1:])

    def _on_step_finished(self, code: int, _status: object) -> None:
        self._process = None
        if code != 0:
            self._queue.clear()
            self.finished.emit(code)
            return
        self._next()

    def stop(self) -> None:
        self._queue.clear()
        if self._process is not None:
            self._process.kill()
            self._process = None
            self.finished.emit(-1)


class MotionMatchingWidget(QWidget):
    """Tabbed interface exposing Matching, Downswing Experiments, and MJX stages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = RunWorker(self)
        self._request: pipeline.MatchRequest | None = None
        self._exp_request: pipeline.ExperimentRequest | None = None

        self.tabs = QTabWidget(self)

        # Tab 1: Matching
        match_widget = self._create_matching_tab()
        self.tabs.addTab(match_widget, "Matching")

        # Tab 2: Downswing experiment
        exp_widget = self._create_experiment_tab()
        self.tabs.addTab(exp_widget, "Downswing experiment")

        # Tab 3: MJX
        mjx_widget = self._create_mjx_tab()
        self.tabs.addTab(mjx_widget, "MJX")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Wire worker signals
        self._worker.output_received.connect(self._on_output)
        self._worker.finished.connect(self._on_finished)

    # -------------------------------------------------------------------------
    # Matching Tab
    # -------------------------------------------------------------------------
    def _create_matching_tab(self) -> QWidget:
        widget = QWidget()
        self.capture = QComboBox()
        self.capture.addItems(list(pipeline.CAPTURES))
        self.club = QComboBox()
        self.club.addItems(list(pipeline.CLUBS))
        self.capture.currentTextChanged.connect(self._default_club)

        self.backend = QComboBox()
        self.backend.addItems(list(pipeline.BACKENDS))
        self.step_mode = QComboBox()
        self.step_mode.addItems(list(pipeline.STEP_MODES))

        self.stature = self._double_spin(1.71, 1.4, 2.2, 0.01)
        self.mass = self._double_spin(78.0, 40.0, 150.0, 0.5)
        self.trunk = self._double_spin(1.15, 0.8, 1.4, 0.01)
        self.arm = self._double_spin(1.10, 0.8, 1.4, 0.01)
        self.shoulder = self._double_spin(1.0, 0.8, 1.3, 0.01)

        form = QFormLayout()
        form.addRow("Backend", self.backend)
        form.addRow("Pink Step Mode", self.step_mode)
        form.addRow("Capture", self.capture)
        form.addRow("Club", self.club)
        form.addRow("Stature (m)", self.stature)
        form.addRow("Mass (kg)", self.mass)
        form.addRow("Trunk scale", self.trunk)
        form.addRow("Arm scale", self.arm)
        form.addRow("Shoulder scale", self.shoulder)

        # Stages group
        self.stages_group = QGroupBox("Stages")
        stages_layout = QFormLayout(self.stages_group)
        self.free_wrists = QCheckBox("Free wrists (--free-wrists)")
        self.bound_wrists = QCheckBox("Bound wrists (--bound-wrists)")
        self.fit_closure = QCheckBox("Fit closure weld (--fit-closure)")
        self.zmp_filter = QCheckBox("ZMP filter (--zmp-filter)")
        self.shooting_fit = QSpinBox()
        self.shooting_fit.setRange(0, 50)
        self.shooting_fit.setValue(0)
        self.shooting_gain = self._double_spin(0.70, 0.0, 1.0, 0.05)

        # Mutual exclusion between free and bound wrists
        def _on_free_wrists(checked: bool) -> None:
            if checked:
                self.bound_wrists.setChecked(False)

        def _on_bound_wrists(checked: bool) -> None:
            if checked:
                self.free_wrists.setChecked(False)

        self.free_wrists.toggled.connect(_on_free_wrists)
        self.bound_wrists.toggled.connect(_on_bound_wrists)

        stages_layout.addRow(self.free_wrists)
        stages_layout.addRow(self.bound_wrists)
        stages_layout.addRow(self.fit_closure)
        stages_layout.addRow(self.zmp_filter)
        stages_layout.addRow("Shooting fit iterations", self.shooting_fit)
        stages_layout.addRow("Shooting gain", self.shooting_gain)

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

        layout = QVBoxLayout(widget)
        layout.addLayout(form)
        layout.addWidget(self.stages_group)
        layout.addLayout(buttons)
        layout.addWidget(self.log, stretch=1)
        layout.addWidget(self.results)
        return widget

    # -------------------------------------------------------------------------
    # Downswing Experiment Tab
    # -------------------------------------------------------------------------
    def _create_experiment_tab(self) -> QWidget:
        widget = QWidget()
        default_dir = str(pipeline.FULL_BODY / "evidence/ground_support/anthro_driver")
        self.exp_run_dir = QLineEdit(default_dir)
        browse_run = QPushButton("Browse...")
        browse_run.clicked.connect(
            lambda: self._browse_dir(self.exp_run_dir, "Select Run Directory")
        )
        run_box = QHBoxLayout()
        run_box.addWidget(self.exp_run_dir)
        run_box.addWidget(browse_run)

        self.exp_name = QLineEdit("experiment_1")
        self.exp_cutoff = self._double_spin(0.0, 0.0, 179.0, 1.0)
        self.exp_omega = self._double_spin(120.0, 10.0, 500.0, 5.0)
        self.exp_zeta = self._double_spin(1.0, 0.1, 5.0, 0.1)
        self.exp_feedforward = self._double_spin(1.0, 0.0, 1.0, 0.05)
        self.exp_balance = QCheckBox("Enable balance regulation")
        self.exp_balance.setChecked(True)
        self.exp_legs_omega = self._double_spin(0.0, 0.0, 200.0, 5.0)
        self.exp_stiffness = self._double_spin(0.0, 0.0, 200000.0, 1000.0)

        self.exp_reference = QLineEdit()
        browse_ref = QPushButton("Browse...")
        browse_ref.clicked.connect(
            lambda: self._browse_file(self.exp_reference, "NPZ files (*.npz)")
        )
        ref_box = QHBoxLayout()
        ref_box.addWidget(self.exp_reference)
        ref_box.addWidget(browse_ref)

        form = QFormLayout()
        form.addRow("Run directory", run_box)
        form.addRow("Experiment name", self.exp_name)
        form.addRow("Cutoff (Hz, 0=none)", self.exp_cutoff)
        form.addRow("Omega (rad/s)", self.exp_omega)
        form.addRow("Zeta", self.exp_zeta)
        form.addRow("Feedforward gain", self.exp_feedforward)
        form.addRow(self.exp_balance)
        form.addRow("Legs omega (0=default)", self.exp_legs_omega)
        form.addRow("Sole stiffness N/m (0=default)", self.exp_stiffness)
        form.addRow("Custom reference", ref_box)

        self.exp_run_btn = QPushButton("Run downswing experiment")
        self.exp_run_btn.clicked.connect(self.start_experiment)
        self.exp_stop_btn = QPushButton("Stop")
        self.exp_stop_btn.setEnabled(False)
        self.exp_stop_btn.clicked.connect(self.stop)

        btns = QHBoxLayout()
        btns.addWidget(self.exp_run_btn)
        btns.addWidget(self.exp_stop_btn)

        self.exp_log = QPlainTextEdit()
        self.exp_log.setReadOnly(True)
        self.exp_results = QLabel("No experiment yet.")
        self.exp_results.setWordWrap(True)

        layout = QVBoxLayout(widget)
        layout.addLayout(form)
        layout.addLayout(btns)
        layout.addWidget(self.exp_log, stretch=1)
        layout.addWidget(self.exp_results)
        return widget

    # -------------------------------------------------------------------------
    # MJX Tab
    # -------------------------------------------------------------------------
    def _create_mjx_tab(self) -> QWidget:
        widget = QWidget()
        default_dir = str(pipeline.FULL_BODY / "evidence/ground_support/anthro_driver")
        self.mjx_run_dir = QLineEdit(default_dir)
        browse_dir = QPushButton("Browse...")
        browse_dir.clicked.connect(
            lambda: self._browse_dir(self.mjx_run_dir, "Select Run Directory")
        )
        run_box = QHBoxLayout()
        run_box.addWidget(self.mjx_run_dir)
        run_box.addWidget(browse_dir)

        self.mjx_export_btn = QPushButton("Export MJX package")
        self.mjx_export_btn.clicked.connect(self.start_mjx_export)

        self.mjx_ref_path = QLineEdit()
        browse_ref = QPushButton("Browse...")
        browse_ref.clicked.connect(
            lambda: self._browse_file(self.mjx_ref_path, "NPZ files (*.npz)")
        )
        ref_box = QHBoxLayout()
        ref_box.addWidget(self.mjx_ref_path)
        ref_box.addWidget(browse_ref)

        self.mjx_validate_btn = QPushButton("Validate reference in shared plant")
        self.mjx_validate_btn.clicked.connect(self.start_mjx_validation)

        self.mjx_stop_btn = QPushButton("Stop")
        self.mjx_stop_btn.setEnabled(False)
        self.mjx_stop_btn.clicked.connect(self.stop)

        form = QFormLayout()
        form.addRow("Run directory", run_box)
        form.addRow(self.mjx_export_btn)
        form.addRow("Optimised reference NPZ", ref_box)
        form.addRow(self.mjx_validate_btn)

        self.mjx_log = QPlainTextEdit()
        self.mjx_log.setReadOnly(True)
        self.mjx_results = QLabel("No MJX operation yet.")
        self.mjx_results.setWordWrap(True)

        layout = QVBoxLayout(widget)
        layout.addLayout(form)
        layout.addWidget(self.mjx_stop_btn)
        layout.addWidget(self.mjx_log, stretch=1)
        layout.addWidget(self.mjx_results)
        return widget

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _double_spin(
        value: float, low: float, high: float, step: float
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setValue(value)
        return box

    def _browse_dir(self, line_edit: QLineEdit, caption: str) -> None:
        chosen = QFileDialog.getExistingDirectory(self, caption)
        if chosen:
            line_edit.setText(chosen)

    def _browse_file(self, line_edit: QLineEdit, filter_str: str) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Select file", filter=filter_str)
        if chosen:
            line_edit.setText(chosen)

    def _default_club(self, capture: str) -> None:
        self.club.setCurrentText(pipeline.CLUB_FOR_CAPTURE.get(capture, "driver"))

    # -------------------------------------------------------------------------
    # Request & Command Builders (LoD: widgets bind, pipeline builds)
    # -------------------------------------------------------------------------
    def request(self) -> pipeline.MatchRequest:
        return pipeline.MatchRequest(
            capture=self.capture.currentText(),
            club=self.club.currentText(),
            stature_m=self.stature.value(),
            mass_kg=self.mass.value(),
            trunk_scale=self.trunk.value(),
            arm_scale=self.arm.value(),
            shoulder_scale=self.shoulder.value(),
            free_wrists=self.free_wrists.isChecked(),
            bound_wrists=self.bound_wrists.isChecked(),
            fit_closure=self.fit_closure.isChecked(),
            zmp_filter=self.zmp_filter.isChecked(),
            shooting_fit=self.shooting_fit.value(),
            shooting_gain=self.shooting_gain.value(),
            backend=self.backend.currentText(),
            step_mode=self.step_mode.currentText(),
        )

    def experiment_request(self) -> pipeline.ExperimentRequest:
        cutoff = self.exp_cutoff.value()
        legs_omega = self.exp_legs_omega.value()
        stiffness = self.exp_stiffness.value()
        ref = self.exp_reference.text().strip()
        return pipeline.ExperimentRequest(
            run=Path(self.exp_run_dir.text().strip()),
            name=self.exp_name.text().strip() or "experiment_1",
            cutoff_hz=cutoff if cutoff > 0 else None,
            omega=self.exp_omega.value(),
            zeta=self.exp_zeta.value(),
            feedforward=self.exp_feedforward.value(),
            legs_omega=legs_omega if legs_omega > 0 else None,
            balance=self.exp_balance.isChecked(),
            stiffness=stiffness if stiffness > 0 else None,
            reference=Path(ref) if ref else None,
        )

    def experiment_command(self) -> list[str]:
        return pipeline.experiment_command(self.experiment_request())

    def mjx_export_command(self) -> list[str]:
        run_path = Path(self.mjx_run_dir.text().strip())
        return pipeline.export_mjx_command(run_path)

    def mjx_validate_command(self) -> list[str]:
        run_path = Path(self.mjx_run_dir.text().strip())
        ref_path = Path(self.mjx_ref_path.text().strip())
        return pipeline.validate_reference_command(run_path, ref_path)

    # -------------------------------------------------------------------------
    # Execution Triggers
    # -------------------------------------------------------------------------
    def start(self) -> None:
        if self._worker.is_running():
            return
        self._request = self.request()
        self.log.clear()
        self.results.setText("Running matching pipeline...")
        self._set_active_buttons(self.run_button, self.stop_button, running=True)
        commands = [
            pipeline.build_command(self._request),
            pipeline.match_command(self._request),
        ]
        self._worker.start(commands)

    def start_experiment(self) -> None:
        if self._worker.is_running():
            return
        self._exp_request = self.experiment_request()
        self.exp_log.clear()
        self.exp_results.setText("Running downswing experiment...")
        self._set_active_buttons(self.exp_run_btn, self.exp_stop_btn, running=True)
        self._worker.start([self.experiment_command()])

    def start_mjx_export(self) -> None:
        if self._worker.is_running():
            return
        self.mjx_log.clear()
        self.mjx_results.setText("Exporting MJX package...")
        self._set_active_buttons(self.mjx_export_btn, self.mjx_stop_btn, running=True)
        self._worker.start([self.mjx_export_command()])

    def start_mjx_validation(self) -> None:
        if self._worker.is_running():
            return
        self.mjx_log.clear()
        self.mjx_results.setText("Validating MJX reference...")
        self._set_active_buttons(self.mjx_validate_btn, self.mjx_stop_btn, running=True)
        self._worker.start([self.mjx_validate_command()])

    def stop(self) -> None:
        self._worker.stop()

    def _set_active_buttons(
        self, run_btn: QPushButton, stop_btn: QPushButton, running: bool
    ) -> None:
        run_btn.setEnabled(not running)
        stop_btn.setEnabled(running)

    def _on_output(self, text: str) -> None:
        # Route output to current active tab log
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.log.appendPlainText(text)
        elif idx == 1:
            self.exp_log.appendPlainText(text)
        elif idx == 2:
            self.mjx_log.appendPlainText(text)

    def _on_finished(self, code: int) -> None:
        self._set_active_buttons(self.run_button, self.stop_button, running=False)
        self._set_active_buttons(self.exp_run_btn, self.exp_stop_btn, running=False)
        self._set_active_buttons(self.mjx_export_btn, self.mjx_stop_btn, running=False)
        self._set_active_buttons(
            self.mjx_validate_btn, self.mjx_stop_btn, running=False
        )

        idx = self.tabs.currentIndex()
        if code != 0:
            msg = f"Step failed with exit code {code}; see log for details."
            if idx == 0:
                self.results.setText(msg)
            elif idx == 1:
                self.exp_results.setText(msg)
            elif idx == 2:
                self.mjx_results.setText(msg)
            return

        if idx == 0 and self._request is not None:
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
        elif idx == 1 and self._exp_request is not None:
            try:
                summary = pipeline.read_experiment_summary(
                    self._exp_request.run, self._exp_request.name
                )
                self.exp_results.setText(
                    f"Experiment '{summary['name']}' Complete:\n"
                    f"Root Error Max: {summary['root_error_max_mm']} mm\n"
                    f"Marker RMS to 1.5s: {summary['marker_rms_to_1_5s_mm']} mm ({summary['marker_rms_to_1_5s_m']} m)\n"
                    f"Marker RMS whole run: {summary['marker_rms_mm']} mm\n"
                    f"Root Error Timeline (m): {json.dumps(summary['root_error_timeline_m'])}\n"
                    f"Inside Support Polygon: {summary['inside_support_polygon_fraction']}"
                )
            except ValueError as exc:
                self.exp_results.setText(str(exc))
        elif idx == 2:
            self.mjx_results.setText("MJX operation completed successfully.")


def get_dockable_ui() -> QMainWindow:
    """Main window for docking in the launcher."""
    window = QMainWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.setCentralWidget(MotionMatchingWidget())
    window.resize(780, 680)
    return window


def main(argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    window = get_dockable_ui()
    window.show()
    return app.exec()


__all__ = ["MotionMatchingWidget", "RunWorker", "get_dockable_ui", "main"]
