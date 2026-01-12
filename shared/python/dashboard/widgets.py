"""Widgets for the Unified Dashboard.

Contains:
- LivePlotWidget: Real-time plotting of simulation data.
- ControlPanel: Playback controls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from shared.python.dashboard.recorder import GenericPhysicsRecorder
from shared.python.plotting import MplCanvas

LOGGER = logging.getLogger(__name__)


class LivePlotWidget(QtWidgets.QWidget):
    """Widget for displaying real-time plots of simulation data.

    Supports plotting multiple metrics including:
    - Kinematics (Positions, Velocities)
    - Dynamics (Torques)
    - Advanced (ZTCF, Induced Accelerations)
    """

    def __init__(self, recorder: GenericPhysicsRecorder) -> None:
        """Initialize the widget.

        Args:
            recorder: The data recorder instance.
        """
        super().__init__()
        self.setAccessibleName("Live Simulation Plot")
        self.setAccessibleDescription(
            "Real-time plot of simulation metrics such as joint positions, velocities, and torques."
        )

        self.recorder = recorder
        self._main_layout = QtWidgets.QVBoxLayout(self)

        # Create Matplotlib canvas
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self._main_layout.addWidget(self.canvas)

        # Setup plot
        self.ax = self.canvas.fig.add_subplot(111)
        self.line_objects: list[Any] = []

        # Available metrics
        self.metric_options = {
            "Joint Positions": "joint_positions",
            "Joint Velocities": "joint_velocities",
            "Joint Torques": "joint_torques",
            "Ground Forces": "ground_forces",
            "Kinetic Energy": "kinetic_energy",
            "Club Head Speed": "club_head_speed",
            "ZTCF (Zero Torque)": "ztcf_accel",
            "ZVCF (Zero Velocity)": "zvcf_accel",
            "Drift Acceleration": "drift_accel",
            "Total Control Accel": "control_accel",
            "Induced Accel (Specific Source)": "induced_accel_source",
        }
        self.current_key = "joint_positions"
        self.current_label = "Joint Positions"

        # Controls Layout
        controls_layout = QtWidgets.QHBoxLayout()

        # Selector for data type
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(list(self.metric_options.keys()))
        self.combo.setToolTip("Select data to plot")
        self.combo.currentTextChanged.connect(self.set_plot_metric)

        lbl_metric = QtWidgets.QLabel("Metric:")
        lbl_metric.setBuddy(self.combo)
        controls_layout.addWidget(lbl_metric)
        controls_layout.addWidget(self.combo)

        # Plot Mode Selector
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["All Dimensions", "Single Dimension", "Norm"])
        self.mode_combo.setToolTip("Select plot mode")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        controls_layout.addWidget(self.mode_combo)

        # Selector for Dimension Index (Hidden by default)
        self.dim_spin = QtWidgets.QSpinBox()
        self.dim_spin.setRange(0, 100)  # Assume max 100 dims
        self.dim_spin.setPrefix("Dim: ")
        self.dim_spin.setVisible(False)  # Only for Single Dimension mode
        self.dim_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.dim_spin)

        # Selector for Induced Accel Source (Hidden by default)
        self.source_spin = QtWidgets.QSpinBox()
        self.source_spin.setRange(0, 100)  # Assume max 100 joints
        self.source_spin.setPrefix("Source Idx: ")
        self.source_spin.setToolTip("Index of the joint torque source to analyze for induced acceleration.")
        self.source_spin.setVisible(False)
        self.source_spin.valueChanged.connect(self._on_source_changed)

        lbl_source = QtWidgets.QLabel("Source:")
        lbl_source.setBuddy(self.source_spin)
        lbl_source.setVisible(False)
        self.source_label = lbl_source

        controls_layout.addWidget(lbl_source)
        controls_layout.addWidget(self.source_spin)

        # Checkbox for enabling computation (if expensive)
        self.chk_compute = QtWidgets.QCheckBox("Compute Real-time")
        self.chk_compute.setToolTip(
            "Enable real-time computation for advanced metrics (ZTCF, etc). May affect performance."
        )
        self.chk_compute.stateChanged.connect(self.toggle_computation)
        controls_layout.addWidget(self.chk_compute)

        self._main_layout.addLayout(controls_layout)

        # Initial plot setup
        self.ax.set_title("Live Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True)

    def set_plot_metric(self, label: str) -> None:
        """Change the metric being plotted."""
        self.current_label = label
        self.current_key = self.metric_options[label]

        # Show/Hide Source Spinner
        is_induced = self.current_key == "induced_accel_source"
        self.source_spin.setVisible(is_induced)
        self.source_label.setVisible(is_induced)

        # Update recorder config if needed
        self._update_recorder_config()

        # Reset plot
        self.ax.clear()
        self.ax.set_title(f"Live {label}")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True)
        self.line_objects = []  # Reset lines
        self.update_plot()

    def _on_source_changed(self, value: int) -> None:
        """Handle source index change."""
        self._update_recorder_config()
        # Clear plot as data source changed
        self.ax.clear()
        self.line_objects = []
        self.update_plot()

    def _on_mode_changed(self, mode: str) -> None:
        """Handle plot mode change."""
        self.dim_spin.setVisible(mode == "Single Dimension")
        self.ax.clear()
        self.line_objects = []
        self.update_plot()

    def toggle_computation(self, state: int) -> None:
        """Handle checkbox toggle."""
        self._update_recorder_config()

    def _update_recorder_config(self) -> None:
        """Update recorder configuration based on current selection and checkbox."""
        compute_enabled = self.chk_compute.isChecked()
        config = {
            "ztcf": False,
            "zvcf": False,
            "track_drift": False,
            "track_total_control": False,
            "induced_accel_sources": [],
        }

        if compute_enabled:
            if self.current_key == "ztcf_accel":
                config["ztcf"] = True
            elif self.current_key == "zvcf_accel":
                config["zvcf"] = True
            elif self.current_key == "drift_accel":
                config["track_drift"] = True
            elif self.current_key == "control_accel":
                config["track_total_control"] = True
            elif self.current_key == "induced_accel_source":
                # Add the selected source index to the list
                config["induced_accel_sources"] = [self.source_spin.value()]

        self.recorder.set_analysis_config(config)

    def update_plot(self) -> None:
        """Update the plot with the latest data."""
        times: np.ndarray = np.array([])
        data: np.ndarray | None = None

        if self.current_key == "induced_accel_source":
            # Fetch specific induced acceleration
            src_idx = self.source_spin.value()
            if src_idx in self.recorder.data["induced_accelerations"]:
                val = self.recorder.data["induced_accelerations"][src_idx]
                times = self.recorder.data["times"][: self.recorder.current_idx]
                data = val[: self.recorder.current_idx]
            else:
                times = np.array([])
                data = None
        else:
            # Standard metric
            times, data = self.recorder.get_time_series(self.current_key)

        if len(times) == 0:
            return

        # Ensure data is 2D
        if data is None:
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Apply filtering based on Plot Mode
        plot_mode = self.mode_combo.currentText()
        dim_label = "Dim"

        if plot_mode == "Single Dimension":
            dim_idx = self.dim_spin.value()
            if dim_idx < data.shape[1]:
                data = data[:, dim_idx : dim_idx + 1]
                dim_label = f"Dim {dim_idx}"
            else:
                data = np.zeros((len(data), 1))  # Invalid index fallback

        elif plot_mode == "Norm":
            # Compute Euclidean norm
            data = np.linalg.norm(data, axis=1).reshape(-1, 1)
            dim_label = "Norm"

        # Type assertion: data is guaranteed non-None at this point (we returned early above)
        assert data is not None

        # Limit to recent history to keep it fast
        max_points = 500
        if len(times) > max_points:
            times = times[-max_points:]
            data = data[-max_points:]

        n_dims = data.shape[1]

        # Initialize lines if needed
        if len(self.line_objects) != n_dims:
            self.ax.clear()
            self.line_objects = []
            for i in range(n_dims):
                label = (
                    dim_label
                    if n_dims == 1
                    else f"{dim_label} {i}" if plot_mode != "Norm" else "Norm"
                )
                if plot_mode == "All Dimensions":
                    label = f"Dim {i}"
                (line,) = self.ax.plot([], [], label=label)
                self.line_objects.append(line)

            if n_dims < 10 or plot_mode != "All Dimensions":
                self.ax.legend(loc="upper right")
            self.ax.set_title(f"Live {self.current_label}")
            self.ax.grid(True)

        # Update data
        for i, line in enumerate(self.line_objects):
            if i < data.shape[1]:
                line.set_data(times, data[:, i])

                # Update label if Single Dimension (to reflect changing index)
                if plot_mode == "Single Dimension":
                    line.set_label(dim_label)

        if plot_mode == "Single Dimension":
            self.ax.legend(loc="upper right")

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


class ControlPanel(QtWidgets.QGroupBox):
    """Control panel for simulation playback."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    pause_requested = QtCore.pyqtSignal()
    reset_requested = QtCore.pyqtSignal()
    toggle_playback_requested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Simulation Controls")
        self.setAccessibleName("Simulation Control Panel")
        self.setAccessibleDescription(
            "Controls for starting, pausing, stopping, and resetting the simulation playback."
        )

        layout = QtWidgets.QHBoxLayout(self)
        style = self.style()
        if style is None:
            return

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setIcon(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_start.setToolTip("Start simulation playback (Ctrl+R)")
        self.btn_start.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.btn_start.setStatusTip("Start the simulation")
        self.btn_start.clicked.connect(self.start_requested.emit)
        layout.addWidget(self.btn_start)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setIcon(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        )
        self.btn_pause.setToolTip("Pause/Resume simulation (Space)")
        self.btn_pause.setShortcut(QtGui.QKeySequence("Space"))
        self.btn_pause.setStatusTip("Pause or resume the simulation")
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self.pause_requested.emit)
        layout.addWidget(self.btn_pause)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setIcon(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop)
        )
        self.btn_stop.setToolTip("Stop simulation (S)")
        self.btn_stop.setStatusTip("Stop the simulation and reset time")
        self.btn_stop.setShortcut("S")
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self.btn_stop)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setIcon(
            style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.btn_reset.setToolTip("Reset simulation (R)")
        self.btn_reset.setStatusTip("Reset the simulation to initial state")
        self.btn_reset.setShortcut("R")
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.btn_reset)

        # Space shortcut for toggle playback (Start/Pause)
        self.shortcut_space = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self
        )
        self.shortcut_space.activated.connect(self.toggle_playback_requested.emit)
