"""Widgets for the Unified Dashboard.

Contains:
- LivePlotWidget: Real-time plotting of simulation data.
- ControlPanel: Playback controls.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6 import QtCore, QtWidgets

from shared.python.dashboard.recorder import GenericPhysicsRecorder
from shared.python.plotting import MplCanvas

LOGGER = logging.getLogger(__name__)


class LivePlotWidget(QtWidgets.QWidget):
    """Widget for displaying real-time plots of simulation data."""

    def __init__(self, recorder: GenericPhysicsRecorder) -> None:
        """Initialize the widget.

        Args:
            recorder: The data recorder instance.
        """
        super().__init__()
        self.recorder = recorder
        self.layout = QtWidgets.QVBoxLayout(self)

        # Create Matplotlib canvas
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self.layout.addWidget(self.canvas)

        # Setup plot
        self.ax = self.canvas.fig.add_subplot(111)
        self.line_objects: list[Any] = []
        self.data_keys = ["joint_positions", "joint_velocities", "joint_torques"]
        self.current_key = "joint_positions"

        # Selector for data type
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(self.data_keys)
        self.combo.currentTextChanged.connect(self.set_plot_metric)
        self.layout.addWidget(self.combo)

        # Initial plot setup
        self.ax.set_title("Live Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True)

    def set_plot_metric(self, key: str) -> None:
        """Change the metric being plotted."""
        self.current_key = key
        self.ax.clear()
        self.ax.set_title(f"Live {key}")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True)
        self.line_objects = []  # Reset lines
        self.update_plot()

    def update_plot(self) -> None:
        """Update the plot with the latest data."""
        times, data = self.recorder.get_time_series(self.current_key)

        if len(times) == 0:
            return

        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

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
                (line,) = self.ax.plot([], [], label=f"Dim {i}")
                self.line_objects.append(line)
            self.ax.legend(loc="upper right")
            self.ax.set_title(f"Live {self.current_key}")
            self.ax.grid(True)

        # Update data
        for i, line in enumerate(self.line_objects):
            line.set_data(times, data[:, i])

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


class ControlPanel(QtWidgets.QGroupBox):
    """Control panel for simulation playback."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    pause_requested = QtCore.pyqtSignal()
    reset_requested = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Simulation Controls")

        layout = QtWidgets.QHBoxLayout(self)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self.start_requested.emit)
        layout.addWidget(self.btn_start)

        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self.pause_requested.emit)
        layout.addWidget(self.btn_pause)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self.btn_stop)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self.btn_reset)
