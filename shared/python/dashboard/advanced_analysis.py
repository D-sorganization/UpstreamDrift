"""Advanced Analysis Dialog for the Unified Dashboard.

Contains widgets and a dialog for advanced signal processing analysis:
- Spectrogram (Time-Frequency Analysis)
- Phase Plane (Position vs Velocity)
- Coherence (Frequency correlation between two signals)
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6 import QtWidgets

from shared.python.interfaces import RecorderInterface
from shared.python.plotting import MplCanvas
from shared.python.signal_processing import (
    compute_coherence,
    compute_spectrogram,
)

LOGGER = logging.getLogger(__name__)


class SpectrogramTab(QtWidgets.QWidget):
    """Tab for Spectrogram Analysis."""

    def __init__(
        self, recorder: RecorderInterface, initial_key: str = "joint_positions"
    ):
        super().__init__()
        self.recorder = recorder
        self.current_key = initial_key

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        self.combo_metric = QtWidgets.QComboBox()
        # Populate with standard metrics (can be extended)
        self.metric_options = {
            "Joint Positions": "joint_positions",
            "Joint Velocities": "joint_velocities",
            "Joint Torques": "joint_torques",
            "Ground Forces": "ground_forces",
            "Club Head Speed": "club_head_speed",
            "Total Control Accel": "control_accel",
        }
        self.combo_metric.addItems(list(self.metric_options.keys()))
        # Set initial selection if possible
        for label, key in self.metric_options.items():
            if key == self.current_key:
                self.combo_metric.setCurrentText(label)
                break

        self.combo_metric.currentTextChanged.connect(self._on_metric_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Metric:"))
        controls_layout.addWidget(self.combo_metric)

        self.spin_dim = QtWidgets.QSpinBox()
        self.spin_dim.setPrefix("Dim: ")
        self.spin_dim.setRange(0, 100)
        self.spin_dim.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.spin_dim)

        layout.addLayout(controls_layout)

        # Plot
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self.ax = self.canvas.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        self.update_plot()

    def _on_metric_changed(self, label: str) -> None:
        self.current_key = self.metric_options[label]
        self.update_plot()

    def update_plot(self) -> None:
        self.ax.clear()

        times, data = self.recorder.get_time_series(self.current_key)
        if isinstance(data, list):
            data = np.array(data) if data else None

        if data is None or len(times) == 0:
            self.ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            self.canvas.draw()
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dim_idx = self.spin_dim.value()
        if dim_idx >= data.shape[1]:
            dim_idx = 0  # Fallback

        signal_data = data[:, dim_idx]

        # Estimate fs
        if len(times) > 1:
            fs = 1.0 / np.mean(np.diff(times))
        else:
            fs = 100.0

        f, t, Sxx = compute_spectrogram(signal_data, fs)

        # Plot
        self.ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_xlabel("Time [sec]")
        self.ax.set_title(f"Spectrogram: {self.current_key} (Dim {dim_idx})")
        # We can add colorbar if we want, but keeping it simple for now

        self.canvas.draw()


class PhasePlaneTab(QtWidgets.QWidget):
    """Tab for Phase Plane Analysis (Position vs Velocity)."""

    def __init__(self, recorder: RecorderInterface):
        super().__init__()
        self.recorder = recorder

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        controls_layout = QtWidgets.QHBoxLayout()

        self.spin_dim = QtWidgets.QSpinBox()
        self.spin_dim.setPrefix("Dim: ")
        self.spin_dim.setRange(0, 100)
        self.spin_dim.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(QtWidgets.QLabel("Joint Index:"))
        controls_layout.addWidget(self.spin_dim)

        layout.addLayout(controls_layout)

        # Plot
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self.ax = self.canvas.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        self.update_plot()

    def update_plot(self) -> None:
        self.ax.clear()

        # Fetch Position and Velocity
        # Assuming generic recorder has these keys or we standardized them
        t_pos, pos = self.recorder.get_time_series("joint_positions")
        t_vel, vel = self.recorder.get_time_series("joint_velocities")

        # Helper to convert list to array
        if isinstance(pos, list):
            pos = np.array(pos) if pos else None
        if isinstance(vel, list):
            vel = np.array(vel) if vel else None

        if pos is None or vel is None or len(t_pos) == 0:
            self.ax.text(
                0.5, 0.5, "No Position/Velocity Data", ha="center", va="center"
            )
            self.canvas.draw()
            return

        # Ensure shapes match
        min_len = min(len(pos), len(vel))
        pos = pos[:min_len]
        vel = vel[:min_len]

        dim_idx = self.spin_dim.value()
        if dim_idx >= pos.shape[1] or dim_idx >= vel.shape[1]:
            self.ax.text(
                0.5, 0.5, f"Dimension {dim_idx} out of bounds", ha="center", va="center"
            )
            self.canvas.draw()
            return

        p = pos[:, dim_idx]
        v = vel[:, dim_idx]

        self.ax.plot(p, v)
        self.ax.set_xlabel("Position (rad or m)")
        self.ax.set_ylabel("Velocity (rad/s or m/s)")
        self.ax.set_title(f"Phase Plane (Dim {dim_idx})")
        self.ax.grid(True)

        # Mark start and end
        if len(p) > 0:
            self.ax.plot(p[0], v[0], "go", label="Start")
            self.ax.plot(p[-1], v[-1], "ro", label="End")
            self.ax.legend()

        self.canvas.draw()


class CoherenceTab(QtWidgets.QWidget):
    """Tab for Coherence Analysis between two signals."""

    def __init__(
        self,
        recorder: RecorderInterface,
        key1: str = "joint_positions",
        key2: str = "joint_torques",
    ):
        super().__init__()
        self.recorder = recorder

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        controls_layout = QtWidgets.QHBoxLayout()

        # Metric 1
        self.combo1 = QtWidgets.QComboBox()
        # Metric 2
        self.combo2 = QtWidgets.QComboBox()

        self.metric_options = {
            "Joint Positions": "joint_positions",
            "Joint Velocities": "joint_velocities",
            "Joint Torques": "joint_torques",
            "Ground Forces": "ground_forces",
            "Club Head Speed": "club_head_speed",
            "Total Control Accel": "control_accel",
        }

        self.combo1.addItems(list(self.metric_options.keys()))
        self.combo2.addItems(list(self.metric_options.keys()))

        # Set defaults
        for k, v in self.metric_options.items():
            if v == key1:
                self.combo1.setCurrentText(k)
            if v == key2:
                self.combo2.setCurrentText(k)

        self.combo1.currentTextChanged.connect(self.update_plot)
        self.combo2.currentTextChanged.connect(self.update_plot)

        controls_layout.addWidget(QtWidgets.QLabel("Signal 1:"))
        controls_layout.addWidget(self.combo1)
        controls_layout.addWidget(QtWidgets.QLabel("Signal 2:"))
        controls_layout.addWidget(self.combo2)

        self.spin_dim = QtWidgets.QSpinBox()
        self.spin_dim.setPrefix("Dim: ")
        self.spin_dim.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.spin_dim)

        layout.addLayout(controls_layout)

        # Plot
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self.ax = self.canvas.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        self.update_plot()

    def update_plot(self) -> None:
        self.ax.clear()

        key1 = self.metric_options[self.combo1.currentText()]
        key2 = self.metric_options[self.combo2.currentText()]

        t1, d1 = self.recorder.get_time_series(key1)
        t2, d2 = self.recorder.get_time_series(key2)

        if isinstance(d1, list):
            d1 = np.array(d1) if d1 else None
        if isinstance(d2, list):
            d2 = np.array(d2) if d2 else None

        if d1 is None or d2 is None or len(t1) == 0:
            self.ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            self.canvas.draw()
            return

        # Sync lengths
        min_len = min(len(d1), len(d2))
        d1 = d1[:min_len]
        d2 = d2[:min_len]
        t1 = t1[:min_len]

        dim_idx = self.spin_dim.value()
        if dim_idx >= d1.shape[1] or dim_idx >= d2.shape[1]:
            self.ax.text(0.5, 0.5, "Dimension out of bounds", ha="center", va="center")
            self.canvas.draw()
            return

        x = d1[:, dim_idx]
        y = d2[:, dim_idx]

        if len(t1) > 1:
            fs = 1.0 / np.mean(np.diff(t1))
        else:
            fs = 100.0

        f, Cxy = compute_coherence(x, y, fs)

        self.ax.plot(f, Cxy)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Coherence")
        self.ax.set_title(f"Coherence: {key1} vs {key2} (Dim {dim_idx})")
        self.ax.set_ylim(0, 1.05)
        self.ax.grid(True)

        self.canvas.draw()


class AdvancedAnalysisDialog(QtWidgets.QDialog):
    """Main Dialog for Advanced Analysis Tools."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        recorder: RecorderInterface,
        current_key: str = "joint_positions",
        comparison_key: str | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Advanced Analysis Tools")
        self.resize(800, 600)

        self.recorder = recorder

        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()

        # Spectrogram
        self.tab_spectrogram = SpectrogramTab(recorder, initial_key=current_key)
        self.tabs.addTab(self.tab_spectrogram, "Spectrogram")

        # Phase Plane
        self.tab_phase = PhasePlaneTab(recorder)
        self.tabs.addTab(self.tab_phase, "Phase Plane")

        # Coherence
        initial_key2 = comparison_key if comparison_key else "joint_torques"
        self.tab_coherence = CoherenceTab(recorder, key1=current_key, key2=initial_key2)
        self.tabs.addTab(self.tab_coherence, "Coherence")

        layout.addWidget(self.tabs)

        # Close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
