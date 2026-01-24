"""Advanced Analysis Dialog for the Unified Dashboard.

Contains widgets and a dialog for advanced signal processing analysis:
- Spectrogram (Time-Frequency Analysis)
- Wavelet Analysis (CWT)
- Phase Plane (Position vs Velocity)
- Coherence (Frequency correlation between two signals)
- Swing Plane Analysis (3D Trajectory & Deviation)
- Correlation Heatmap (Multi-variable analysis)
"""

from __future__ import annotations

from src.shared.python.logging_config import get_logger

import numpy as np
from PyQt6 import QtWidgets

from src.shared.python.interfaces import RecorderInterface
from src.shared.python.plotting import MplCanvas
from src.shared.python.signal_processing import (
    compute_coherence,
    compute_cwt,
    compute_spectrogram,
)
from src.shared.python.swing_plane_analysis import SwingPlaneAnalyzer

logger = get_logger(__name__)

# Numerical constants for signal processing
LOG_EPSILON = 1e-12  # Small epsilon to avoid log(0) in dB calculations
DB_CONVERSION = 10  # Decibel conversion factor: dB = 10 * log10(power)


def _validate_dimension_index(dim_idx: int, *arrays: np.ndarray) -> bool:
    """Validate that a dimension index is valid for all provided arrays.

    Args:
        dim_idx: The dimension index to validate.
        *arrays: Variable number of numpy arrays to check against.

    Returns:
        True if the dimension index is valid for all arrays, False otherwise.
    """
    for arr in arrays:
        if arr is None:
            return False
        if arr.ndim < 2:
            # 1D arrays only have dim 0
            if dim_idx > 0:
                return False
        elif dim_idx >= arr.shape[1]:
            return False
    return True


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

        times, raw_data = self.recorder.get_time_series(self.current_key)
        data: np.ndarray | None
        if isinstance(raw_data, list):
            data = np.array(raw_data) if raw_data else None
        else:
            data = raw_data

        if data is None or len(times) == 0:
            self.ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            self.canvas.draw()
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dim_idx = self.spin_dim.value()
        if not _validate_dimension_index(dim_idx, data):
            logger.warning(
                "Dimension index %d out of bounds for data shape %s, falling back to 0",
                dim_idx,
                data.shape,
            )
            dim_idx = 0  # Fallback to first dimension if invalid

        signal_data = data[:, dim_idx]

        # Estimate fs
        if len(times) > 1:
            fs = float(1.0 / np.mean(np.diff(times)))
        else:
            fs = 100.0

        f, t, Sxx = compute_spectrogram(signal_data, fs)

        # Plot
        self.ax.pcolormesh(
            t, f, DB_CONVERSION * np.log10(Sxx + LOG_EPSILON), shading="gouraud"
        )
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_xlabel("Time [sec]")
        self.ax.set_title(f"Spectrogram: {self.current_key} (Dim {dim_idx})")
        # We can add colorbar if we want, but keeping it simple for now

        self.canvas.draw()


class WaveletTab(QtWidgets.QWidget):
    """Tab for Continuous Wavelet Transform (CWT) Analysis."""

    def __init__(
        self, recorder: RecorderInterface, initial_key: str = "joint_velocities"
    ):
        super().__init__()
        self.recorder = recorder
        self.current_key = initial_key

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        self.combo_metric = QtWidgets.QComboBox()
        self.metric_options = {
            "Joint Positions": "joint_positions",
            "Joint Velocities": "joint_velocities",
            "Joint Torques": "joint_torques",
            "Club Head Speed": "club_head_speed",
            "Total Control Accel": "control_accel",
        }
        self.combo_metric.addItems(list(self.metric_options.keys()))
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

        self.spin_w0 = QtWidgets.QDoubleSpinBox()
        self.spin_w0.setPrefix("w0: ")
        self.spin_w0.setRange(2.0, 20.0)
        self.spin_w0.setValue(6.0)
        self.spin_w0.setSingleStep(0.5)
        self.spin_w0.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.spin_w0)

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

        times, raw_data = self.recorder.get_time_series(self.current_key)
        data: np.ndarray | None = np.array(raw_data) if raw_data is not None else None

        if data is None or len(times) == 0:
            self.ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            self.canvas.draw()
            return

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dim_idx = self.spin_dim.value()
        if not _validate_dimension_index(dim_idx, data):
            self.ax.text(0.5, 0.5, "Dimension out of bounds", ha="center", va="center")
            self.canvas.draw()
            return

        signal_data = data[:, dim_idx]

        # Estimate fs
        if len(times) > 1:
            fs = float(1.0 / np.mean(np.diff(times)))
        else:
            fs = 100.0

        w0 = self.spin_w0.value()
        freqs, t, cwt_mat = compute_cwt(signal_data, fs, w0=w0, num_freqs=64)

        # Plot Magnitude
        # Use abs(cwt_mat)
        mag = np.abs(cwt_mat)
        self.ax.pcolormesh(t, freqs, mag, shading="gouraud", cmap="plasma")
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_xlabel("Time [sec]")
        self.ax.set_yscale("log")
        self.ax.set_title(
            f"Wavelet Transform (CWT): {self.current_key} (Dim {dim_idx})"
        )

        self.canvas.draw()


class SwingPlaneTab(QtWidgets.QWidget):
    """Tab for Swing Plane Analysis (3D)."""

    def __init__(self, recorder: RecorderInterface):
        super().__init__()
        self.recorder = recorder
        self.analyzer = SwingPlaneAnalyzer()

        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = MplCanvas(width=5, height=6, dpi=100)
        # We need two subplots: 3D view and Deviation plot
        # But MplCanvas init creates one figure.
        # We can clear and add subplots dynamically.
        layout.addWidget(self.canvas)

        self.update_plot()

    def update_plot(self) -> None:
        self.canvas.fig.clear()
        ax3d = self.canvas.fig.add_subplot(211, projection="3d")
        ax_dev = self.canvas.fig.add_subplot(212)

        times, raw_pos = self.recorder.get_time_series("club_head_position")
        pos: np.ndarray | None = np.array(raw_pos) if raw_pos is not None else None

        if pos is None or len(times) == 0 or pos.ndim != 2 or pos.shape[1] != 3:
            ax3d.text2D(0.5, 0.5, "No 3D Club Head Data", transform=ax3d.transAxes)  # type: ignore[attr-defined]
            self.canvas.draw()
            return

        # Perform Analysis
        try:
            metrics = self.analyzer.analyze(pos)
        except ValueError:
            ax3d.text2D(0.5, 0.5, "Insufficient points", transform=ax3d.transAxes)  # type: ignore[attr-defined]
            self.canvas.draw()
            return

        # 1. 3D Plot
        ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=times, cmap="viridis")
        ax3d.set_title("Club Head Trajectory & Fitted Plane")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")  # type: ignore[attr-defined]

        # Draw Plane
        # Generate grid around centroid
        centroid = metrics.point_on_plane
        normal = metrics.normal_vector
        # Create a grid of points
        d = -centroid.dot(normal)
        # Plane eq: ax + by + cz + d = 0 => z = (-d - ax - by) / c
        # Handle c close to 0
        xlim = ax3d.get_xlim()
        ylim = ax3d.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10)
        )

        if abs(normal[2]) > 0.001:
            zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
            ax3d.plot_surface(xx, yy, zz, alpha=0.2, color="blue")  # type: ignore[attr-defined]
        else:
            # Vertical plane, maybe just skip drawing surface or handle differently
            pass

        # 2. Deviation Plot
        centroid, normal = self.analyzer.fit_plane(pos)
        deviations = self.analyzer.calculate_deviation(pos, centroid, normal)
        ax_dev.plot(times, deviations)
        ax_dev.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax_dev.set_xlabel("Time [s]")
        ax_dev.set_ylabel("Deviation from Plane [m]")
        ax_dev.set_title("Swing Plane Deviation")
        ax_dev.grid(True)

        # Add text box with metrics
        textstr = "\n".join(
            (
                f"Steepness: {metrics.steepness_deg:.1f}°",
                f"Direction: {metrics.direction_deg:.1f}°",
                f"RMSE: {metrics.rmse * 100:.1f} cm",
                f"Max Dev: {metrics.max_deviation * 100:.1f} cm",
            )
        )
        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
        ax_dev.text(
            0.05,
            0.95,
            textstr,
            transform=ax_dev.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class CorrelationTab(QtWidgets.QWidget):
    """Tab for Correlation Heatmap of scalar metrics."""

    def __init__(self, recorder: RecorderInterface):
        super().__init__()
        self.recorder = recorder

        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = MplCanvas(width=6, height=5, dpi=100)
        self.ax = self.canvas.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        # Refresh button (since it computes a lot)
        btn_refresh = QtWidgets.QPushButton("Refresh Correlation Matrix")
        btn_refresh.clicked.connect(self.update_plot)
        layout.addWidget(btn_refresh)

        # Do not update automatically on init to save time, or do it once
        # self.update_plot()

    def update_plot(self) -> None:
        self.ax.clear()

        # Define metrics to correlate
        # We need scalar time series. For vectors, we take norm or specific components.
        metrics_map: dict[str, str] = {
            "Club Speed": "club_head_speed",
            "Kinetic Energy": "kinetic_energy",
            "Total Energy": "total_energy",
            "Left Foot Force": "left_foot_force",
            "Right Foot Force": "right_foot_force",
            "Control Accel": "control_accel",
        }

        data_dict = {}
        min_len = float("inf")

        for label, key in metrics_map.items():
            _, raw_vals = self.recorder.get_time_series(key)
            vals: np.ndarray | None = (
                np.array(raw_vals) if raw_vals is not None else None
            )

            if vals is None or len(vals) == 0:
                continue

            # Reduce to scalar
            if vals.ndim > 1:
                # Take L2 norm for vectors (forces, etc.)
                # Assuming shape (N, D)
                scalar_vals = np.linalg.norm(vals, axis=1)
            else:
                scalar_vals = vals

            data_dict[label] = scalar_vals
            if len(scalar_vals) < min_len:
                min_len = len(scalar_vals)

        if len(data_dict) < 2:
            self.ax.text(0.5, 0.5, "Not enough data for correlation", ha="center")
            self.canvas.draw()
            return

        # Stack into matrix (N_samples, N_features)
        feature_names = list(data_dict.keys())
        matrix_list = []
        for name in feature_names:
            matrix_list.append(data_dict[name][: int(min_len)])

        X = np.column_stack(matrix_list)  # (N, F)

        # Compute Correlation Matrix (F, F)
        # Handle constant columns (std=0) to avoid NaNs
        try:
            corr_mat = np.corrcoef(X, rowvar=False)
        except Exception:
            self.ax.text(0.5, 0.5, "Computation Error", ha="center")
            self.canvas.draw()
            return

        # Plot Heatmap
        im = self.ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
        self.ax.set_xticks(np.arange(len(feature_names)))
        self.ax.set_yticks(np.arange(len(feature_names)))
        self.ax.set_xticklabels(feature_names, rotation=45, ha="right")
        self.ax.set_yticklabels(feature_names)
        self.ax.set_title("Correlation Matrix")

        # Add colorbar
        self.canvas.fig.colorbar(im, ax=self.ax)

        # Annotate
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                self.ax.text(
                    j,
                    i,
                    f"{corr_mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        self.canvas.fig.tight_layout()
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
        t_pos, raw_pos = self.recorder.get_time_series("joint_positions")
        t_vel, raw_vel = self.recorder.get_time_series("joint_velocities")

        # Helper to convert list to array
        pos: np.ndarray | None
        vel: np.ndarray | None
        if isinstance(raw_pos, list):
            pos = np.array(raw_pos) if raw_pos else None
        else:
            pos = raw_pos
        if isinstance(raw_vel, list):
            vel = np.array(raw_vel) if raw_vel else None
        else:
            vel = raw_vel

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
        if not _validate_dimension_index(dim_idx, pos, vel):
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

        t1, raw_d1 = self.recorder.get_time_series(key1)
        t2, raw_d2 = self.recorder.get_time_series(key2)

        d1: np.ndarray | None
        d2: np.ndarray | None
        if isinstance(raw_d1, list):
            d1 = np.array(raw_d1) if raw_d1 else None
        else:
            d1 = raw_d1
        if isinstance(raw_d2, list):
            d2 = np.array(raw_d2) if raw_d2 else None
        else:
            d2 = raw_d2

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
        if not _validate_dimension_index(dim_idx, d1, d2):
            self.ax.text(0.5, 0.5, "Dimension out of bounds", ha="center", va="center")
            self.canvas.draw()
            return

        x = d1[:, dim_idx]
        y = d2[:, dim_idx]

        if len(t1) > 1:
            fs = float(1.0 / np.mean(np.diff(t1)))
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
        self.resize(1000, 800)  # Increased size for more tabs

        self.recorder = recorder

        layout = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()

        # Spectrogram
        self.tab_spectrogram = SpectrogramTab(recorder, initial_key=current_key)
        self.tabs.addTab(self.tab_spectrogram, "Spectrogram")

        # Wavelet (New)
        self.tab_wavelet = WaveletTab(recorder, initial_key=current_key)
        self.tabs.addTab(self.tab_wavelet, "Wavelet Analysis")

        # Phase Plane
        self.tab_phase = PhasePlaneTab(recorder)
        self.tabs.addTab(self.tab_phase, "Phase Plane")

        # Coherence
        initial_key2 = comparison_key if comparison_key else "joint_torques"
        self.tab_coherence = CoherenceTab(recorder, key1=current_key, key2=initial_key2)
        self.tabs.addTab(self.tab_coherence, "Coherence")

        # Swing Plane (New)
        self.tab_swing_plane = SwingPlaneTab(recorder)
        self.tabs.addTab(self.tab_swing_plane, "Swing Plane")

        # Correlation (New)
        self.tab_correlation = CorrelationTab(recorder)
        self.tabs.addTab(self.tab_correlation, "Correlation Heatmap")

        layout.addWidget(self.tabs)

        # Close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
