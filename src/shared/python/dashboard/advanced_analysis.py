# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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

from collections.abc import Callable
from typing import Any

import numpy as np
from PyQt6 import QtWidgets

from src.shared.python.biomechanics.swing_plane_analysis import SwingPlaneAnalyzer
from src.shared.python.dashboard._analysis_refresh import (
    BoundedResultCache,
    DebouncedRefresh,
    analysis_cache_key,
)
from src.shared.python.engine_core.interfaces import RecorderInterface
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plotting import MplCanvas
from src.shared.python.signal_toolkit.signal_processing import (
    compute_coherence,
    compute_cwt,
    compute_spectrogram,
)

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


def _estimate_fs(times: Any) -> float:
    """Sampling rate from the mean timestep, defaulting to 100 Hz."""
    return float(1.0 / np.mean(np.diff(times))) if len(times) > 1 else 100.0


class _SignalTransformTab(QtWidgets.QWidget):
    """Shared scaffold for the per-signal transform tabs (spectrogram, CWT).

    Owns the metric selector, the debounced dimension spinbox, the plot axes
    and a bounded memo of transform results (#8932). Subclasses add controls
    in ``_add_extra_controls`` and implement ``update_plot``.
    """

    # Provided by every concrete subclass; redraws the selected transform.
    update_plot: Callable[[], None]

    def __init__(
        self,
        recorder: RecorderInterface,
        initial_key: str,
        metric_options: dict[str, str],
    ) -> None:
        if recorder is None:
            raise ValueError("recorder must be provided")
        super().__init__()
        self.recorder = recorder
        self.current_key = initial_key
        self.metric_options = metric_options
        self._refresh = DebouncedRefresh(self.update_plot, parent=self)
        self._cache = BoundedResultCache()

        layout = QtWidgets.QVBoxLayout(self)
        controls_layout = QtWidgets.QHBoxLayout()
        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems(list(self.metric_options.keys()))
        for label, key in self.metric_options.items():
            if key == self.current_key:
                self.combo_metric.setCurrentText(label)
                break
        self.combo_metric.currentTextChanged.connect(self._on_metric_changed)
        controls_layout.addWidget(QtWidgets.QLabel("Metric:"))
        controls_layout.addWidget(self.combo_metric)

        self.spin_dim = QtWidgets.QSpinBox()
        self.spin_dim.setRange(0, 100)
        self._add_debounced_spinbox(controls_layout, self.spin_dim, "Dim: ")
        self._add_extra_controls(controls_layout)
        layout.addLayout(controls_layout)

        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self.ax = self.canvas.fig.add_subplot(111)
        layout.addWidget(self.canvas)

        self.update_plot()

    def _add_extra_controls(self, controls_layout: QtWidgets.QHBoxLayout) -> None:
        """Hook for subclass-specific controls; none by default."""

    def _add_debounced_spinbox(
        self,
        controls_layout: QtWidgets.QHBoxLayout,
        spin: QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox,
        prefix: str,
    ) -> None:
        """Route an already-configured spinbox through the shared debounce."""
        spin.setPrefix(prefix)
        spin.valueChanged.connect(self._refresh.trigger)
        controls_layout.addWidget(spin)

    def _on_metric_changed(self, label: str) -> None:
        self.current_key = self.metric_options[label]
        self.update_plot()

    def _begin_plot(self) -> tuple[Any, np.ndarray | None]:
        """Clear the axes and load the selected metric.

        Returns ``(times, data)`` with ``data`` shaped ``(N, D)``. When there is
        no data, a "No Data" message is drawn and ``data`` is ``None``.
        """
        self.ax.clear()
        times, raw_data = self.recorder.get_time_series(self.current_key)
        if raw_data is None or len(raw_data) == 0 or len(times) == 0:
            self._show_message("No Data")
            return times, None
        data = np.asarray(raw_data)
        return times, data.reshape(-1, 1) if data.ndim == 1 else data

    def _show_message(self, message: str) -> None:
        self.ax.text(0.5, 0.5, message, ha="center", va="center")
        self.canvas.draw_idle()

    def _memoized(
        self,
        dim_idx: int,
        fs: float,
        signal: np.ndarray,
        compute: Callable[[], Any],
        **params: float,
    ) -> Any:
        """Return ``compute()`` memoized on the metric, dim, fs, params and data."""
        key = analysis_cache_key(self.current_key, dim_idx, fs, signal, **params)
        return self._cache.get_or_compute(key, compute)


class SpectrogramTab(_SignalTransformTab):
    """Tab for Spectrogram Analysis."""

    def __init__(
        self, recorder: RecorderInterface, initial_key: str = "joint_positions"
    ) -> None:
        super().__init__(
            recorder,
            initial_key,
            {
                "Joint Positions": "joint_positions",
                "Joint Velocities": "joint_velocities",
                "Joint Torques": "joint_torques",
                "Ground Forces": "ground_forces",
                "Club Head Speed": "club_head_speed",
                "Total Control Accel": "control_accel",
            },
        )

    def update_plot(self) -> None:
        """Redraw the spectrogram for the selected metric and dimension."""
        times, data = self._begin_plot()
        if data is None:
            return

        dim_idx = self.spin_dim.value()
        if not _validate_dimension_index(dim_idx, data):
            logger.warning(
                "Dimension index %d out of bounds for data shape %s, falling back to 0",
                dim_idx,
                data.shape,
            )
            dim_idx = 0  # Fallback to first dimension if invalid

        signal_data = data[:, dim_idx]
        fs = _estimate_fs(times)
        f, t, Sxx = self._memoized(
            dim_idx, fs, signal_data, lambda: compute_spectrogram(signal_data, fs)
        )

        self.ax.pcolormesh(
            t, f, DB_CONVERSION * np.log10(Sxx + LOG_EPSILON), shading="gouraud"
        )
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_xlabel("Time [sec]")
        self.ax.set_title(f"Spectrogram: {self.current_key} (Dim {dim_idx})")
        self.canvas.draw_idle()


class WaveletTab(_SignalTransformTab):
    """Tab for Continuous Wavelet Transform (CWT) Analysis."""

    def __init__(
        self, recorder: RecorderInterface, initial_key: str = "joint_velocities"
    ) -> None:
        super().__init__(
            recorder,
            initial_key,
            {
                "Joint Positions": "joint_positions",
                "Joint Velocities": "joint_velocities",
                "Joint Torques": "joint_torques",
                "Club Head Speed": "club_head_speed",
                "Total Control Accel": "control_accel",
            },
        )

    def _add_extra_controls(self, controls_layout: QtWidgets.QHBoxLayout) -> None:
        self.spin_w0 = QtWidgets.QDoubleSpinBox()
        self.spin_w0.setRange(2.0, 20.0)
        self.spin_w0.setValue(6.0)
        self.spin_w0.setSingleStep(0.5)
        self._add_debounced_spinbox(controls_layout, self.spin_w0, "w0: ")

    def update_plot(self) -> None:
        """Redraw the continuous wavelet transform plot for the selected metric."""
        times, data = self._begin_plot()
        if data is None:
            return

        dim_idx = self.spin_dim.value()
        if not _validate_dimension_index(dim_idx, data):
            self._show_message("Dimension out of bounds")
            return

        signal_data = data[:, dim_idx]
        fs = _estimate_fs(times)
        w0 = self.spin_w0.value()
        freqs, t, cwt_mat = self._memoized(
            dim_idx,
            fs,
            signal_data,
            lambda: compute_cwt(signal_data, fs, w0=w0, num_freqs=64),
            w0=w0,
        )

        mag = np.abs(cwt_mat)
        self.ax.pcolormesh(t, freqs, mag, shading="gouraud", cmap="plasma")
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_xlabel("Time [sec]")
        self.ax.set_yscale("log")
        self.ax.set_title(
            f"Wavelet Transform (CWT): {self.current_key} (Dim {dim_idx})"
        )
        self.canvas.draw_idle()


class SwingPlaneTab(QtWidgets.QWidget):
    """Tab for Swing Plane Analysis (3D).

    The 3D and deviation axes are created once. A refresh only swaps the
    data-bearing artists, so the expensive 3D axes is never rebuilt (#8932).
    """

    def __init__(self, recorder: RecorderInterface) -> None:
        if recorder is None:
            raise ValueError("recorder must be provided")
        super().__init__()
        self.recorder = recorder
        self.analyzer = SwingPlaneAnalyzer()
        self._data_artists: list[Any] = []

        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = MplCanvas(width=5, height=6, dpi=100)
        self.ax3d = self.canvas.fig.add_subplot(211, projection="3d")
        self.ax_dev = self.canvas.fig.add_subplot(212)
        self._init_axes()
        layout.addWidget(self.canvas)

        self.update_plot()

    def _init_axes(self) -> None:
        """Create the static decorations and the reusable artists once."""
        ax3d, ax_dev = self.ax3d, self.ax_dev
        ax3d.set_title("Club Head Trajectory & Fitted Plane")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")  # type: ignore[attr-defined]
        self._status_text = ax3d.text2D(0.5, 0.5, "", transform=ax3d.transAxes)  # type: ignore[attr-defined]

        (self._dev_line,) = ax_dev.plot([], [])
        ax_dev.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax_dev.set_xlabel("Time [s]")
        ax_dev.set_ylabel("Deviation from Plane [m]")
        ax_dev.set_title("Swing Plane Deviation")
        ax_dev.grid(True)
        props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
        self._metrics_text = ax_dev.text(
            0.05,
            0.95,
            "",
            transform=ax_dev.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        self.canvas.fig.tight_layout()

    def _clear_data_artists(self) -> None:
        for artist in self._data_artists:
            artist.remove()
        self._data_artists.clear()

    def _show_status(self, message: str) -> None:
        """Replace the plotted data with a status message."""
        self._clear_data_artists()
        self._status_text.set_text(message)
        self._dev_line.set_data([], [])
        self._metrics_text.set_visible(False)
        self.canvas.draw_idle()

    def update_plot(self) -> None:
        """Redraw the 3D swing plane and deviation plots."""
        times, raw_pos = self.recorder.get_time_series("club_head_position")
        pos: np.ndarray | None = np.asarray(raw_pos) if raw_pos is not None else None

        if pos is None or len(times) == 0 or pos.ndim != 2 or pos.shape[1] != 3:
            self._show_status("No 3D Club Head Data")
            return

        try:
            metrics = self.analyzer.analyze(pos)
        except ValueError:
            self._show_status("Insufficient points")
            return

        self._clear_data_artists()
        self._status_text.set_text("")
        self._draw_trajectory(pos, times, metrics)
        self._draw_deviation(pos, times, metrics)
        self.canvas.draw_idle()

    def _draw_trajectory(self, pos: np.ndarray, times: Any, metrics: Any) -> None:
        """Scatter the club path and overlay the fitted plane on the 3D axes."""
        ax3d: Any = self.ax3d
        self._data_artists.append(
            ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=times, cmap="viridis")
        )

        # Plane eq: ax + by + cz + d = 0 => z = (-d - ax - by) / c
        centroid = metrics.point_on_plane
        normal = metrics.normal_vector
        d = -centroid.dot(normal)
        xlim = ax3d.get_xlim()
        ylim = ax3d.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10)
        )
        # A (near-)vertical plane cannot be expressed as z(x, y); skip it.
        if abs(normal[2]) > 0.001:
            zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
            self._data_artists.append(
                ax3d.plot_surface(xx, yy, zz, alpha=0.2, color="blue")
            )

    def _draw_deviation(self, pos: np.ndarray, times: Any, metrics: Any) -> None:
        """Update the deviation trace and the metrics text box in place."""
        centroid, normal = self.analyzer.fit_plane(pos)
        deviations = self.analyzer.calculate_deviation(pos, centroid, normal)
        self._dev_line.set_data(times, deviations)
        self.ax_dev.relim()
        self.ax_dev.autoscale_view()

        self._metrics_text.set_text(
            "\n".join(
                (
                    f"Steepness: {metrics.steepness_deg:.1f}°",
                    f"Direction: {metrics.direction_deg:.1f}°",
                    f"RMSE: {metrics.rmse * 100:.1f} cm",
                    f"Max Dev: {metrics.max_deviation * 100:.1f} cm",
                )
            )
        )
        self._metrics_text.set_visible(True)


class CorrelationTab(QtWidgets.QWidget):
    """Tab for Correlation Heatmap of scalar metrics."""

    def __init__(self, recorder: RecorderInterface) -> None:
        if recorder is None:
            raise ValueError("recorder must be provided")
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

    def update_plot(self) -> None:
        """Redraw the correlation matrix heatmap across recorded metrics."""
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
            # Take L2 norm for vectors (forces, etc.)
            # Assuming shape (N, D)
            # ⚡ Bolt: Explicit element-wise sum of squares is faster than np.linalg.norm(..., axis=1) for small inner dimensions
            vals_f = vals.astype(float, copy=False)
            scalar_vals = (
                np.sqrt(np.einsum("...i,...i->...", vals_f, vals_f))
                if vals.ndim > 1
                else vals
            )

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
        matrix_list.extend([data_dict[name][: int(min_len)] for name in feature_names])

        X = np.column_stack(matrix_list)  # (N, F)

        # Compute Correlation Matrix (F, F)
        # Handle constant columns (std=0) to avoid NaNs
        try:
            corr_mat = np.corrcoef(X, rowvar=False)
        except (ValueError, TypeError, RuntimeError):
            self.ax.text(0.5, 0.5, "Computation Error", ha="center")
            self.canvas.draw()
            return

        # Plot Heatmap
        im = self.ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax_any: Any = self.ax
        ax_any.set_xticks(np.arange(len(feature_names)))
        ax_any.set_yticks(np.arange(len(feature_names)))
        ax_any.set_xticklabels(feature_names, rotation=45, ha="right")
        ax_any.set_yticklabels(feature_names)
        ax_any.set_title("Correlation Matrix")

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

    def __init__(self, recorder: RecorderInterface) -> None:
        if recorder is None:
            raise ValueError("recorder must be provided")
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
        """Redraw the phase plane plot of position versus velocity."""
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
    ) -> None:
        if recorder is None:
            raise ValueError("recorder must be provided")
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
        """Redraw the coherence plot between two selected signals."""
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

        fs = float(1.0 / np.mean(np.diff(t1))) if len(t1) > 1 else 100.0

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
    ) -> None:
        if recorder is None:
            raise ValueError("recorder must be provided")
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
        initial_key2 = comparison_key or "joint_torques"
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
