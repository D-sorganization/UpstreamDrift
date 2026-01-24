"""Widgets for the Unified Dashboard.

Contains:
- LivePlotWidget: Real-time plotting of simulation data.
- ControlPanel: Playback controls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.dashboard.advanced_analysis import AdvancedAnalysisDialog
from src.shared.python.export import export_recording_all_formats
from src.shared.python.interfaces import RecorderInterface
from src.shared.python.logging_config import get_logger
from src.shared.python.plotting import MplCanvas
from src.shared.python.signal_processing import compute_psd

logger = get_logger(__name__)

MAX_DIMENSIONS = 100

# Numerical constants for signal processing
LOG_EPSILON = 1e-12  # Small epsilon to avoid log(0) in dB calculations
DB_CONVERSION = 10  # Decibel conversion factor: dB = 10 * log10(power)


class FrequencyAnalysisDialog(QtWidgets.QDialog):
    """Dialog for frequency domain analysis (PSD)."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        data: np.ndarray | None = None,
        fs: float = 100.0,
        label: str = "Data",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Frequency Analysis - {label}")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        if data is None or len(data) == 0:
            lbl = QtWidgets.QLabel("No data available.")
            layout.addWidget(lbl)
            return

        # Handle dimensions: (Time, Dims)
        # compute_psd expects (Dims, Time) or we handle the output.
        # welch defaults to axis=-1.
        data_t = data.T
        if data_t.ndim == 1:
            data_t = data_t.reshape(1, -1)

        # Compute PSD
        freqs, psd = compute_psd(data_t, fs=fs)
        # psd is (Dims, Freqs)

        self.ax = self.canvas.fig.add_subplot(111)
        self.ax.set_title(f"PSD: {label}")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Power Spectral Density (dB/Hz)")
        self.ax.grid(True)

        n_dims = psd.shape[0]
        # Limit number of lines to avoid clutter
        limit_dims = min(n_dims, 10)

        for i in range(limit_dims):
            # Convert to dB (using named constants for clarity)
            # LOG_EPSILON avoids log(0), DB_CONVERSION = 10 for power to dB
            psd_db = DB_CONVERSION * np.log10(psd[i] + LOG_EPSILON)
            label_str = f"Dim {i}" if n_dims > 1 else "PSD"
            self.ax.semilogx(freqs, psd_db, label=label_str)

        if n_dims > 1:
            self.ax.legend()

        self.canvas.draw()


class LivePlotWidget(QtWidgets.QWidget):
    """Widget for displaying real-time plots of simulation data.

    Supports plotting multiple metrics including:
    - Kinematics (Positions, Velocities)
    - Dynamics (Torques)
    - Advanced (ZTCF, Induced Accelerations)
    - Parametric (X-Y) plots
    - Frequency Analysis
    """

    def __init__(self, recorder: RecorderInterface) -> None:
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
        self.ax2: Axes | None = None  # Secondary axis
        self.line_objects: list[Any] = []
        self.line_objects2: list[Any] = []

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

        # Comparison metrics
        self.comparison_key: str | None = None
        self.comparison_label: str | None = None

        # Stats Labels
        self.lbl_stats = QtWidgets.QLabel(
            "Mean: 0.00 | Std: 0.00 | Min: 0.00 | Max: 0.00"
        )
        self.lbl_stats.setStyleSheet("font-family: monospace;")
        self._main_layout.addWidget(self.lbl_stats)

        # Controls Layout
        controls_layout = QtWidgets.QHBoxLayout()

        # Selector for data type
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(list(self.metric_options.keys()))
        self.combo.setToolTip("Select primary data to plot")
        self.combo.setStatusTip("Select the simulation metric to plot in real-time")
        self.combo.setAccessibleName("Metric Selector")
        self.combo.currentTextChanged.connect(self.set_plot_metric)

        lbl_metric = QtWidgets.QLabel("Metric:")
        lbl_metric.setBuddy(self.combo)
        controls_layout.addWidget(lbl_metric)
        controls_layout.addWidget(self.combo)

        # Comparison Selector
        self.chk_compare = QtWidgets.QCheckBox("Compare:")
        self.chk_compare.setToolTip("Enable comparison with another metric")
        self.chk_compare.stateChanged.connect(self._toggle_comparison)
        controls_layout.addWidget(self.chk_compare)

        self.combo_compare = QtWidgets.QComboBox()
        self.combo_compare.addItems(list(self.metric_options.keys()))
        self.combo_compare.setToolTip("Select secondary metric to compare")
        self.combo_compare.setEnabled(False)
        self.combo_compare.currentTextChanged.connect(self._set_comparison_metric)
        controls_layout.addWidget(self.combo_compare)

        # X-Y Plot Mode (Parametric)
        self.chk_xy = QtWidgets.QCheckBox("X-Y Plot")
        self.chk_xy.setToolTip("Plot Primary Metric (X) vs Secondary Metric (Y)")
        self.chk_xy.stateChanged.connect(self._toggle_xy_mode)
        controls_layout.addWidget(self.chk_xy)

        # Plot Mode Selector
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["All Dimensions", "Single Dimension", "Norm"])
        self.mode_combo.setToolTip("Select plot mode")
        self.mode_combo.setStatusTip(
            "Select how to display the data (All Dimensions, Single Dimension, or Norm)"
        )
        self.mode_combo.setAccessibleName("Plot Mode Selector")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        controls_layout.addWidget(self.mode_combo)

        # Selector for Dimension Index (Hidden by default)
        self.dim_spin = QtWidgets.QSpinBox()
        self.dim_spin.setAccessibleName("Dimension Index")
        self.dim_spin.setAccessibleDescription("Select the dimension index to plot")
        self.dim_spin.setRange(0, MAX_DIMENSIONS)
        self.dim_spin.setPrefix("Dim: ")
        self.dim_spin.setToolTip("Select dimension index")
        self.dim_spin.setStatusTip("Select the specific dimension index to plot")
        self.dim_spin.setAccessibleName("Dimension Index")
        self.dim_spin.setVisible(False)  # Only for Single Dimension mode
        self.dim_spin.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.dim_spin)

        # Checkbox for enabling computation (if expensive)
        # Initialize early to avoid AttributeError during signal callbacks
        self.chk_compute = QtWidgets.QCheckBox("Compute Real-time")
        self.chk_compute.setToolTip(
            "Enable real-time computation for advanced metrics (ZTCF, etc). May affect performance."
        )
        self.chk_compute.setStatusTip(
            "Enable or disable real-time computation of advanced metrics"
        )
        self.chk_compute.stateChanged.connect(self.toggle_computation)

        # Selector for Induced Accel Source (Hidden by default)
        # Using a ComboBox for user-friendly name selection
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.setAccessibleName("Source Selector")
        self.source_combo.setAccessibleDescription(
            "Select the joint torque source to analyze for induced acceleration"
        )
        self.source_combo.setToolTip("Select the joint torque source to analyze.")
        self.source_combo.setStatusTip("Select the induced acceleration source")
        self.source_combo.setVisible(False)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        # Populate with joint names if available, or indices
        self._populate_source_combo()

        lbl_source = QtWidgets.QLabel("Source:")
        lbl_source.setBuddy(self.source_combo)
        lbl_source.setVisible(False)
        self.source_label = lbl_source

        controls_layout.addWidget(lbl_source)
        controls_layout.addWidget(self.source_combo)
        # Snapshot Button
        self.btn_snapshot = QtWidgets.QPushButton("Snapshot")
        self.btn_snapshot.setToolTip("Copy current plot to clipboard")
        self.btn_snapshot.setStatusTip(
            "Capture the current plot image to the clipboard"
        )
        self.btn_snapshot.setAccessibleName("Snapshot Button")
        self.btn_snapshot.clicked.connect(self.copy_snapshot)
        controls_layout.addWidget(self.btn_snapshot)

        # Frequency Analysis Button
        self.btn_freq = QtWidgets.QPushButton("Freq Analysis")
        self.btn_freq.setToolTip("Show Power Spectral Density (PSD) of current view")
        self.btn_freq.clicked.connect(self.show_freq_analysis)
        controls_layout.addWidget(self.btn_freq)

        # Advanced Analysis Button
        self.btn_advanced = QtWidgets.QPushButton("Advanced...")
        self.btn_advanced.setToolTip(
            "Show Advanced Analysis Tools (Spectrogram, Phase Plane, Coherence)"
        )
        self.btn_advanced.clicked.connect(self.show_advanced_analysis)
        controls_layout.addWidget(self.btn_advanced)

        # Export Button
        self.btn_export = QtWidgets.QPushButton("Export Data")
        self.btn_export.setToolTip("Export recording to CSV/JSON/MAT")
        self.btn_export.setStatusTip("Export the full recorded data set")
        self.btn_export.clicked.connect(self.export_data)
        controls_layout.addWidget(self.btn_export)

        controls_layout.addWidget(self.chk_compute)

        self._main_layout.addLayout(controls_layout)

        # Initial plot setup
        self.ax.set_title("Live Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.grid(True)

    def _populate_source_combo(self) -> None:
        """Populate the source combo box with joint names or indices."""
        self.source_combo.clear()

        # Try to get joint names from the engine
        joint_names = []
        if hasattr(self.recorder, "engine") and hasattr(
            self.recorder.engine, "get_joint_names"
        ):
            try:
                joint_names = self.recorder.engine.get_joint_names()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to get joint names from engine")

        if joint_names:
            self.source_combo.addItems(joint_names)
        else:
            # Fallback to generic indices
            for i in range(MAX_DIMENSIONS):
                self.source_combo.addItem(f"Source {i}")

    def set_joint_names(self, names: list[str]) -> None:
        """Update source selector with human-readable joint names."""
        if not names:
            return

        current_idx = self.source_combo.currentIndex()
        self.source_combo.clear()
        self.source_combo.addItems(names)

        # Restore index if valid, otherwise 0
        if 0 <= current_idx < len(names):
            self.source_combo.setCurrentIndex(current_idx)
        else:
            self.source_combo.setCurrentIndex(0)

    def set_plot_metric(self, label: str) -> None:
        """Change the metric being plotted."""
        self.current_label = label
        self.current_key = self.metric_options[label]

        # Show/Hide Source Selector
        self._update_source_visibility()

        # Update recorder config if needed
        self._update_recorder_config()

        # Reset plot
        self._reset_plot()
        self.update_plot()

    def _toggle_comparison(self, state: int) -> None:
        """Enable or disable comparison mode."""
        enabled = self.chk_compare.isChecked()
        self.combo_compare.setEnabled(enabled)
        if enabled:
            label = self.combo_compare.currentText()
            self.comparison_label = label
            self.comparison_key = self.metric_options[label]
        else:
            self.comparison_key = None
            self.comparison_label = None

        self._update_source_visibility()
        self._update_recorder_config()
        self._reset_plot()
        self.update_plot()

    def _set_comparison_metric(self, label: str) -> None:
        """Set the comparison metric."""
        if self.chk_compare.isChecked():
            self.comparison_label = label
            self.comparison_key = self.metric_options[label]
            self._update_source_visibility()
            self._update_recorder_config()
            self._reset_plot()
            self.update_plot()

    def _update_source_visibility(self) -> None:
        """Show source combo if any active metric requires it."""
        needs_source = (
            self.current_key == "induced_accel_source"
            or self.comparison_key == "induced_accel_source"
        )
        self.source_combo.setVisible(needs_source)
        self.source_label.setVisible(needs_source)

        # Refresh sources if becoming visible
        if needs_source and self.source_combo.count() == 0:
            self._populate_source_combo()

    def _on_source_changed(self, index: int) -> None:
        """Handle source selection change."""
        self._update_recorder_config()
        self._reset_plot()
        self.update_plot()

    def _toggle_xy_mode(self, state: int) -> None:
        """Handle X-Y mode toggle."""
        is_xy = self.chk_xy.isChecked()
        if is_xy and not self.chk_compare.isChecked():
            # Force enable comparison if not already enabled
            self.chk_compare.setChecked(True)
        self._reset_plot()
        self.update_plot()

    def _on_mode_changed(self, mode: str) -> None:
        """Handle plot mode change."""
        self.dim_spin.setVisible(mode == "Single Dimension")
        self._reset_plot()
        self.update_plot()

    def toggle_computation(self, state: int) -> None:
        """Handle checkbox toggle."""
        self._update_recorder_config()

    def _update_recorder_config(self) -> None:
        """Update recorder configuration based on current selection and checkbox."""
        compute_enabled = self.chk_compute.isChecked()
        config: dict[str, Any] = {
            "ztcf": False,
            "zvcf": False,
            "track_drift": False,
            "track_total_control": False,
            "induced_accel_sources": [],
        }

        keys_to_check = [self.current_key]
        if self.comparison_key:
            keys_to_check.append(self.comparison_key)

        if compute_enabled:
            for key in keys_to_check:
                if key == "ztcf_accel":
                    config["ztcf"] = True
                elif key == "zvcf_accel":
                    config["zvcf"] = True
                elif key == "drift_accel":
                    config["track_drift"] = True
                elif key == "control_accel":
                    config["track_total_control"] = True
                elif key == "induced_accel_source":
                    # Add the selected source identifier (name if possible, else index)
                    source_val: str | int = self.source_combo.currentIndex()
                    txt = self.source_combo.currentText()
                    # If text looks like a name (not just 'Source X'), use it
                    if txt and not txt.startswith("Source "):
                        source_val = txt

                    if source_val not in config["induced_accel_sources"]:
                        config["induced_accel_sources"].append(source_val)

        self.recorder.set_analysis_config(config)

    def _reset_plot(self) -> None:
        """Clear the axes and lines."""
        self.ax.clear()

        # Always remove secondary axis first to avoid stacking
        if self.ax2:
            self.ax2.remove()
            self.ax2 = None

        self.line_objects = []
        self.line_objects2 = []

        title = f"Live {self.current_label}"

        is_xy = self.chk_xy.isChecked()

        if is_xy:
            if self.comparison_label:
                title = f"{self.current_label} (X) vs {self.comparison_label} (Y)"
                self.ax.set_xlabel(self.current_label)
                self.ax.set_ylabel(self.comparison_label)
            else:
                title += " (X-Y Mode - Select Comparison)"
                self.ax.set_xlabel(self.current_label)
        else:
            if self.comparison_label:
                title += f" vs {self.comparison_label}"
                # Create secondary axis
                self.ax2 = self.ax.twinx()
            self.ax.set_xlabel("Time (s)")

        self.ax.set_title(title)
        self.ax.grid(True)

    def _get_data_for_key(self, key: str) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Fetch data for a specific key."""
        times: np.ndarray = np.array([])
        data: np.ndarray | None = None
        dim_label = "Dim"

        if key == "induced_accel_source":
            # Fetch specific induced acceleration
            src_val: str | int = self.source_combo.currentIndex()
            txt = self.source_combo.currentText()
            # Prefer name if available and not generic
            if txt and not txt.startswith("Source "):
                src_val = txt

            # Use specific interface
            times, data = self.recorder.get_induced_acceleration_series(src_val)
        else:
            # Standard metric
            times, data_raw = self.recorder.get_time_series(key)
            # Convert to ndarray if it's a list
            if isinstance(data_raw, list):
                data = np.array(data_raw) if data_raw else None
            else:
                data = data_raw

        if len(times) == 0 or data is None:
            return np.array([]), None, dim_label

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Apply filtering based on Plot Mode
        plot_mode = self.mode_combo.currentText()

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

        return times, data, dim_label

    def update_plot(self) -> None:
        """Update the plot with the latest data."""
        times, data, dim_label = self._get_data_for_key(self.current_key)

        if len(times) == 0 or data is None:
            # Update stats even if empty?
            return

        # Limit to recent history to keep it fast
        max_points = 500
        if len(times) > max_points:
            times = times[-max_points:]
            data = data[-max_points:]

        # Update Statistics
        if data.size > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            self.lbl_stats.setText(
                f"Mean: {mean_val:.2f} | Std: {std_val:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}"
            )

        n_dims = data.shape[1]
        plot_mode = self.mode_combo.currentText()
        is_xy = self.chk_xy.isChecked()

        # Handle X-Y Plot Mode
        if is_xy:
            if self.comparison_key:
                times2, data2, dim_label2 = self._get_data_for_key(self.comparison_key)
                if len(times2) > 0 and data2 is not None:
                    # Sync lengths
                    if len(times2) > max_points:
                        times2 = times2[-max_points:]
                        data2 = data2[-max_points:]

                    min_len = min(len(data), len(data2))
                    data = data[:min_len]
                    data2 = data2[:min_len]

                    # Initialize lines if needed
                    # If dimensions mismatch, we can only plot matching dimensions
                    n_common_dims = min(n_dims, data2.shape[1])

                    if len(self.line_objects) != n_common_dims:
                        self.ax.clear()
                        self.line_objects = []
                        # No secondary axis in XY mode (same plot)

                        title = (
                            f"{self.current_label} (X) vs {self.comparison_label} (Y)"
                        )
                        self.ax.set_title(title)
                        self.ax.set_xlabel(self.current_label or "")
                        self.ax.set_ylabel(self.comparison_label or "")
                        self.ax.grid(True)

                        for i in range(n_common_dims):
                            label = f"Dim {i}" if n_common_dims > 1 else "Trajectory"
                            if plot_mode == "Single Dimension":
                                label = f"Dim {dim_label} vs {dim_label2}"
                            elif plot_mode == "Norm":
                                label = "Norm vs Norm"

                            (line,) = self.ax.plot([], [], label=label)
                            self.line_objects.append(line)

                        if n_common_dims < 10 or plot_mode != "All Dimensions":
                            self.ax.legend(loc="upper right")

                    # Update lines
                    for i, line in enumerate(self.line_objects):
                        line.set_data(data[:, i], data2[:, i])

                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.canvas.draw()
            return

        # Standard Time Series Plot

        # Initialize lines for primary axis
        if len(self.line_objects) != n_dims:
            # Re-initialize only if dimension count changed
            self.ax.clear()
            self.line_objects = []
            if self.ax2:
                if self.ax2:
                    self.ax2.clear()
                self.line_objects2 = []

            # Re-setup
            title = f"Live {self.current_label}"
            if self.comparison_label:
                title += f" vs {self.comparison_label}"
            self.ax.set_title(title)
            self.ax.set_xlabel("Time (s)")
            self.ax.grid(True)

            for i in range(n_dims):
                label = (
                    dim_label
                    if n_dims == 1
                    else f"{dim_label} {i}" if plot_mode != "Norm" else "Norm"
                )
                if plot_mode == "All Dimensions":
                    label = f"Dim {i}"

                # Primary color cycle
                (line,) = self.ax.plot([], [], label=label)
                self.line_objects.append(line)

            if n_dims < 10 or plot_mode != "All Dimensions":
                self.ax.legend(loc="upper left")

        # Update primary lines
        for i, line in enumerate(self.line_objects):
            if i < data.shape[1]:
                line.set_data(times, data[:, i])
                if plot_mode == "Single Dimension":
                    line.set_label(dim_label)

        self.ax.relim()
        self.ax.autoscale_view()

        # Handle Comparison
        if self.comparison_key and self.ax2:
            times2, data2, dim_label2 = self._get_data_for_key(self.comparison_key)
            if len(times2) > 0 and data2 is not None:
                # Sync lengths
                if len(times2) > max_points:
                    times2 = times2[-max_points:]
                    data2 = data2[-max_points:]

                n_dims2 = data2.shape[1]

                # Init secondary lines
                if len(self.line_objects2) != n_dims2:
                    self.ax2.clear()
                    self.line_objects2 = []
                    for i in range(n_dims2):
                        label2 = (
                            f"{self.comparison_label} {i}"
                            if n_dims2 > 1
                            else self.comparison_label
                        )
                        if plot_mode == "Single Dimension":
                            label2 = f"{self.comparison_label} {dim_label2}"
                        elif plot_mode == "Norm":
                            label2 = f"{self.comparison_label} Norm"

                        # Secondary style (dashed)
                        (line,) = self.ax2.plot(
                            [], [], label=label2, linestyle="--", alpha=0.7
                        )
                        self.line_objects2.append(line)

                    self.ax2.legend(loc="upper right")

                # Update secondary lines
                for i, line in enumerate(self.line_objects2):
                    if i < data2.shape[1]:
                        line.set_data(times2, data2[:, i])

                self.ax2.relim()
                self.ax2.autoscale_view()

        self.canvas.draw()

    def show_freq_analysis(self) -> None:
        """Show Frequency Analysis Dialog."""
        times, data, _ = self._get_data_for_key(self.current_key)
        if len(times) == 0:
            return

        # Estimate fs
        if len(times) > 1:
            fs: float = float(1.0 / np.mean(np.diff(times)))
        else:
            fs = 100.0

        dlg = FrequencyAnalysisDialog(self, data, fs, self.current_label)
        dlg.exec()

    def show_advanced_analysis(self) -> None:
        """Show Advanced Analysis Dialog."""
        dlg = AdvancedAnalysisDialog(
            self,
            self.recorder,
            current_key=self.current_key,
            comparison_key=self.comparison_key,
        )
        dlg.exec()

    def copy_snapshot(self) -> None:
        """Capture the current plot and copy to clipboard."""
        # Grab the canvas content
        pixmap = self.canvas.grab()

        # Copy to clipboard
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard:
            clipboard.setPixmap(pixmap)

        # Visual feedback
        self.btn_snapshot.setText("Copied!")
        self.btn_snapshot.setEnabled(False)

        # Restore after 2 seconds
        QtCore.QTimer.singleShot(2000, self._restore_snapshot_btn)

    def _restore_snapshot_btn(self) -> None:
        """Restore snapshot button state."""
        self.btn_snapshot.setText("Snapshot")
        self.btn_snapshot.setEnabled(True)

    def export_data(self) -> None:
        """Export recorded data."""
        # Try to get data dictionary
        data_dict = {}
        if hasattr(self.recorder, "export_to_dict"):
            data_dict = self.recorder.export_to_dict()  # type: ignore
        elif hasattr(self.recorder, "frames"):
            # Try to build minimal dict from frames if accessible
            # This is specific to some recorder implementations, risky but helpful fallback
            pass

        if not data_dict:
            # Fallback: Scrape metric options?
            # Or just warn
            QtWidgets.QMessageBox.warning(
                self,
                "Export Warning",
                "Recorder does not support full export. Exporting only current plot data is not yet implemented.",
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", "swing_data", "All Files (*)"
        )
        if not filename:
            return

        results = export_recording_all_formats(filename, data_dict)
        msg = "Export Results:\n"
        for fmt, success in results.items():
            msg += f"{fmt}: {'Success' if success else 'Failed'}\n"

        QtWidgets.QMessageBox.information(self, "Export Complete", msg)


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
