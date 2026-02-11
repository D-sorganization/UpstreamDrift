"""Pinocchio analysis and plotting mixin.

Extracts post-hoc analysis tab setup, plot generation, induced/counterfactual
analysis, swing profile, and data export from PinocchioGUI (gui.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plotting import GolfSwingPlotter, MplCanvas
from src.shared.python.validation_pkg.statistical_analysis import StatisticalAnalyzer

logger = get_logger(__name__)


class PinocchioAnalysisMixin:
    """Mixin for Pinocchio GUI analysis, plotting, and data export.

    Provides:
    - ``_setup_analysis_tab``: Post-hoc analysis UI construction
    - ``_generate_plot``: Plot dispatch for all plot types
    - ``_plot_swing_profile``: Radar chart generation
    - ``_plot_induced_accelerations``: Induced acceleration analysis
    - ``_plot_counterfactuals``: Counterfactual comparison plot
    - ``_ensure_analysis_data_populated``: Post-hoc data population
    - ``_export_statistics``: Multi-format data export
    - ``_on_live_analysis_toggled``: Live analysis toggle handler
    """

    def _on_live_analysis_toggled(self: Any, checked: bool) -> None:  # noqa: FBT001
        """Handle live analysis toggle."""
        if checked:
            self.log_write("Live Analysis Enabled")
        else:
            self.log_write("Live Analysis Disabled")
            self.latest_induced = None
            self.latest_cf = None
            self._update_viewer()

    def _setup_analysis_tab(self: Any) -> None:
        """Setup the analysis and plotting tab."""
        analysis_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(analysis_page)

        # Controls
        controls = QtWidgets.QHBoxLayout()

        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.addItems(
            [
                "Dashboard",
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Energy Analysis",
                "Kinematic Sequence",
                "Phase Diagram",
                "Frequency Analysis (PSD)",
                "Correlation Matrix",
                "Induced Accelerations",
                "Counterfactuals (ZTCF/ZVCF)",
                "Swing Profile (Radar)",
                "Power Flow",
            ]
        )

        self.joint_select_combo = QtWidgets.QComboBox()
        self.joint_select_combo.setMinimumWidth(120)

        controls.addWidget(QtWidgets.QLabel("Joint:"))
        controls.addWidget(self.joint_select_combo)
        controls.addWidget(QtWidgets.QLabel("Plot Type:"))
        controls.addWidget(self.plot_combo)

        self.btn_plot = QtWidgets.QPushButton("Generate Plot")
        self.btn_plot.clicked.connect(self._generate_plot)
        controls.addWidget(self.btn_plot)

        self.btn_export_csv = QtWidgets.QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self._export_statistics)
        controls.addWidget(self.btn_export_csv)

        controls.addStretch()
        layout.addLayout(controls)

        # Canvas
        try:
            self.canvas = MplCanvas(width=5, height=4, dpi=100)
            layout.addWidget(self.canvas)
        except RuntimeError:
            self.canvas = None  # type: ignore[assignment]
            layout.addWidget(QtWidgets.QLabel("Plotting requires GUI environment"))

        self.main_tabs.addTab(analysis_page, "Post-Hoc Analysis")

    def _generate_plot(self: Any) -> None:
        """Generate the selected plot."""
        if self.canvas is None:
            return

        if self.recorder.get_num_frames() == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No simulation data recorded yet."
            )
            return

        self.canvas.fig.clear()

        # Initialize plotter
        plotter = GolfSwingPlotter(self.recorder, self.joint_names)

        plot_type = self.plot_combo.currentText()

        if plot_type == "Dashboard":
            plotter.plot_summary_dashboard(self.canvas.fig)
        elif plot_type == "Joint Angles":
            plotter.plot_joint_angles(self.canvas.fig)
        elif plot_type == "Joint Velocities":
            plotter.plot_joint_velocities(self.canvas.fig)
        elif plot_type == "Joint Torques":
            plotter.plot_joint_torques(self.canvas.fig)
        elif plot_type == "Energy Analysis":
            plotter.plot_energy_analysis(self.canvas.fig)
        elif plot_type == "Kinematic Sequence":
            segments = {name: i for i, name in enumerate(self.joint_names)}
            plotter.plot_kinematic_sequence(self.canvas.fig, segments)
        elif plot_type == "Phase Diagram":
            plotter.plot_phase_diagram(self.canvas.fig, joint_idx=0)
        elif plot_type == "Frequency Analysis (PSD)":
            plotter.plot_frequency_analysis(self.canvas.fig, joint_idx=0)
        elif plot_type == "Correlation Matrix":
            plotter.plot_correlation_matrix(self.canvas.fig)
        elif plot_type == "Induced Accelerations":
            self._plot_induced_accelerations()
        elif plot_type == "Counterfactuals (ZTCF/ZVCF)":
            self._plot_counterfactuals()
        elif plot_type == "Swing Profile (Radar)":
            self._plot_swing_profile(plotter)
        elif plot_type == "Power Flow":
            if any(f.actuator_powers.size > 0 for f in self.recorder.frames):
                plotter.plot_power_flow(self.canvas.fig)
            else:
                ax = self.canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No actuator power data", ha="center", va="center")

        self.canvas.draw()

    def _plot_swing_profile(self: Any, plotter: GolfSwingPlotter) -> None:
        """Plot the Swing Profile radar chart."""
        times, positions = self.recorder.get_time_series("joint_positions")
        _, velocities = self.recorder.get_time_series("joint_velocities")
        _, torques = self.recorder.get_time_series("joint_torques")
        _, club_speed = self.recorder.get_time_series("club_head_speed")

        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        torques = np.asarray(torques)
        club_speed = np.asarray(club_speed)

        analyzer = StatisticalAnalyzer(
            times, positions, velocities, torques, club_head_speed=club_speed
        )
        report = analyzer.generate_comprehensive_report()

        metrics = {
            "Speed": 0.0,
            "Efficiency": 0.0,
            "Tempo": 0.0,
            "Stability": 0.0,
            "Power": 0.0,
        }

        if "club_head_speed" in report:
            peak = report["club_head_speed"]["peak_value"]
            metrics["Speed"] = min(peak / 50.0, 1.0)

        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            err = abs(ratio - 3.0)
            metrics["Tempo"] = max(0.0, 1.0 - (err / 2.0))

        if "energy_efficiency" in report:
            metrics["Efficiency"] = report["energy_efficiency"] / 100.0

        plotter.plot_radar_chart(self.canvas.fig, metrics)

    def _ensure_analyzer_initialized(self: Any) -> None:
        """Ensure the InducedAccelerationAnalyzer is initialized."""
        if self.analyzer is None and self.model is not None:
            from .induced_acceleration import InducedAccelerationAnalyzer

            self.analyzer = InducedAccelerationAnalyzer(self.model, self.data)

    def _plot_induced_accelerations(self: Any) -> None:
        """Calculate and plot induced accelerations for selected joint."""
        if not self.recorder.frames:
            return

        # Get selected joint
        joint_name = self.joint_select_combo.currentText()
        if not joint_name:
            if self.joint_names:
                joint_name = self.joint_names[0]
            else:
                return

        try:
            if self.model is None:
                return
            joint_idx = list(self.model.names).index(joint_name)
            v_idx = self.model.joints[joint_idx].idx_v
        except ValueError:
            return

        if not getattr(self, "_analysis_data_populated", False):
            self._ensure_analysis_data_populated()
            self._analysis_data_populated = True

        has_specific = any(
            "specific_control" in f.induced_accelerations for f in self.recorder.frames
        )

        txt = self.combo_induced.currentText()
        if (
            not has_specific
            and txt
            and txt not in ["gravity", "velocity", "total"]
            and self.analyzer
        ):
            try:
                parts = [float(x) for x in txt.split(",")]
                if len(parts) == self.model.nv:
                    spec_tau = np.array(parts)
                    QtWidgets.QApplication.setOverrideCursor(
                        QtCore.Qt.CursorShape.WaitCursor
                    )
                    for frame in self.recorder.frames:
                        if frame.joint_positions is not None:
                            a_spec = self.analyzer.compute_specific_control(
                                frame.joint_positions, spec_tau
                            )
                            frame.induced_accelerations["specific_control"] = a_spec
                    QtWidgets.QApplication.restoreOverrideCursor()
                    has_specific = True
            except ValueError:
                pass

        plotter = GolfSwingPlotter(self.recorder, self.joint_names)

        plotter.plot_induced_acceleration(
            self.canvas.fig, "breakdown", joint_idx=v_idx, breakdown_mode=True
        )

        if has_specific:
            times, spec_vals = self.recorder.get_induced_acceleration_series(
                "specific_control"
            )
            if len(times) > 0 and spec_vals.size > 0:
                ax = self.canvas.fig.axes[0]
                if v_idx < spec_vals.shape[1]:
                    ax.plot(
                        times,
                        spec_vals[:, v_idx],
                        label="Specific Source",
                        color="magenta",
                        linewidth=2,
                        linestyle=":",
                    )
                    ax.legend()

    def _plot_counterfactuals(self: Any) -> None:
        """Plot ZTCF (Zero Torque Accel) and ZVCF (Zero Velocity Torque)."""
        if not self.recorder.frames:
            return

        joint_name = self.joint_select_combo.currentText()
        if not joint_name:
            if self.joint_names:
                joint_name = self.joint_names[0]
            else:
                return

        try:
            if self.model is None:
                return
            joint_idx = list(self.model.names).index(joint_name)
            v_idx = self.model.joints[joint_idx].idx_v
        except ValueError:
            return

        self._ensure_analysis_data_populated()

        plotter = GolfSwingPlotter(self.recorder, self.joint_names)
        plotter.plot_counterfactual_comparison(
            self.canvas.fig, "dual", metric_idx=v_idx
        )

    def _ensure_analysis_data_populated(self: Any) -> None:
        """Populate recorder frames with analysis data if missing."""
        if not self.recorder.frames:
            return

        # Check first frame
        if (
            self.recorder.frames[0].induced_accelerations
            and self.recorder.frames[0].counterfactuals
        ):
            return  # Already populated

        self._ensure_analyzer_initialized()
        if self.analyzer is None:
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            for frame in self.recorder.frames:
                if not frame.induced_accelerations:
                    frame.induced_accelerations = self.analyzer.compute_components(
                        frame.joint_positions,
                        frame.joint_velocities,
                        frame.joint_torques,
                    )
                if not frame.counterfactuals:
                    if hasattr(self.analyzer, "compute_counterfactuals"):
                        frame.counterfactuals = self.analyzer.compute_counterfactuals(
                            frame.joint_positions, frame.joint_velocities
                        )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _export_statistics(self: Any) -> None:
        """Export recorded data to multiple formats."""
        if self.recorder.get_num_frames() == 0:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "pinocchio_data", "All Files (*)"
        )
        if not filename:
            return

        # Ensure advanced metrics are computed
        self._ensure_analysis_data_populated()

        try:
            from shared.python.data_io.export import export_recording_all_formats

            data_dict = self.recorder.export_to_dict()
            results = export_recording_all_formats(filename, data_dict)

            msg = "Export Results:\n"
            for fmt, success in results.items():
                msg += f"{fmt}: {'Success' if success else 'Failed'}\n"

            QtWidgets.QMessageBox.information(self, "Export Complete", msg)
            self.log_write(f"Data exported to {filename}")

        except ImportError as e:
            self.log_write(f"Error exporting data: {e}")
            logger.exception("Export failed")
