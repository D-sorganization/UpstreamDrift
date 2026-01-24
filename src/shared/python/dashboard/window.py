"""Unified Dashboard Window for Physics Engine Analysis.

Provides a unified interface for:
- Simulation control (Start/Stop/Reset)
- Real-time visualization (Live Plots)
- Post-hoc Analysis (ZTCF, Induced Acceleration, Energies)
- Data Export
"""

from __future__ import annotations

from src.shared.python.logging_config import get_logger

import numpy as np
from PyQt6 import QtCore, QtWidgets

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.dashboard.runner import SimulationRunner
from src.shared.python.dashboard.widgets import ControlPanel, LivePlotWidget

# Updated import to use generic export module
from src.shared.python.export import (
    export_recording_all_formats,
    get_available_export_formats,
)
from src.shared.python.interfaces import PhysicsEngine
from src.shared.python.plotting import GolfSwingPlotter, MplCanvas
from src.shared.python.statistical_analysis import StatisticalAnalyzer

logger = get_logger(__name__)


class UnifiedDashboardWindow(QtWidgets.QMainWindow):
    """Main window for the unified physics dashboard."""

    def __init__(self, engine: PhysicsEngine, title: str = "Physics Dashboard") -> None:
        """Initialize the dashboard.

        Args:
            engine: The physics engine instance to control and analyze.
            title: Window title.
        """
        super().__init__()
        self.setWindowTitle(title)
        self.setAccessibleName("Physics Dashboard Main Window")
        self.setAccessibleDescription(
            "Main application window containing simulation controls, live plots, and analysis tools."
        )
        self.resize(1200, 800)

        self.engine = engine
        self.recorder = GenericPhysicsRecorder(self.engine)
        self.runner = SimulationRunner(self.engine, self.recorder)
        # Resolve joint names from engine if available
        joint_names = self._resolve_joint_names()

        self.plotter = GolfSwingPlotter(self.recorder, joint_names=joint_names)

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.addWidget(self.status_label)

        # Setup UI
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Create and arrange UI components."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # --- Left Panel: Live View & Controls ---
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # Control Panel
        self.control_panel = ControlPanel()
        left_layout.addWidget(self.control_panel)

        # Live Plots
        self.live_plot = LivePlotWidget(self.recorder)
        if hasattr(self, "plotter") and self.plotter.joint_names:
            self.live_plot.set_joint_names(self.plotter.joint_names)

        left_layout.addWidget(self.live_plot)

        main_layout.addWidget(left_panel, stretch=1)

        # --- Right Panel: Analysis Tabs ---
        right_panel = QtWidgets.QTabWidget()

        # Tab 1: Detailed Plotting
        self.plotting_tab = QtWidgets.QWidget()
        self._setup_plotting_tab(self.plotting_tab)
        right_panel.addTab(self.plotting_tab, "Plotting")

        # Tab 2: Advanced Analysis (Counterfactuals)
        self.analysis_tab = QtWidgets.QWidget()
        self._setup_analysis_tab(self.analysis_tab)
        right_panel.addTab(self.analysis_tab, "Counterfactuals")

        # Tab 3: Export
        self.export_tab = QtWidgets.QWidget()
        self._setup_export_tab(self.export_tab)
        right_panel.addTab(self.export_tab, "Export")

        main_layout.addWidget(right_panel, stretch=2)

    def _setup_plotting_tab(self, parent: QtWidgets.QWidget) -> None:
        """Setup standard plotting tab."""
        layout = QtWidgets.QVBoxLayout(parent)

        # Plot Selector
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.setToolTip("Select the type of plot to display")
        self.plot_type_combo.addItems(
            [
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Energies",
                "Club Head Speed",
                "Angular Momentum",
                "Power Flow",
                "Joint Power Curves",
                "Impulse Accumulation",
                "Phase Diagram (Joint 0)",
                "Poincaré Map (3D)",
                "Chaos Analysis (Lyapunov)",
                "Recurrence Plot",
                "Stability Diagram (CoM vs CoP)",
                "CoP Trajectory",
                "GRF Butterfly Diagram",
                "Club Head Trajectory (3D)",
                "Kinematic Sequence (Bars)",
                "Swing Profile (Radar)",
                "Summary Dashboard",
            ]
        )

        lbl_plot_type = QtWidgets.QLabel("Plot Type:")
        lbl_plot_type.setBuddy(self.plot_type_combo)
        layout.addWidget(lbl_plot_type)
        layout.addWidget(self.plot_type_combo)

        btn_refresh = QtWidgets.QPushButton("Refresh Plot")
        btn_refresh.setToolTip("Refresh the plot with the latest data")
        btn_refresh.clicked.connect(self.refresh_static_plot)
        layout.addWidget(btn_refresh)

        # Canvas
        self.static_canvas = MplCanvas(width=5, height=4, dpi=100)
        layout.addWidget(self.static_canvas)

    def _setup_analysis_tab(self, parent: QtWidgets.QWidget) -> None:
        """Setup advanced analysis tab."""
        layout = QtWidgets.QVBoxLayout(parent)

        self.btn_compute = QtWidgets.QPushButton("Compute Analysis (Post-Hoc)")
        self.btn_compute.setToolTip(
            "Run expensive post-hoc analysis algorithms (ZTCF, etc)"
        )
        self.btn_compute.clicked.connect(self.compute_analysis)
        layout.addWidget(self.btn_compute)

        self.analysis_combo = QtWidgets.QComboBox()
        self.analysis_combo.setToolTip("Select the type of analysis to visualize")
        self.analysis_combo.addItems(
            [
                "ZTCF vs ZVCF",
                "Induced Acceleration (Gravity)",
                "Induced Acceleration (Control)",
                "Club Induced Acceleration Breakdown",
                "Stability Metrics",
            ]
        )

        lbl_analysis_type = QtWidgets.QLabel("Analysis Type:")
        lbl_analysis_type.setBuddy(self.analysis_combo)
        layout.addWidget(lbl_analysis_type)
        layout.addWidget(self.analysis_combo)

        btn_show = QtWidgets.QPushButton("Show Analysis")
        btn_show.clicked.connect(self.show_analysis_plot)
        layout.addWidget(btn_show)

        self.analysis_canvas = MplCanvas(width=5, height=4, dpi=100)
        layout.addWidget(self.analysis_canvas)

    def _setup_export_tab(self, parent: QtWidgets.QWidget) -> None:
        """Setup export tab."""
        layout = QtWidgets.QVBoxLayout(parent)

        info_label = QtWidgets.QLabel("Export recorded data to various formats.")
        layout.addWidget(info_label)

        self.export_formats_list = QtWidgets.QListWidget()
        self.export_formats_list.setAccessibleName("Export Formats")
        self.export_formats_list.setAccessibleDescription(
            "List of available formats for exporting simulation data"
        )
        formats = get_available_export_formats()
        for fmt, info in formats.items():
            item = QtWidgets.QListWidgetItem(f"{info['name']} ({info['extension']})")
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, fmt)
            self.export_formats_list.addItem(item)

        layout.addWidget(self.export_formats_list)

        btn_export = QtWidgets.QPushButton("Export Data")
        btn_export.setToolTip("Export recorded data to file")
        btn_export.clicked.connect(self.export_data)
        layout.addWidget(btn_export)

        layout.addStretch()

    def _resolve_joint_names(self) -> list[str]:
        """Try to resolve joint names from the engine."""
        # Check for get_joint_names method (standard interface)
        if hasattr(self.engine, "get_joint_names"):
            try:
                names = self.engine.get_joint_names()  # type: ignore
                if names:
                    return names
            except Exception as e:
                logger.warning(f"Failed to get joint names from engine: {e}")

        return []

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Runner signals
        self.runner.frame_ready.connect(self.live_plot.update_plot)
        self.runner.status_message.connect(self.status_label.setText)

        # Control panel signals
        self.control_panel.start_requested.connect(self.runner.start)
        self.control_panel.stop_requested.connect(self.runner.stop)
        self.control_panel.pause_requested.connect(self.runner.toggle_pause)
        self.control_panel.reset_requested.connect(self._reset_simulation)
        self.control_panel.toggle_playback_requested.connect(self._toggle_playback)

    def _toggle_playback(self) -> None:
        """Handle toggle playback (e.g., from Space key)."""
        if self.runner.isRunning():
            self.runner.toggle_pause()
        else:
            self.runner.start()

    def _reset_simulation(self) -> None:
        """Reset simulation and recorder."""
        if self.runner.isRunning():
            self.runner.stop()
            self.runner.wait()

        self.engine.reset()
        self.recorder.reset()
        self.live_plot.ax.clear()
        self.live_plot.canvas.draw()
        self.status_label.setText("Reset complete.")

    def refresh_static_plot(self) -> None:
        """Update the static plot based on selection."""
        plot_type = self.plot_type_combo.currentText()
        self.static_canvas.fig.clear()

        try:
            if plot_type == "Joint Angles":
                self.plotter.plot_joint_angles(self.static_canvas.fig)
            elif plot_type == "Joint Velocities":
                self.plotter.plot_joint_velocities(self.static_canvas.fig)
            elif plot_type == "Joint Torques":
                self.plotter.plot_joint_torques(self.static_canvas.fig)
            elif plot_type == "Energies":
                self.plotter.plot_energy_analysis(self.static_canvas.fig)
            elif plot_type == "Club Head Speed":
                self.plotter.plot_club_head_speed(self.static_canvas.fig)
            elif plot_type == "Angular Momentum":
                self.plotter.plot_angular_momentum(self.static_canvas.fig)
            elif plot_type == "Power Flow":
                self.plotter.plot_power_flow(self.static_canvas.fig)
            elif plot_type == "Joint Power Curves":
                self.plotter.plot_joint_power_curves(self.static_canvas.fig)
            elif plot_type == "Impulse Accumulation":
                self.plotter.plot_impulse_accumulation(self.static_canvas.fig)
            elif plot_type == "Phase Diagram (Joint 0)":
                self.plotter.plot_phase_diagram(self.static_canvas.fig, joint_idx=0)
            elif plot_type == "Poincaré Map (3D)":
                # Default: Pos 0, Vel 0, Acc 0. Section: Vel 0 = 0.
                self.plotter.plot_poincare_map_3d(
                    self.static_canvas.fig,
                    dimensions=[("position", 0), ("velocity", 0), ("acceleration", 0)],
                    section_condition=("velocity", 0, 0.0),
                    title="Poincaré Map (Joint 0)",
                )
            elif plot_type == "Chaos Analysis (Lyapunov)":
                self.plotter.plot_lyapunov_exponent(self.static_canvas.fig, joint_idx=0)
            elif plot_type == "Recurrence Plot":
                # Need to compute matrix
                times, positions = self.recorder.get_time_series("joint_positions")
                _, velocities = self.recorder.get_time_series("joint_velocities")
                if len(times) > 0:
                    analyzer = StatisticalAnalyzer(
                        times=np.asarray(times),
                        joint_positions=np.asarray(positions),
                        joint_velocities=np.asarray(velocities),
                        joint_torques=np.zeros_like(positions),
                    )
                    rm = analyzer.compute_recurrence_matrix()
                    self.plotter.plot_recurrence_plot(self.static_canvas.fig, rm)
                else:
                    raise ValueError("No data available")
            elif plot_type == "Stability Diagram (CoM vs CoP)":
                self.plotter.plot_stability_diagram(self.static_canvas.fig)
            elif plot_type == "CoP Trajectory":
                self.plotter.plot_cop_trajectory(self.static_canvas.fig)
            elif plot_type == "GRF Butterfly Diagram":
                self.plotter.plot_grf_butterfly_diagram(self.static_canvas.fig)
            elif plot_type == "Club Head Trajectory (3D)":
                self.plotter.plot_club_head_trajectory(self.static_canvas.fig)
            elif plot_type == "Kinematic Sequence (Bars)":
                # Heuristic: First few joints in order
                _, vels = self.recorder.get_time_series("joint_velocities")
                vels = np.asarray(vels)
                n_joints = vels.shape[1] if len(vels) > 0 else 0
                if n_joints >= 3:
                    # Assume Pelvis (0), Thorax (1), Arm (2)...
                    indices = {
                        "Pelvis": 0,
                        "Thorax": 1,
                        "Arm": min(2, n_joints - 1),
                    }
                    if n_joints > 3:
                        indices["Club"] = n_joints - 1
                    self.plotter.plot_kinematic_sequence_bars(
                        self.static_canvas.fig, indices
                    )
                else:
                    # Fallback
                    indices = {f"Joint {i}": i for i in range(n_joints)}
                    self.plotter.plot_kinematic_sequence_bars(
                        self.static_canvas.fig, indices
                    )
            elif plot_type == "Swing Profile (Radar)":
                times, positions = self.recorder.get_time_series("joint_positions")
                _, velocities = self.recorder.get_time_series("joint_velocities")
                _, torques = self.recorder.get_time_series("joint_torques")
                try:
                    _, club_speed = self.recorder.get_time_series("club_head_speed")
                except (KeyError, AttributeError):
                    club_speed = None

                if len(times) > 0:
                    analyzer = StatisticalAnalyzer(
                        times=np.asarray(times),
                        joint_positions=np.asarray(positions),
                        joint_velocities=np.asarray(velocities),
                        joint_torques=np.asarray(torques),
                        club_head_speed=(
                            np.asarray(club_speed) if club_speed is not None else None
                        ),
                    )
                    dna = analyzer.compute_swing_profile()
                    if dna:
                        metrics = {
                            "Speed": dna.speed_score,
                            "Sequence": dna.sequence_score,
                            "Stability": dna.stability_score,
                            "Efficiency": dna.efficiency_score,
                            "Power": dna.power_score,
                        }
                        self.plotter.plot_radar_chart(self.static_canvas.fig, metrics)
                    else:
                        raise ValueError("Could not compute Swing Profile")
                else:
                    raise ValueError("No data available")
            elif plot_type == "Summary Dashboard":
                self.plotter.plot_summary_dashboard(self.static_canvas.fig)
        except Exception as e:
            ax = self.static_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot Error: {e}", ha="center", va="center")
            logger.error(f"Error generating static plot '{plot_type}': {e}")

        self.static_canvas.draw()

    def compute_analysis(self) -> None:
        """Trigger post-hoc analysis computation."""
        # Safety Check: Stop running simulation
        if self.runner.isRunning():
            self.runner.stop()
            self.runner.wait()
            self.status_label.setText("Simulation stopped for analysis.")
            logger.info("Stopped simulation for analysis safety.")

        self.status_label.setText("Computing analysis... (may take a moment)")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.btn_compute.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        try:
            self.recorder.compute_analysis_post_hoc()
            self.status_label.setText("Analysis complete.")
        except Exception as e:
            self.status_label.setText(f"Analysis failed: {e}")
            logger.error("Analysis error: %s", e)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.btn_compute.setEnabled(True)

    def show_analysis_plot(self) -> None:
        """Show selected analysis plot."""
        analysis_type = self.analysis_combo.currentText()
        self.analysis_canvas.fig.clear()

        try:
            if analysis_type == "ZTCF vs ZVCF":
                # Plot for Joint 0 as example, or add selector
                self.plotter.plot_counterfactual_comparison(
                    self.analysis_canvas.fig, "dual", metric_idx=0
                )
            elif analysis_type == "Induced Acceleration (Gravity)":
                self.plotter.plot_induced_acceleration(
                    self.analysis_canvas.fig, "gravity"
                )
            elif analysis_type == "Induced Acceleration (Control)":
                self.plotter.plot_induced_acceleration(
                    self.analysis_canvas.fig, "control"
                )
            elif analysis_type == "Club Induced Acceleration Breakdown":
                self.plotter.plot_club_induced_acceleration(
                    self.analysis_canvas.fig, breakdown_mode=True
                )
            elif analysis_type == "Stability Metrics":
                self.plotter.plot_stability_metrics(self.analysis_canvas.fig)
        except Exception as e:
            ax = self.analysis_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha="center")

        self.analysis_canvas.draw()

    def export_data(self) -> None:
        """Export data to selected formats."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", "swing_data", "All Files (*)"
        )
        if not filename:
            return

        selected_formats = []
        for i in range(self.export_formats_list.count()):
            item = self.export_formats_list.item(i)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                selected_formats.append(item.data(QtCore.Qt.ItemDataRole.UserRole))

        data = self.recorder.get_data_dict()
        results = export_recording_all_formats(filename, data, selected_formats)

        msg = "Export Results:\n"
        for fmt, success in results.items():
            msg += f"{fmt}: {'Success' if success else 'Failed'}\n"

        QtWidgets.QMessageBox.information(self, "Export Complete", msg)
