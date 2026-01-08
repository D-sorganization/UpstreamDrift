"""Unified Dashboard Window for Physics Engine Analysis.

Provides a unified interface for:
- Simulation control (Start/Stop/Reset)
- Real-time visualization (Live Plots)
- Post-hoc Analysis (ZTCF, Induced Acceleration, Energies)
- Data Export
"""

from __future__ import annotations

import logging

from PyQt6 import QtCore, QtWidgets

from shared.python.dashboard.recorder import GenericPhysicsRecorder
from shared.python.dashboard.runner import SimulationRunner
from shared.python.dashboard.widgets import ControlPanel, LivePlotWidget

# Updated import to use generic export module
from shared.python.export import (
    export_recording_all_formats,
    get_available_export_formats,
)
from shared.python.interfaces import PhysicsEngine
from shared.python.plotting import GolfSwingPlotter, MplCanvas

LOGGER = logging.getLogger(__name__)


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
        self.resize(1200, 800)

        self.engine = engine
        self.recorder = GenericPhysicsRecorder(self.engine)
        self.runner = SimulationRunner(self.engine, self.recorder)
        self.plotter = GolfSwingPlotter(self.recorder)

        # Setup UI
        self._setup_ui()
        self._connect_signals()

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

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
        self.plot_type_combo.addItems(
            [
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Energies",
                "Club Head Speed",
                "Angular Momentum",
                "Power Flow",
            ]
        )
        layout.addWidget(self.plot_type_combo)

        btn_refresh = QtWidgets.QPushButton("Refresh Plot")
        btn_refresh.clicked.connect(self.refresh_static_plot)
        layout.addWidget(btn_refresh)

        # Canvas
        self.static_canvas = MplCanvas(width=5, height=4, dpi=100)
        layout.addWidget(self.static_canvas)

    def _setup_analysis_tab(self, parent: QtWidgets.QWidget) -> None:
        """Setup advanced analysis tab."""
        layout = QtWidgets.QVBoxLayout(parent)

        btn_compute = QtWidgets.QPushButton("Compute Analysis (Post-Hoc)")
        btn_compute.clicked.connect(self.compute_analysis)
        layout.addWidget(btn_compute)

        self.analysis_combo = QtWidgets.QComboBox()
        self.analysis_combo.addItems(
            [
                "ZTCF vs ZVCF",
                "Induced Acceleration (Gravity)",
                "Induced Acceleration (Control)",
                "Club Induced Acceleration Breakdown",
                "Stability Metrics",
            ]
        )
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
        formats = get_available_export_formats()
        for fmt, info in formats.items():
            item = QtWidgets.QListWidgetItem(f"{info['name']} ({info['extension']})")
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, fmt)
            self.export_formats_list.addItem(item)

        layout.addWidget(self.export_formats_list)

        btn_export = QtWidgets.QPushButton("Export Data")
        btn_export.clicked.connect(self.export_data)
        layout.addWidget(btn_export)

        layout.addStretch()

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

        self.static_canvas.draw()

    def compute_analysis(self) -> None:
        """Trigger post-hoc analysis computation."""
        # Safety Check: Stop running simulation
        if self.runner.isRunning():
            self.runner.stop()
            self.runner.wait()
            self.status_label.setText("Simulation stopped for analysis.")
            LOGGER.info("Stopped simulation for analysis safety.")

        self.status_label.setText("Computing analysis... (may take a moment)")
        QtWidgets.QApplication.processEvents()
        try:
            self.recorder.compute_analysis_post_hoc()
            self.status_label.setText("Analysis complete.")
        except Exception as e:
            self.status_label.setText(f"Analysis failed: {e}")
            LOGGER.error("Analysis error: %s", e)

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
