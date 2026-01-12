"""Unified PySide6 GUI application for golf biomechanics platform."""

from __future__ import annotations

import logging
import sys

import numpy as np
import pinocchio as pin
from PySide6 import QtWidgets
from shared.python.biomechanics_data import BiomechanicalData
from shared.python.plotting import GolfSwingPlotter, MplCanvas, RecorderInterface

from ..sim.dynamics import DynamicsEngine

logger = logging.getLogger(__name__)


class GuiRecorder(RecorderInterface):
    """Recorder adapter for the GUI data."""

    def __init__(self, data_store: list[BiomechanicalData]):
        self.data_store = data_store

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        if not self.data_store:
            return np.array([]), np.array([])

        times = np.array([d.time for d in self.data_store])

        # Helper to extract attribute safely
        values = []
        for d in self.data_store:
            val = getattr(d, field_name, None)
            if val is None:
                # Try specific mappings if needed, or fallback
                pass
            values.append(val if val is not None else 0.0)  # Handle None generally?

        # Refine handling based on field_name
        if field_name == "joint_positions":
            values = [d.joint_positions for d in self.data_store]
        elif field_name == "joint_velocities":
            values = [d.joint_velocities for d in self.data_store]
        elif field_name == "joint_torques":
            values = [d.joint_torques for d in self.data_store]
        elif field_name == "club_head_speed":
            values = [d.club_head_speed for d in self.data_store]
        else:
            # Generic fallback for scalar or simple arrays
            try:
                values = [val if val is not None else 0.0 for val in values]
            except Exception:
                pass

        return times, values

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.data_store:
            return np.array([]), np.array([])

        times = []
        values = []
        for d in self.data_store:
            val = None
            if isinstance(source_name, str):
                val = d.induced_accelerations.get(source_name)

            if val is not None:
                times.append(d.time)
                values.append(val)

        if not values:
            return np.array([]), np.array([])

        return np.array(times), np.array(values)

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.data_store:
            return np.array([]), np.array([])

        times = []
        values = []
        for d in self.data_store:
            val = d.counterfactuals.get(cf_name)
            if val is not None:
                times.append(d.time)
                values.append(val)

        if not values:
            return np.array([]), np.array([])

        return np.array(times), np.array(values)


class UnifiedGolfGUI(QtWidgets.QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self) -> None:
        """Initialize unified GUI."""
        super().__init__()
        self.setWindowTitle("Unified Golf Biomechanics Platform")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget with tabs
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Data storage
        self.recorded_data: list[BiomechanicalData] = []
        self.recorder = GuiRecorder(self.recorded_data)

        # Physics Engine
        self.model = None
        self.data = None
        self.dynamics_engine = None

        # Create tabs
        self._create_model_viewer_tab()
        self._create_ik_tab()
        self._create_dynamics_tab()
        self._create_counterfactuals_tab()
        self._create_results_tab()
        self._create_ml_tab()
        self._create_settings_tab()

        logger.info("Unified GUI initialized")

    def _create_model_viewer_tab(self) -> None:
        """Create model viewer tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Model loading
        load_group = QtWidgets.QGroupBox("Model Loading")
        load_layout = QtWidgets.QHBoxLayout()
        self.model_path_edit = QtWidgets.QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to canonical YAML or URDF/MJCF")
        load_btn = QtWidgets.QPushButton("Load Model")
        load_btn.clicked.connect(self._load_model)
        load_layout.addWidget(self.model_path_edit)
        load_layout.addWidget(load_btn)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # Viewer selection
        viewer_group = QtWidgets.QGroupBox("Viewer")
        viewer_layout = QtWidgets.QHBoxLayout()
        self.viewer_combo = QtWidgets.QComboBox()
        self.viewer_combo.addItems(["MeshCat", "MuJoCo", "Geppetto"])
        viewer_layout.addWidget(QtWidgets.QLabel("Viewer:"))
        viewer_layout.addWidget(self.viewer_combo)
        viewer_group.setLayout(viewer_layout)
        layout.addWidget(viewer_group)

        # Placeholder for embedded viewer
        self.viewer_widget = QtWidgets.QWidget()
        self.viewer_widget.setMinimumHeight(400)
        self.viewer_widget.setStyleSheet("background-color: #2b2b2b;")
        layout.addWidget(self.viewer_widget)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Model Viewer")

    def _create_ik_tab(self) -> None:
        """Create IK tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # IK task configuration
        task_group = QtWidgets.QGroupBox("IK Tasks")
        task_layout = QtWidgets.QVBoxLayout()

        # Clubface task
        clubface_layout = QtWidgets.QHBoxLayout()
        clubface_layout.addWidget(QtWidgets.QLabel("Clubface Position:"))
        self.clubface_x = QtWidgets.QDoubleSpinBox()
        self.clubface_y = QtWidgets.QDoubleSpinBox()
        self.clubface_z = QtWidgets.QDoubleSpinBox()
        clubface_layout.addWidget(self.clubface_x)
        clubface_layout.addWidget(self.clubface_y)
        clubface_layout.addWidget(self.clubface_z)
        task_layout.addLayout(clubface_layout)

        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        # Solve button
        solve_btn = QtWidgets.QPushButton("Solve IK")
        solve_btn.clicked.connect(self._solve_ik)
        layout.addWidget(solve_btn)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Inverse Kinematics")

    def _create_dynamics_tab(self) -> None:
        """Create dynamics tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Forward simulation
        sim_group = QtWidgets.QGroupBox("Forward Simulation")
        sim_layout = QtWidgets.QHBoxLayout()
        self.sim_start_btn = QtWidgets.QPushButton("Start Simulation")
        self.sim_stop_btn = QtWidgets.QPushButton("Stop Simulation")
        sim_layout.addWidget(self.sim_start_btn)
        sim_layout.addWidget(self.sim_stop_btn)
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Inverse dynamics
        id_group = QtWidgets.QGroupBox("Inverse Dynamics")
        id_layout = QtWidgets.QVBoxLayout()
        self.compute_id_btn = QtWidgets.QPushButton("Compute Torques")
        id_layout.addWidget(self.compute_id_btn)
        id_group.setLayout(id_layout)
        layout.addWidget(id_group)

        # Real-time Plot Placeholder
        self.rt_plot_canvas = MplCanvas(width=5, height=4)
        layout.addWidget(self.rt_plot_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Dynamics")

    def _create_counterfactuals_tab(self) -> None:
        """Create counterfactuals tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Counterfactual selection
        cf_group = QtWidgets.QGroupBox("Counterfactual Analysis")
        cf_layout = QtWidgets.QVBoxLayout()

        self.ztcf_btn = QtWidgets.QPushButton("Run Zero Torque Counterfactual (ZTCF)")
        self.zvcf_btn = QtWidgets.QPushButton("Run Zero Velocity Counterfactual (ZVCF)")

        cf_layout.addWidget(self.ztcf_btn)
        cf_layout.addWidget(self.zvcf_btn)
        cf_group.setLayout(cf_layout)
        layout.addWidget(cf_group)

        # Visualization for CF
        self.cf_plot_canvas = MplCanvas(width=5, height=4)
        layout.addWidget(self.cf_plot_canvas)

        # Connect
        self.ztcf_btn.clicked.connect(lambda: self._run_counterfactual("ztcf"))
        self.zvcf_btn.clicked.connect(lambda: self._run_counterfactual("zvcf"))

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Counterfactuals")

    def _create_results_tab(self) -> None:
        """Create results/plotting tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        # Control Panel
        control_panel = QtWidgets.QGroupBox("Plot Controls")
        control_layout = QtWidgets.QVBoxLayout()

        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(
            [
                "Summary Dashboard",
                "Joint Angles",
                "Joint Velocities",
                "Joint Torques",
                "Induced Accelerations",
                "Energy Analysis",
                "Phase Diagram",
                "Counterfactual Comparison",
            ]
        )
        control_layout.addWidget(QtWidgets.QLabel("Plot Type:"))
        control_layout.addWidget(self.plot_type_combo)

        self.update_plot_btn = QtWidgets.QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self._update_results_plot)
        control_layout.addWidget(self.update_plot_btn)

        self.export_btn = QtWidgets.QPushButton("Export Data")
        self.export_btn.clicked.connect(self._export_data)
        control_layout.addWidget(self.export_btn)

        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel, 1)

        # Canvas
        self.results_canvas = MplCanvas(width=8, height=6)
        layout.addWidget(self.results_canvas, 4)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Results & Analysis")

    def _create_ml_tab(self) -> None:
        """Create ML tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        ml_group = QtWidgets.QGroupBox("Machine Learning")
        ml_layout = QtWidgets.QVBoxLayout()

        self.generate_data_btn = QtWidgets.QPushButton("Generate Dataset")
        self.optimize_swing_btn = QtWidgets.QPushButton("Optimize Swing")

        ml_layout.addWidget(self.generate_data_btn)
        ml_layout.addWidget(self.optimize_swing_btn)
        ml_group.setLayout(ml_layout)
        layout.addWidget(ml_group)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "ML")

    def _create_settings_tab(self) -> None:
        """Create settings tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QFormLayout()

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["Pinocchio", "MuJoCo", "PINK"])
        settings_layout.addRow("Default Backend:", self.backend_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Settings")

    def _load_model(self) -> None:
        """Load model from path."""
        path = self.model_path_edit.text()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a model path")
            return
        logger.info("Loading model from: %s", path)
        try:
            self.model = pin.buildModelFromUrdf(path)
            if self.model is not None:
                self.data = self.model.createData()
                self.dynamics_engine = DynamicsEngine(self.model, self.data)
                logger.info("Pinocchio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def _solve_ik(self) -> None:
        """Solve inverse kinematics."""
        logger.info("Solving IK...")
        # NOTE: Implement IK solving

    def _run_counterfactual(self, cf_type: str) -> None:
        """Run counterfactual analysis."""
        logger.info(f"Running {cf_type} counterfactual...")

        if not self.dynamics_engine or not self.recorded_data:
            # Need real data to run real counterfactuals
            if not self.recorded_data:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Data",
                    "No recorded data to analyze. Run simulation first.",
                )
                return

        # Perform Analysis frame-by-frame
        # Note: This is simplified. ZVCF/ZTCF usually imply re-integration over time.
        # But here we might just compute instantaneous drift or single-step prediction.
        # If we want full trajectory counterfactual, we need to re-simulate
        # the whole sequence.

        # Re-simulation approach:
        # 1. Take initial state from recorded data
        # 2. Integrate using ZTCF/ZVCF dynamics

        if cf_type == "ztcf":
            # Drift simulation (tau=0)
            dt = 0.01  # Assumed time step
            q = self.recorded_data[0].joint_positions.copy()
            v = self.recorded_data[0].joint_velocities.copy()

            for _i, frame in enumerate(self.recorded_data):
                # Store result in frame
                frame.counterfactuals["ztcf"] = q.copy()

                # Step - check if dynamics_engine is available
                if self.dynamics_engine is not None:
                    q, v = self.dynamics_engine.compute_ztcf(q, v, dt)

        elif cf_type == "zvcf":
            # Zero Velocity (tau_g + tau_ctrl but v=0)
            dt = 0.01
            q = self.recorded_data[0].joint_positions.copy()

            for _i, frame in enumerate(self.recorded_data):
                frame.counterfactuals["zvcf"] = q.copy()

                # Step - check if dynamics_engine and model are available
                if self.dynamics_engine is not None and self.model is not None:
                    tau = frame.joint_torques
                    if tau.size == 0:
                        tau = np.zeros(self.model.nv)

                    q, _ = self.dynamics_engine.compute_zvcf(q, tau, dt)

        # Update Plot
        self.cf_plot_canvas.fig.clear()
        plotter = GolfSwingPlotter(self.recorder)
        plotter.plot_counterfactual_comparison(
            self.cf_plot_canvas.fig, cf_type, metric_idx=0
        )
        self.cf_plot_canvas.draw()

        QtWidgets.QMessageBox.information(
            self, "Success", f"{cf_type.upper()} analysis complete."
        )

    def _update_results_plot(self) -> None:
        """Update the plot in the Results tab."""
        plot_type = self.plot_type_combo.currentText()
        self.results_canvas.fig.clear()

        plotter = GolfSwingPlotter(self.recorder)

        if plot_type == "Summary Dashboard":
            plotter.plot_summary_dashboard(self.results_canvas.fig)
        elif plot_type == "Joint Angles":
            plotter.plot_joint_angles(self.results_canvas.fig)
        elif plot_type == "Joint Velocities":
            plotter.plot_joint_velocities(self.results_canvas.fig)
        elif plot_type == "Joint Torques":
            plotter.plot_joint_torques(self.results_canvas.fig)
        elif plot_type == "Induced Accelerations":
            # Just default to gravity for now, ideally UI lets user pick source
            plotter.plot_induced_acceleration(self.results_canvas.fig, "gravity")
            # Or actuator if available
            if (
                self.recorded_data
                and "actuator" in self.recorded_data[0].induced_accelerations
            ):
                # Override for demo
                self.results_canvas.fig.clear()
                plotter.plot_induced_acceleration(self.results_canvas.fig, "actuator")
        elif plot_type == "Energy Analysis":
            plotter.plot_energy_analysis(self.results_canvas.fig)
        elif plot_type == "Phase Diagram":
            plotter.plot_phase_diagram(self.results_canvas.fig)
        elif plot_type == "Counterfactual Comparison":
            plotter.plot_counterfactual_comparison(self.results_canvas.fig, "ztcf")

        self.results_canvas.draw()

    def _export_data(self) -> None:
        """Export data to CSV/Parquet."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV Files (*.csv)"
        )
        if filename:
            import pandas as pd

            # Simple export logic
            data_list = []
            for d in self.recorded_data:
                row = {"time": d.time}
                # Add basic fields
                for i, q in enumerate(d.joint_positions):
                    row[f"q_{i}"] = q
                # Add induced
                for k, v in d.induced_accelerations.items():
                    for i, a in enumerate(v):
                        row[f"acc_induced_{k}_{i}"] = a
                # Add CF
                for k, v in d.counterfactuals.items():
                    for i, val in enumerate(v):
                        row[f"cf_{k}_{i}"] = val

                data_list.append(row)

            df = pd.DataFrame(data_list)
            df.to_csv(filename, index=False)
            logger.info("Data exported to %s", filename)


def main() -> None:
    """Main entry point."""
    app = QtWidgets.QApplication(sys.argv)
    window = UnifiedGolfGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
