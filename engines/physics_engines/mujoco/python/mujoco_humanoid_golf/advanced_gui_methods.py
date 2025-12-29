import json
import logging
from pathlib import Path

import mujoco

logger = logging.getLogger(__name__)


class AdvancedGuiMethodsMixin:
    """Mixin class providing configuration loading methods."""

    def _load_launch_config(self) -> None:
        """Load configuration passed from launcher if available."""
        current_dir = Path.cwd()
        potential_paths = [
            current_dir / "simulation_config.json",
            current_dir.parent / "simulation_config.json",
            current_dir.parent / "docker" / "simulation_config.json",
            # If running from package
            Path(__file__).parent.parent.parent / "simulation_config.json",
        ]

        config_data = {}
        for path in potential_paths:
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        config_data = json.load(f)
                    logger.info("Loaded configuration from %s", path)
                    break
                except Exception as e:
                    logger.warning("Failed to parse config file: %s (%s)", path, e)

        if not config_data:
            return

        # Determine model based on config
        target_model = "full_body"

        # Find index
        model_index = 0
        found_model = False
        # self.model_configs must be defined in the main class
        if hasattr(self, "model_configs"):
            for i, cfg in enumerate(self.model_configs):
                if cfg["name"] == target_model:
                    model_index = i
                    found_model = True
                    break

        if found_model and hasattr(self, "model_combo"):
            # Set combo box (this triggers load_current_model via signal)
            self.model_combo.setCurrentIndex(model_index)

        # Apply colors if present
        if "colors" in config_data:
            self._apply_config_colors(config_data["colors"])

    def _apply_config_colors(self, colors: dict) -> None:
        """Apply colors from config to the model."""
        if not hasattr(self, "sim_widget") or self.sim_widget.model is None:
            return

        # Helper to set color for geoms containing string
        def set_color_contain(name_part: str, rgba: list) -> None:
            for i in range(self.sim_widget.model.ngeom):
                name = mujoco.mj_id2name(
                    self.sim_widget.model, mujoco.mjtObj.mjOBJ_GEOM, i
                )
                if name and name_part in name:
                    self.sim_widget.model.geom_rgba[i] = rgba

        if "shirt" in colors:
            set_color_contain("torso", colors["shirt"])
            set_color_contain("upper_arm", colors["shirt"])

        if "pants" in colors:
            set_color_contain("thigh", colors["pants"])
            set_color_contain("shin", colors["pants"])

        if "shoes" in colors:
            set_color_contain("foot", colors["shoes"])

        if "skin" in colors:
            set_color_contain("head", colors["skin"])
            set_color_contain("hand", colors["skin"])
            set_color_contain("forearm", colors["skin"])

        if "club" in colors:
            set_color_contain("club", colors["club"])

        self.sim_widget._render_once()

    def on_ellipsoid_visualization_changed(self, state: int) -> None:
        """Handle ellipsoid visualization toggle."""
        if hasattr(self, "sim_widget"):
            # Check if mobility ellipsoid checkbox is checked
            show_mobility = False
            if hasattr(self, "show_mobility_ellipsoid_cb"):
                show_mobility = self.show_mobility_ellipsoid_cb.isChecked()

            # Check if force ellipsoid checkbox is checked
            show_force = False
            if hasattr(self, "show_force_ellipsoid_cb"):
                show_force = self.show_force_ellipsoid_cb.isChecked()

            # Update visualization
            self.sim_widget.set_ellipsoid_visualization(show_mobility, show_force)

    def show_advanced_plots_dialog(self) -> None:
        """Show dialog with advanced analysis plots."""
        import numpy as np
        from PyQt6 import QtWidgets

        try:
            from shared.python.plotting import GolfSwingPlotter
            from shared.python.statistical_analysis import StatisticalAnalyzer
        except ImportError:
            from PyQt6 import QtWidgets

            QtWidgets.QMessageBox.warning(
                None, "Error", "Matplotlib or shared modules not found."
            )
            return

        if not hasattr(self, "sim_widget"):
            return

        recorder = self.sim_widget.get_recorder()
        times, _ = recorder.get_time_series("joint_positions")

        if len(times) == 0:
            from PyQt6 import QtWidgets

            QtWidgets.QMessageBox.warning(
                None,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        # Prepare analyzer
        # MuJoCo recorder should implement get_time_series correctly
        # But we need full arrays for StatisticalAnalyzer constructor

        times, positions = recorder.get_time_series("joint_positions")
        _, velocities = recorder.get_time_series("joint_velocities")
        _, torques = recorder.get_time_series("joint_torques")
        _, club_speed = recorder.get_time_series("club_head_speed")
        _, club_pos = recorder.get_time_series("club_head_position")

        # Ensure arrays
        times = np.asarray(times)
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        torques = np.asarray(torques)
        club_speed = np.asarray(club_speed)
        club_pos = np.asarray(club_pos)

        analyzer = StatisticalAnalyzer(
            times,
            positions,
            velocities,
            torques,
            club_head_speed=club_speed,
            club_head_position=club_pos,
        )
        report = analyzer.generate_comprehensive_report()

        # Plotter
        plotter = GolfSwingPlotter(recorder)

        # Metrics for Radar
        metrics = {
            "Speed": 0.0,
            "Efficiency": 0.0,
            "Tempo": 0.0,
            "Consistency": 0.0,
            "Power": 0.0,
        }

        if "club_head_speed" in report:
            peak = report["club_head_speed"]["peak_value"]
            metrics["Speed"] = min(peak / 50.0, 1.0)  # Approx 110 mph = 50 m/s

        if "energy_efficiency" in report:
            metrics["Efficiency"] = report["energy_efficiency"] / 100.0

        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            err = abs(ratio - 3.0)
            metrics["Tempo"] = max(0.0, 1.0 - err / 2.0)

        # Create dialog
        from PyQt6 import QtWidgets

        dialog = QtWidgets.QDialog(None)  # Use None as parent instead of self
        dialog.setWindowTitle("Advanced Swing Analysis")
        dialog.resize(1000, 800)
        layout = QtWidgets.QVBoxLayout(dialog)

        # Matplotlib canvas
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)

        fig.add_gridspec(2, 2)

        # 1. Radar Chart
        # This method centers the polar plot usually
        plotter.plot_radar_chart(fig, metrics)
        # But wait, plot_radar_chart in shared/python/plotting.py uses
        # add_subplot(111, polar=True)
        # which clears the figure. We need to update plot_radar_chart to accept
        # ax or modify usage.
        # Actually my implementation of plot_radar_chart used add_subplot(111).
        # Let's fix that in plotting.py first? Or just show one plot per dialog?
        # The user requested advanced features.

        # To avoid editing plotting.py again right now and risk breaking stuff,
        # let's just make tabs in the dialog for different advanced plots.

        # Wait, I cannot use tabs with a single canvas easily unless I redraw.
        # Let's rebuild the dialog structure.

        layout.removeWidget(canvas)
        canvas.deleteLater()

        tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(tab_widget)

        # Tab 1: Swing DNA
        dna_widget = QtWidgets.QWidget()
        dna_layout = QtWidgets.QVBoxLayout(dna_widget)
        fig1 = Figure(figsize=(8, 6))
        canvas1 = FigureCanvasQTAgg(fig1)
        dna_layout.addWidget(canvas1)
        plotter.plot_radar_chart(fig1, metrics)
        tab_widget.addTab(dna_widget, "Swing DNA")

        # Tab 2: CoP Field
        cop_widget = QtWidgets.QWidget()
        cop_layout = QtWidgets.QVBoxLayout(cop_widget)
        fig2 = Figure(figsize=(8, 6))
        canvas2 = FigureCanvasQTAgg(fig2)
        cop_layout.addWidget(canvas2)
        if any(f.cop_position is not None for f in recorder.frames):
            plotter.plot_cop_vector_field(fig2)
        else:
            ax = fig2.add_subplot(111)
            ax.text(0.5, 0.5, "No CoP Data", ha="center", va="center")
        tab_widget.addTab(cop_widget, "CoP Field")

        # Tab 3: Power Flow
        pwr_widget = QtWidgets.QWidget()
        pwr_layout = QtWidgets.QVBoxLayout(pwr_widget)
        fig3 = Figure(figsize=(8, 6))
        canvas3 = FigureCanvasQTAgg(fig3)
        pwr_layout.addWidget(canvas3)
        if any(f.actuator_powers.size > 0 for f in recorder.frames):
            plotter.plot_power_flow(fig3)
        else:
            ax = fig3.add_subplot(111)
            ax.text(0.5, 0.5, "No Power Data", ha="center", va="center")
        tab_widget.addTab(pwr_widget, "Power Flow")

        # Tab 4: Kinematic Sequence
        ks_widget = QtWidgets.QWidget()
        ks_layout = QtWidgets.QVBoxLayout(ks_widget)
        fig4 = Figure(figsize=(8, 6))
        canvas4 = FigureCanvasQTAgg(fig4)
        ks_layout.addWidget(canvas4)

        try:
            from shared.python.kinematic_sequence import KinematicSequenceAnalyzer

            # Define segments for the standard humanoid model
            # Note: Indices might need adjustment based on specific model loaded (full_body vs upper_body)
            # This is a best-effort mapping for standard models.
            # In a robust system, these would be defined in the model config/metadata.

            # Default mapping for full body / upper body models in this repo
            # Based on inspection of XMLs, joints are named.
            # We need to find the qvel index for specific joints.
            # joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

            # Helper to find index
            model = self.sim_widget.model

            def get_dof_index(joint_name):
                j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if j_id == -1:
                    return None
                # qvel index address
                return model.jnt_dofadr[j_id]

            # Try to map typical chain: Pelvis -> Thorax -> Arm -> Club
            # Adjust names based on actual XML joint names
            # Full Body XML: pelvis (free), spine_rotation, right_shoulder_swing, right_wrist, club_wrist
            # Upper Body XML: spine_rotation, ...

            # Let's try to detect the chain dynamically or use known names
            potential_chain = {
                "Pelvis": ["pelvis", "root", "waist"],
                "Torso": ["spine_rotation", "spine_yaw", "torso_twist"],
                "Lead Arm": ["left_shoulder_swing", "left_shoulder_flexion"], # Assuming right-handed golfer (left arm lead)
                "Club": ["club_wrist", "wrist_flexion"]
            }

            segment_indices = {}
            for seg_name, candidates in potential_chain.items():
                for cand in candidates:
                    idx = get_dof_index(cand)
                    if idx is not None:
                        segment_indices[seg_name] = idx
                        break

            if segment_indices:
                ks_analyzer = KinematicSequenceAnalyzer(expected_order=["Pelvis", "Torso", "Lead Arm", "Club"])

                # Extract velocities
                # Use analyzer helper
                ks_data, ks_times = ks_analyzer.extract_velocities_from_recorder(recorder, segment_indices)

                if ks_data:
                    ks_result = ks_analyzer.analyze(ks_data, ks_times)
                    plotter.plot_kinematic_sequence(fig4, segment_indices, analyzer_result=ks_result)
                else:
                    plotter.plot_kinematic_sequence(fig4, segment_indices)
            else:
                 ax = fig4.add_subplot(111)
                 ax.text(0.5, 0.5, "Could not map joints for kinematic sequence", ha="center", va="center")

        except Exception as e:
            logger.error(f"Failed to plot kinematic sequence: {e}")
            ax = fig4.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")

        tab_widget.addTab(ks_widget, "Kinematic Sequence")

        dialog.exec()
