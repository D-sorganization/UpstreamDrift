# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

logger = get_logger(__name__)


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
                except ImportError as e:
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
            """Set RGBA color for all geoms whose name contains a string."""
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
            from shared.python.validation_pkg.statistical_analysis import (
                StatisticalAnalyzer,
            )
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

        # Prepare data and analysis objects
        analyzer, _report, plotter, metrics = self._prepare_analysis_data(
            recorder, np, StatisticalAnalyzer, GolfSwingPlotter
        )

        # Detect pelvis/torso DOF indices for coordination tabs
        pelvis_idx, torso_idx = self._detect_pelvis_torso_indices()

        # Create dialog with tabbed plots
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyQt6 import QtWidgets

        dialog = QtWidgets.QDialog(None)
        dialog.setWindowTitle("Advanced Swing Analysis")
        dialog.resize(1000, 800)
        layout = QtWidgets.QVBoxLayout(dialog)

        tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(tab_widget)

        tab_widget.addTab(
            self._create_swing_profile_tab(plotter, metrics, Figure, FigureCanvasQTAgg),
            "Swing Profile",
        )
        tab_widget.addTab(
            self._create_cop_tab(plotter, recorder, Figure, FigureCanvasQTAgg),
            "CoP Field",
        )
        tab_widget.addTab(
            self._create_power_flow_tab(plotter, recorder, Figure, FigureCanvasQTAgg),
            "Power Flow",
        )
        tab_widget.addTab(
            self._create_kinematic_sequence_tab(
                plotter, recorder, Figure, FigureCanvasQTAgg
            ),
            "Kinematic Sequence",
        )
        tab_widget.addTab(
            self._create_coordination_tab(
                plotter, analyzer, pelvis_idx, torso_idx, Figure, FigureCanvasQTAgg
            ),
            "Coordination",
        )
        tab_widget.addTab(
            self._create_work_loop_tab(
                plotter, analyzer, torso_idx, Figure, FigureCanvasQTAgg
            ),
            "Work Loop",
        )
        tab_widget.addTab(
            self._create_ssc_tab(
                plotter, pelvis_idx, torso_idx, Figure, FigureCanvasQTAgg
            ),
            "Stretch-Shortening",
        )

        dialog.exec()

    def _prepare_analysis_data(
        self, recorder, np_mod, analyzer_cls, plotter_cls
    ) -> tuple:
        """Prepare analyzer, report, plotter, and radar metrics from recorded data."""
        times, positions = recorder.get_time_series("joint_positions")
        _, velocities = recorder.get_time_series("joint_velocities")
        _, torques = recorder.get_time_series("joint_torques")
        _, club_speed = recorder.get_time_series("club_head_speed")
        _, club_pos = recorder.get_time_series("club_head_position")

        times = np_mod.asarray(times)
        positions = np_mod.asarray(positions)
        velocities = np_mod.asarray(velocities)
        torques = np_mod.asarray(torques)
        club_speed = np_mod.asarray(club_speed)
        club_pos = np_mod.asarray(club_pos)

        analyzer = analyzer_cls(
            times,
            positions,
            velocities,
            torques,
            club_head_speed=club_speed,
            club_head_position=club_pos,
        )
        report = analyzer.generate_comprehensive_report()
        plotter = plotter_cls(recorder)

        metrics = {
            "Speed": 0.0,
            "Efficiency": 0.0,
            "Tempo": 0.0,
            "Consistency": 0.0,
            "Power": 0.0,
        }
        if "club_head_speed" in report:
            peak = report["club_head_speed"]["peak_value"]
            metrics["Speed"] = min(peak / 50.0, 1.0)
        if "energy_efficiency" in report:
            metrics["Efficiency"] = report["energy_efficiency"] / 100.0
        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            err = abs(ratio - 3.0)
            metrics["Tempo"] = max(0.0, 1.0 - err / 2.0)

        return analyzer, report, plotter, metrics

    def _detect_pelvis_torso_indices(self) -> tuple[int | None, int | None]:
        """Detect DOF indices for pelvis and torso joints in the model."""
        pelvis_idx = None
        torso_idx = None

        if not hasattr(self, "sim_widget") or self.sim_widget.model is None:
            return pelvis_idx, torso_idx

        model = self.sim_widget.model

        def get_dof_index(joint_name: str) -> int | None:
            """Return the DOF address for a named joint, or None."""
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if j_id == -1:
                return None
            return int(model.jnt_dofadr[j_id])

        for cand in ["pelvis", "root", "waist"]:
            idx = get_dof_index(cand)
            if idx is not None:
                pelvis_idx = idx
                break

        for cand in ["spine_rotation", "spine_yaw", "torso_twist"]:
            idx = get_dof_index(cand)
            if idx is not None:
                torso_idx = idx
                break

        return pelvis_idx, torso_idx

    def _create_swing_profile_tab(
        self, plotter, metrics, fig_cls, canvas_cls
    ) -> QWidget:
        """Create the Swing Profile (radar chart) tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)
        plotter.plot_radar_chart(fig, metrics)
        return widget

    def _create_cop_tab(self, plotter, recorder, fig_cls, canvas_cls) -> QWidget:
        """Create the Center of Pressure vector field tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)
        if any(f.cop_position is not None for f in recorder.frames):
            plotter.plot_cop_vector_field(fig)
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No CoP Data", ha="center", va="center")
        return widget

    def _create_power_flow_tab(self, plotter, recorder, fig_cls, canvas_cls) -> QWidget:
        """Create the Power Flow tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)
        if any(f.actuator_powers.size > 0 for f in recorder.frames):
            plotter.plot_power_flow(fig)
        else:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No Power Data", ha="center", va="center")
        return widget

    def _create_kinematic_sequence_tab(
        self, plotter, recorder, fig_cls, canvas_cls
    ) -> QWidget:
        """Create the Kinematic Sequence tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)

        try:
            from shared.python.biomechanics.kinematic_sequence import (
                KinematicSequenceAnalyzer,
            )

            model = self.sim_widget.model

            def get_dof_index(joint_name: str) -> int | None:
                """Return the DOF address for a named joint, or None."""
                j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if j_id == -1:
                    return None
                return int(model.jnt_dofadr[j_id])

            potential_chain = {
                "Pelvis": ["pelvis", "root", "waist"],
                "Torso": ["spine_rotation", "spine_yaw", "torso_twist"],
                "Lead Arm": ["left_shoulder_swing", "left_shoulder_flexion"],
                "Club": ["club_wrist", "wrist_flexion"],
            }

            segment_indices = {}
            for seg_name, candidates in potential_chain.items():
                for cand in candidates:
                    idx = get_dof_index(cand)
                    if idx is not None:
                        segment_indices[seg_name] = idx
                        break

            if segment_indices:
                ks_analyzer = KinematicSequenceAnalyzer(
                    expected_order=["Pelvis", "Torso", "Lead Arm", "Club"]
                )
                ks_data, ks_times = ks_analyzer.extract_velocities_from_recorder(
                    recorder, segment_indices
                )
                if ks_data:
                    ks_result = ks_analyzer.analyze(ks_data, ks_times)
                    plotter.plot_kinematic_sequence(
                        fig, segment_indices, analyzer_result=ks_result
                    )
                else:
                    plotter.plot_kinematic_sequence(fig, segment_indices)
            else:
                ax = fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Could not map joints for kinematic sequence",
                    ha="center",
                    va="center",
                )

        except ImportError as e:
            logger.error(f"Failed to plot kinematic sequence: {e}")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")

        return widget

    def _create_coordination_tab(
        self, plotter, analyzer, pelvis_idx, torso_idx, fig_cls, canvas_cls
    ) -> QWidget:
        """Create the Coordination (Angle-Angle and Vector Coding) tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)

        try:
            if pelvis_idx is not None and torso_idx is not None:
                gs = fig.add_gridspec(2, 1)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0])

                plotter.plot_angle_angle_diagram(
                    fig,
                    pelvis_idx,
                    torso_idx,
                    title="Coordination: Pelvis vs Torso (Angle-Angle)",
                    ax=ax1,
                )

                coupling_angles = analyzer.compute_coupling_angles(
                    pelvis_idx, torso_idx
                )
                plotter.plot_coupling_angle(
                    fig,
                    coupling_angles,
                    title="Coupling Angle (Vector Coding)",
                    ax=ax2,
                )

                fig.tight_layout()
            else:
                ax = fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Could not identify Pelvis/Torso for Coordination",
                    ha="center",
                    va="center",
                )

        except ImportError as e:
            logger.error(f"Failed to plot coordination: {e}")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")

        return widget

    def _create_work_loop_tab(
        self, plotter, analyzer, torso_idx, fig_cls, canvas_cls
    ) -> QWidget:
        """Create the Work Loop (Energetics) tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)

        target_work_idx = torso_idx if torso_idx is not None else 0
        plotter.plot_work_loop(fig, target_work_idx)

        work_metrics = analyzer.compute_work_metrics(target_work_idx)
        if work_metrics:
            info_text = (
                f"Net Work: {work_metrics['net_work']:.1f} J\n"
                f"Pos Work: {work_metrics['positive_work']:.1f} J\n"
                f"Neg Work: {work_metrics['negative_work']:.1f} J"
            )
            ax_w = fig.gca()
            ax_w.text(
                0.05,
                0.95,
                info_text,
                transform=ax_w.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

        return widget

    def _create_ssc_tab(
        self, plotter, pelvis_idx, torso_idx, fig_cls, canvas_cls
    ) -> QWidget:
        """Create the Stretch-Shortening Cycle (X-Factor) tab widget."""
        from PyQt6 import QtWidgets

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        fig = fig_cls(figsize=(8, 6))
        canvas = canvas_cls(fig)
        layout.addWidget(canvas)

        if pelvis_idx is not None and torso_idx is not None:
            plotter.plot_x_factor_cycle(fig, torso_idx, pelvis_idx)
        else:
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Could not identify Pelvis/Torso for SSC",
                ha="center",
                va="center",
            )

        return widget
