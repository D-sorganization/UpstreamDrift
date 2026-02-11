"""Drake GUI visualization mixin.

Extracts vector drawing, ellipsoid rendering, analysis plots,
and data export from DrakeSimApp (drake_gui_app.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    MATPLOTLIB_AVAILABLE,
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger

HAS_QT = PYQT6_AVAILABLE
HAS_MATPLOTLIB = MATPLOTLIB_AVAILABLE

if HAS_QT:
    from PyQt6 import QtCore, QtWidgets

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

# Drake imports
if TYPE_CHECKING or HAS_QT:
    try:
        from pydrake.all import (
            BodyIndex,
            Ellipsoid,
            JacobianWrtVariable,
            JointIndex,
            Rgba,
            RigidTransform,
            RotationMatrix,
        )
    except ImportError:
        BodyIndex = None  # type: ignore[misc, assignment]
        Ellipsoid = None  # type: ignore[misc, assignment]
        JacobianWrtVariable = None  # type: ignore[misc, assignment]
        JointIndex = None  # type: ignore[misc, assignment]
        Rgba = None  # type: ignore[misc, assignment]
        RigidTransform = None  # type: ignore[misc, assignment]
        RotationMatrix = None  # type: ignore[misc, assignment]

LOGGER = get_logger(__name__)


class DrakeVisualizationMixin:
    """Mixin for Drake GUI visualization, analysis plots, and data export.

    Provides:
    - Vector drawing (forces, torques, induced accel, counterfactuals)
    - Ellipsoid rendering (mobility, force)
    - Manipulability checkbox management
    - Analysis plot generation (induced accel, counterfactual, swing plane, advanced)
    - Data export to CSV/multi-format
    """

    def _on_visualization_changed(self: Any) -> None:
        """Handle toggling of visualization options."""
        self._update_visualization()

    def _update_visualization(self: Any) -> None:
        """Update all visualizations (ellipsoids, vectors)."""
        if not self.meshcat or not self.plant or not self.context:
            return

        if self.visualizer:
            self.visualizer.update_frame_transforms(self.context)
            self.visualizer.update_com_transforms(self.context)

        # Draw Ellipsoids
        self._draw_ellipsoids()

        # Clear old ellipsoids/vectors if needed
        if not (self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked()):
            self.meshcat.Delete("overlays/ellipsoids")

        if not (
            self.chk_induced_vec.isChecked()
            or self.chk_cf_vec.isChecked()
            or self.chk_show_forces.isChecked()
            or self.chk_show_torques.isChecked()
        ):
            self.meshcat.Delete("overlays/vectors")

        self._update_ellipsoids()
        self._update_vectors()

    def _update_vectors(self: Any) -> None:
        """Draw advanced vectors (Forces, Torques, Induced, CF)."""
        if not self.plant or not self.eval_context:
            return

        # Explicit cleanup of disabled categories
        if self.meshcat is not None:
            if not self.chk_show_torques.isChecked():
                self.meshcat.Delete("overlays/vectors/torques")
            if not self.chk_show_forces.isChecked():
                self.meshcat.Delete("overlays/vectors/forces")
            if not self.chk_induced_vec.isChecked():
                self.meshcat.Delete("overlays/vectors/induced")
            if not self.chk_cf_vec.isChecked():
                self.meshcat.Delete("overlays/vectors/cf")

        # Use eval context synced with current state
        plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.plant.SetPositions(
            self.eval_context, self.plant.GetPositions(plant_context)
        )
        self.plant.SetVelocities(
            self.eval_context, self.plant.GetVelocities(plant_context)
        )

        # Import the analyzer from the recorder module
        from .drake_recorder import DrakeInducedAccelerationAnalyzer

        # 1. Standard Torques (Blue)
        if self.chk_show_torques.isChecked():
            # Visualize gravity compensation torques (holding torque)
            tau = self.plant.CalcGravityGeneralizedForces(self.eval_context)
            self._draw_accel_vectors(-tau, "torques", Rgba(0, 0, 1, 1), scale=0.05)

        # 2. Standard Forces (Green) - Visualize Gravity Force at COM
        if self.chk_show_forces.isChecked():
            for i in range(self.plant.num_bodies()):
                body = self.plant.get_body(BodyIndex(i))
                if body.name() == "world":
                    continue

                mass = body.get_mass(self.eval_context)
                if mass <= 1e-6:
                    continue

                # Gravity force = mass * g (down Z)
                gravity = self.plant.gravity_field().gravity_vector()
                force_vec = gravity * mass

                # Draw at COM
                X_WB = self.plant.EvalBodyPoseInWorld(self.eval_context, body)
                com_B = body.CalcCenterOfMassInBodyFrame(self.eval_context)
                pos_W = X_WB.multiply(com_B)

                scale = 0.01  # Force scale
                end_pos = pos_W + force_vec * scale

                points = np.vstack([pos_W, end_pos]).T
                path = f"overlays/vectors/forces/{body.name()}"
                if self.meshcat is not None:
                    self.meshcat.SetLineSegments(path, points, 2.0, Rgba(0, 1, 0, 1))

        # 3. Advanced Vectors (Induced / CF)
        if not (self.chk_induced_vec.isChecked() or self.chk_cf_vec.isChecked()):
            return

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)

        # Induced
        if self.chk_induced_vec.isChecked():
            source = self.combo_induced_source.currentText()
            accels = np.zeros(self.plant.num_velocities())

            if source in ["gravity", "velocity", "total"]:
                res = analyzer.compute_components(self.eval_context)
                accels = res.get(source, accels)
            else:
                # Specific actuator by name or index
                tau = np.zeros(self.plant.num_velocities())
                found = False
                # Try name match
                if self.plant.HasJointNamed(source):
                    joint = self.plant.GetJointByName(source)
                    if joint.num_velocities() == 1:
                        v_idx = joint.velocity_start()
                        tau[v_idx] = 1.0
                        found = True

                if not found:
                    try:
                        act_idx = int(source)
                        if 0 <= act_idx < len(tau):
                            tau[act_idx] = 1.0
                            found = True
                    except ValueError:
                        pass

                if found:
                    accels = analyzer.compute_specific_control(self.eval_context, tau)

            self._draw_accel_vectors(accels, "induced", Rgba(1, 0, 1, 1))

        # Counterfactuals
        if self.chk_cf_vec.isChecked():
            cf_type = self.combo_cf_type.currentText()
            res = analyzer.compute_counterfactuals(self.eval_context)

            # Default to ZTCF accel if not found
            if cf_type == "ztcf_accel":
                vals = res.get("ztcf_accel", np.zeros(self.plant.num_velocities()))
                self._draw_accel_vectors(vals, "cf", Rgba(1, 1, 0, 1))
            elif cf_type == "zvcf_torque":
                vals = res.get("zvcf_torque", np.zeros(self.plant.num_velocities()))
                # Visualize torque as vectors? reusing accel visualizer for now (scaled)
                self._draw_accel_vectors(vals, "cf", Rgba(1, 1, 0, 1))

    def _draw_accel_vectors(
        self: Any,
        values: np.ndarray,
        name_prefix: str,
        color: Rgba,
        scale: float = 0.1,
    ) -> None:
        """Draw vectors at joints (accel, torque, etc)."""
        if not self.meshcat or self.plant is None:
            return

        for i in range(self.plant.num_joints()):
            joint = self.plant.get_joint(JointIndex(i))
            if joint.num_velocities() != 1:
                continue

            # Map to velocity index
            v_start = joint.velocity_start()
            val = values[v_start]
            if abs(val) < 1e-3:
                continue

            # Get joint frame
            frame_J = joint.frame_on_child()
            if self.plant is not None and self.eval_context is not None:
                X_WJ = self.plant.EvalBodyPoseInWorld(self.eval_context, frame_J.body())
                start_pos = X_WJ.translation()
            else:
                continue

            # Axis direction
            if hasattr(joint, "revolute_axis"):
                axis_C = joint.revolute_axis()
            elif hasattr(joint, "translation_axis"):
                axis_C = joint.translation_axis()
            else:
                continue

            axis_W = X_WJ.rotation().multiply(axis_C)

            vector = axis_W * val * scale
            end_pos = start_pos + vector

            # Draw line
            path = f"overlays/vectors/{name_prefix}/{joint.name()}"

            # Meshcat SetLineSegments expects 3xN array
            points = np.vstack([start_pos, end_pos]).T
            self.meshcat.SetLineSegments(path, points, 2.0, color)

    def _update_ellipsoids(self: Any) -> None:
        """Compute and draw ellipsoids."""
        if not (self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked()):
            return

        if not self.plant or not self.context:
            return

        plant_context = self.plant.GetMyContextFromRoot(self.context)

        body_names = ["clubhead", "club_body", "wrist", "hand", "link_7"]
        target_body = None
        for name in body_names:
            if self.plant.HasBodyNamed(name):
                target_body = self.plant.GetBodyByName(name)
                break

        if target_body is None:
            target_body = self.plant.get_body(BodyIndex(self.plant.num_bodies() - 1))

        if target_body.name() == "world":
            return

        frame_W = self.plant.world_frame()
        frame_B = target_body.body_frame()

        J_spatial = self.plant.CalcJacobianSpatialVelocity(
            plant_context,
            JacobianWrtVariable.kV,
            frame_B,
            [0, 0, 0],
            frame_W,
            frame_W,
        )
        J = J_spatial[3:, :]  # Translational

        M = self.plant.CalcMassMatrix(plant_context)

        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
            self.lbl_cond.setText(f"{cond:.2f}")

            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.plant.num_velocities()}")

            Minv = np.linalg.inv(M)
            Lambda_inv = J @ Minv @ J.T

            eigvals, eigvecs = np.linalg.eigh(Lambda_inv)

            X_WB = self.plant.EvalBodyPoseInWorld(plant_context, target_body)
            pos = X_WB.translation()

            if self.meshcat:
                if self.chk_mobility.isChecked():
                    radii = np.sqrt(np.maximum(eigvals, 1e-6))
                    path = "overlays/ellipsoids/mobility"

                    # Create ellipsoid geometry directly
                    ellipsoid = Ellipsoid(radii[0], radii[1], radii[2])
                    self.meshcat.SetObject(path, ellipsoid, Rgba(0, 1, 0, 0.3))

                    # Transform (Rotation + Position)
                    R = RotationMatrix(eigvecs)
                    T = RigidTransform(R, pos)
                    self.meshcat.SetTransform(path, T)

                if self.chk_force_ellip.isChecked():
                    radii_f = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))
                    radii_f = np.clip(radii_f, 0.01, 5.0)
                    path = "overlays/ellipsoids/force"

                    ellipsoid_f = Ellipsoid(radii_f[0], radii_f[1], radii_f[2])
                    self.meshcat.SetObject(path, ellipsoid_f, Rgba(1, 0, 0, 0.3))

                    R = RotationMatrix(eigvecs)
                    T = RigidTransform(R, pos)
                    self.meshcat.SetTransform(path, T)

        except (ValueError, TypeError, RuntimeError) as e:
            LOGGER.warning("Ellipsoid calc error: %s", e)

    def _populate_manip_checkboxes(self: Any) -> None:
        """Populate checkboxes for manipulability analysis."""
        if not self.manip_analyzer or not self.manip_body_layout:
            return

        # Clear existing
        while self.manip_body_layout.count():
            item = self.manip_body_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.manip_checkboxes.clear()

        bodies = self.manip_analyzer.find_potential_bodies()

        cols = 3
        for i, name in enumerate(bodies):
            chk = QtWidgets.QCheckBox(name)
            chk.toggled.connect(self._on_visualization_changed)
            self.manip_checkboxes[name] = chk
            self.manip_body_layout.addWidget(chk, i // cols, i % cols)

            # Default check relevant parts
            if any(x in name.lower() for x in ["club", "hand", "wrist"]):
                chk.setChecked(True)

    def _draw_ellipsoids(self: Any) -> None:
        """Draw force/mobility ellipsoids using Meshcat."""
        if (
            not self.meshcat
            or not self.manip_analyzer
            or not self.context
            or not self.plant
        ):
            return

        # We'll use a specific path prefix
        prefix = "ellipsoids"

        # Check if enabled
        show_m = self.chk_mobility.isChecked()
        show_f = self.chk_force_ellip.isChecked()

        if not (show_m or show_f):
            self.meshcat.Delete(prefix)
            return

        # Get selected
        selected = [n for n, c in self.manip_checkboxes.items() if c.isChecked()]
        if not selected:
            self.meshcat.Delete(prefix)
            return

        # Compute
        results = self.manip_analyzer.compute_metrics(self.context, selected)

        # Draw
        for res in results:
            name = res.body_name
            # Mobility
            if show_m and res.mobility_ellipsoid:
                path = f"{prefix}/{name}/mobility"
                radii = res.mobility_ellipsoid.radii
                # Scale for viz
                scale = 0.5
                radii_viz = radii * scale

                # Check for NaNs or zeros
                if np.any(radii_viz <= 1e-9) or np.any(np.isnan(radii_viz)):
                    continue

                shape = Ellipsoid(radii_viz[0], radii_viz[1], radii_viz[2])
                color = Rgba(0.0, 1.0, 0.0, 0.5)

                # Pose
                R_matrix = RotationMatrix(res.mobility_ellipsoid.axes)
                X_WE = RigidTransform(R_matrix, res.mobility_ellipsoid.center)

                self.meshcat.SetObject(path, shape, color)
                self.meshcat.SetTransform(path, X_WE)
            else:
                self.meshcat.Delete(f"{prefix}/{name}/mobility")

            # Force
            if show_f and res.force_ellipsoid:
                path = f"{prefix}/{name}/force"
                radii = res.force_ellipsoid.radii
                scale = 0.1  # Force ellipsoids can be huge
                radii_viz = radii * scale

                if np.any(radii_viz <= 1e-9) or np.any(np.isnan(radii_viz)):
                    continue

                shape = Ellipsoid(radii_viz[0], radii_viz[1], radii_viz[2])
                color = Rgba(1.0, 0.0, 0.0, 0.5)

                R_matrix = RotationMatrix(res.force_ellipsoid.axes)
                X_WE = RigidTransform(R_matrix, res.force_ellipsoid.center)

                self.meshcat.SetObject(path, shape, color)
                self.meshcat.SetTransform(path, X_WE)
            else:
                self.meshcat.Delete(f"{prefix}/{name}/force")

    # ------------------------------------------------------------------
    # Analysis plot methods
    # ------------------------------------------------------------------

    def _show_induced_acceleration_plot(self: Any) -> None:
        """Calculate and plot induced accelerations."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        from shared.python.plotting import GolfSwingPlotter

        from .drake_recorder import DrakeInducedAccelerationAnalyzer

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        if not self.plant or not self.eval_context:
            return

        spec_act_idx = -1
        txt = self.combo_induced_source.currentText()
        if txt and txt not in ["gravity", "velocity", "total"]:
            try:
                spec_act_idx = int(txt)
            except ValueError:
                pass

        g_induced = []
        c_induced = []
        spec_induced = []
        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            for _i, (q, v) in enumerate(
                zip(self.recorder.q_history, self.recorder.v_history, strict=False)
            ):
                self.plant.SetPositions(self.eval_context, q)
                self.plant.SetVelocities(self.eval_context, v)

                res = analyzer.compute_components(self.eval_context)

                g_induced.append(res["gravity"])
                c_induced.append(res["velocity"])

                if spec_act_idx >= 0:
                    tau = np.zeros(self.plant.num_velocities())
                    if spec_act_idx < len(tau):
                        tau[spec_act_idx] = 1.0
                    spec = analyzer.compute_specific_control(self.eval_context, tau)
                    spec_induced.append(spec)

        except (ValueError, TypeError, RuntimeError) as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        g_induced_arr = np.array(g_induced)
        c_induced_arr = np.array(c_induced)
        total_arr = g_induced_arr + c_induced_arr

        self.recorder.induced_accelerations["gravity"] = list(g_induced_arr)
        self.recorder.induced_accelerations["velocity"] = list(c_induced_arr)
        self.recorder.induced_accelerations["total"] = list(total_arr)

        if spec_induced:
            self.recorder.induced_accelerations["control"] = list(
                np.array(spec_induced)
            )

        joint_idx = 0
        if g_induced_arr.shape[1] > 2:
            joint_idx = 2

        plotter = GolfSwingPlotter(self.recorder)
        fig = plt.figure(figsize=(10, 6))

        plotter.plot_induced_acceleration(
            fig, "breakdown", joint_idx=joint_idx, breakdown_mode=True
        )
        plt.show()

    def _show_counterfactuals_plot(self: Any) -> None:
        """Calculate and plot Counterfactuals (ZTCF/ZVCF)."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        from shared.python.plotting import GolfSwingPlotter

        from .drake_recorder import DrakeInducedAccelerationAnalyzer

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        if not self.plant or not self.eval_context:
            return

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)
        ztcf_list = []
        zvcf_list = []

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

            for q, v in zip(
                self.recorder.q_history, self.recorder.v_history, strict=False
            ):
                self.plant.SetPositions(self.eval_context, q)
                self.plant.SetVelocities(self.eval_context, v)

                res = analyzer.compute_counterfactuals(self.eval_context)
                ztcf_list.append(res["ztcf_accel"])
                zvcf_list.append(res["zvcf_torque"])

        except (RuntimeError, ValueError, OSError) as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.recorder.counterfactuals["ztcf_accel"] = list(np.array(ztcf_list))
        self.recorder.counterfactuals["zvcf_torque"] = list(np.array(zvcf_list))

        joint_idx = 0
        if np.array(ztcf_list).shape[1] > 2:
            joint_idx = 2

        plotter = GolfSwingPlotter(self.recorder)
        fig = plt.figure(figsize=(10, 6))

        plotter.plot_counterfactual_comparison(fig, "dual", metric_idx=joint_idx)
        plt.show()

    def _export_data(self: Any) -> None:
        """Export recorded data to multiple formats."""
        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "drake_sim_data", "All Files (*)"
        )
        if not filename:
            return

        try:
            from shared.python.data_io.export import export_recording_all_formats

            data_dict = self.recorder.export_to_dict()
            results = export_recording_all_formats(filename, data_dict)

            msg = "Export Results:\n"
            for fmt, success in results.items():
                msg += f"{fmt}: {'Success' if success else 'Failed'}\n"

            QtWidgets.QMessageBox.information(self, "Export Complete", msg)
            self._update_status(f"Data exported to {filename}")

        except ImportError as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))
            LOGGER.error("Export failed: %s", e)

    def _show_swing_plane_analysis(self: Any) -> None:
        """Show swing plane analysis using shared plotter."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        from shared.python.plotting import GolfSwingPlotter

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        plotter = GolfSwingPlotter(self.recorder)
        fig = plt.figure(figsize=(10, 8))
        plotter.plot_swing_plane(fig)
        plt.show()

    def _show_advanced_plots(self: Any) -> None:
        """Show advanced analysis plots."""
        if not HAS_MATPLOTLIB:
            QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib not found.")
            return

        from shared.python.plotting import GolfSwingPlotter
        from shared.python.validation_pkg.statistical_analysis import (
            StatisticalAnalyzer,
        )

        if not self.recorder.times:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No recording available. Please Record a simulation first.",
            )
            return

        plotter = GolfSwingPlotter(self.recorder)

        times = np.array(self.recorder.times)
        q_history = np.array(self.recorder.q_history)
        v_history = np.array(self.recorder.v_history)
        tau_history = np.zeros((len(times), v_history.shape[1]))

        _, club_pos = self.recorder.get_time_series("club_head_position")

        # Convert club position to numpy array if needed
        club_head_pos = np.array(club_pos) if isinstance(club_pos, list) else club_pos

        analyzer = StatisticalAnalyzer(
            times, q_history, v_history, tau_history, club_head_position=club_head_pos
        )
        report = analyzer.generate_comprehensive_report()

        metrics = {
            "Club Speed": 0.0,
            "Swing Efficiency": 0.0,
            "Tempo": 0.0,
            "Consistency": 0.8,
            "Power Transfer": 0.0,
        }

        if "club_head_speed" in report:
            peak_speed = report["club_head_speed"]["peak_value"]
            metrics["Club Speed"] = min(peak_speed / 50.0, 1.0)

        if "tempo" in report:
            ratio = report["tempo"]["ratio"]
            error = abs(ratio - 3.0)
            metrics["Tempo"] = max(0, 1.0 - error)

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        plotter.plot_radar_chart(fig, metrics)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, "CoP Data Not Available in Drake", ha="center", va="center")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(
            0.5, 0.5, "Power Data Not Available in Drake", ha="center", va="center"
        )

        plt.tight_layout()
        plt.show()
