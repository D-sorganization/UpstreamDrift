"""Drake GUI Visualization Mixin.

Contains all Meshcat visualization methods: vectors, ellipsoids, and overlays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger

HAS_QT = PYQT6_AVAILABLE

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

from .drake_analysis import DrakeInducedAccelerationAnalyzer  # noqa: E402

LOGGER = get_logger(__name__)


class VisualizationMixin:
    """Mixin providing all Meshcat visualization methods for DrakeSimApp."""

    def _on_visualization_changed(self) -> None:
        """Handle toggling of visualization options."""
        self._update_visualization()

    def _update_visualization(self) -> None:
        """Update all visualizations (ellipsoids, vectors)."""
        if not self.meshcat or not self.plant or not self.context:  # type: ignore[attr-defined]
            return

        if self.visualizer:  # type: ignore[attr-defined]
            self.visualizer.update_frame_transforms(self.context)  # type: ignore[attr-defined]
            self.visualizer.update_com_transforms(self.context)  # type: ignore[attr-defined]

        # Draw Ellipsoids
        self._draw_ellipsoids()

        # Clear old ellipsoids/vectors if needed
        if not (self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked()):  # type: ignore[attr-defined]
            self.meshcat.Delete("overlays/ellipsoids")  # type: ignore[attr-defined]

        if not (
            self.chk_induced_vec.isChecked()  # type: ignore[attr-defined]
            or self.chk_cf_vec.isChecked()  # type: ignore[attr-defined]
            or self.chk_show_forces.isChecked()  # type: ignore[attr-defined]
            or self.chk_show_torques.isChecked()  # type: ignore[attr-defined]
        ):
            self.meshcat.Delete("overlays/vectors")  # type: ignore[attr-defined]

        self._update_ellipsoids()
        self._update_vectors()

    def _cleanup_disabled_vector_categories(self) -> None:
        if self.meshcat is not None:  # type: ignore[attr-defined]
            if not self.chk_show_torques.isChecked():  # type: ignore[attr-defined]
                self.meshcat.Delete("overlays/vectors/torques")  # type: ignore[attr-defined]
            if not self.chk_show_forces.isChecked():  # type: ignore[attr-defined]
                self.meshcat.Delete("overlays/vectors/forces")  # type: ignore[attr-defined]
            if not self.chk_induced_vec.isChecked():  # type: ignore[attr-defined]
                self.meshcat.Delete("overlays/vectors/induced")  # type: ignore[attr-defined]
            if not self.chk_cf_vec.isChecked():  # type: ignore[attr-defined]
                self.meshcat.Delete("overlays/vectors/cf")  # type: ignore[attr-defined]

    def _sync_eval_context(self) -> None:
        assert self.plant is not None  # type: ignore[attr-defined]
        assert self.context is not None  # type: ignore[attr-defined]
        assert self.eval_context is not None  # type: ignore[attr-defined]
        plant_context = self.plant.GetMyContextFromRoot(self.context)  # type: ignore[attr-defined]
        self.plant.SetPositions(  # type: ignore[attr-defined]
            self.eval_context,
            self.plant.GetPositions(plant_context),  # type: ignore[attr-defined]
        )
        self.plant.SetVelocities(  # type: ignore[attr-defined]
            self.eval_context,
            self.plant.GetVelocities(plant_context),  # type: ignore[attr-defined]
        )

    def _draw_torque_vectors(self) -> None:
        assert self.plant is not None  # type: ignore[attr-defined]
        assert self.eval_context is not None  # type: ignore[attr-defined]
        tau = self.plant.CalcGravityGeneralizedForces(self.eval_context)  # type: ignore[attr-defined]
        self._draw_accel_vectors(-tau, "torques", Rgba(0, 0, 1, 1), scale=0.05)

    def _draw_gravity_force_vectors(self) -> None:
        assert self.plant is not None  # type: ignore[attr-defined]
        assert self.eval_context is not None  # type: ignore[attr-defined]
        for i in range(self.plant.num_bodies()):  # type: ignore[attr-defined]
            body = self.plant.get_body(BodyIndex(i))  # type: ignore[attr-defined]
            if body.name() == "world":
                continue

            mass = body.get_mass(self.eval_context)  # type: ignore[attr-defined]
            if mass <= 1e-6:
                continue

            gravity = self.plant.gravity_field().gravity_vector()  # type: ignore[attr-defined]
            force_vec = gravity * mass

            X_WB = self.plant.EvalBodyPoseInWorld(self.eval_context, body)  # type: ignore[attr-defined]
            com_B = body.CalcCenterOfMassInBodyFrame(self.eval_context)  # type: ignore[attr-defined]
            pos_W = X_WB.multiply(com_B)

            scale = 0.01
            end_pos = pos_W + force_vec * scale

            points = np.vstack([pos_W, end_pos]).T
            path = f"overlays/vectors/forces/{body.name()}"
            if self.meshcat is not None:  # type: ignore[attr-defined]
                self.meshcat.SetLineSegments(path, points, 2.0, Rgba(0, 1, 0, 1))  # type: ignore[attr-defined, arg-type]

    def _resolve_induced_accels(self, analyzer: Any, source: str) -> np.ndarray:
        assert self.plant is not None  # type: ignore[attr-defined]
        accels = np.zeros(self.plant.num_velocities())  # type: ignore[attr-defined]

        if source in ["gravity", "velocity", "total"]:
            res = analyzer.compute_components(self.eval_context)  # type: ignore[attr-defined]
            accels = res.get(source, accels)
        else:
            tau = np.zeros(self.plant.num_velocities())  # type: ignore[attr-defined]
            found = False
            if self.plant.HasJointNamed(source):  # type: ignore[attr-defined]
                joint = self.plant.GetJointByName(source)  # type: ignore[attr-defined]
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
                accels = analyzer.compute_specific_control(self.eval_context, tau)  # type: ignore[attr-defined]

        return accels

    def _draw_induced_vectors(self, analyzer: Any) -> None:
        source = self.combo_induced_source.currentText()  # type: ignore[attr-defined]
        accels = self._resolve_induced_accels(analyzer, source)
        self._draw_accel_vectors(accels, "induced", Rgba(1, 0, 1, 1))

    def _draw_counterfactual_vectors(self, analyzer: Any) -> None:
        assert self.plant is not None  # type: ignore[attr-defined]
        cf_type = self.combo_cf_type.currentText()  # type: ignore[attr-defined]
        res = analyzer.compute_counterfactuals(self.eval_context)  # type: ignore[attr-defined]

        if cf_type == "ztcf_accel":
            vals = res.get("ztcf_accel", np.zeros(self.plant.num_velocities()))  # type: ignore[attr-defined]
            self._draw_accel_vectors(vals, "cf", Rgba(1, 1, 0, 1))
        elif cf_type == "zvcf_torque":
            vals = res.get("zvcf_torque", np.zeros(self.plant.num_velocities()))  # type: ignore[attr-defined]
            self._draw_accel_vectors(vals, "cf", Rgba(1, 1, 0, 1))

    def _update_vectors(self) -> None:
        """Draw advanced vectors (Forces, Torques, Induced, CF)."""
        if not self.plant or not self.eval_context:  # type: ignore[attr-defined]
            return

        self._cleanup_disabled_vector_categories()
        self._sync_eval_context()

        if self.chk_show_torques.isChecked():  # type: ignore[attr-defined]
            self._draw_torque_vectors()

        if self.chk_show_forces.isChecked():  # type: ignore[attr-defined]
            self._draw_gravity_force_vectors()

        if not (self.chk_induced_vec.isChecked() or self.chk_cf_vec.isChecked()):  # type: ignore[attr-defined]
            return

        analyzer = DrakeInducedAccelerationAnalyzer(self.plant)  # type: ignore[attr-defined]

        if self.chk_induced_vec.isChecked():  # type: ignore[attr-defined]
            self._draw_induced_vectors(analyzer)

        if self.chk_cf_vec.isChecked():  # type: ignore[attr-defined]
            self._draw_counterfactual_vectors(analyzer)

    def _draw_accel_vectors(
        self,
        values: np.ndarray,
        name_prefix: str,
        color: Any,
        scale: float = 0.1,
    ) -> None:
        """Draw vectors at joints (accel, torque, etc)."""
        if not self.meshcat or self.plant is None:  # type: ignore[attr-defined]
            return

        for i in range(self.plant.num_joints()):  # type: ignore[attr-defined]
            joint = self.plant.get_joint(JointIndex(i))  # type: ignore[attr-defined]
            if joint.num_velocities() != 1:
                continue

            # Map to velocity index
            v_start = joint.velocity_start()
            val = values[v_start]
            if abs(val) < 1e-3:
                continue

            # Get joint frame
            frame_J = joint.frame_on_child()
            if self.plant is not None and self.eval_context is not None:  # type: ignore[attr-defined]
                X_WJ = self.plant.EvalBodyPoseInWorld(self.eval_context, frame_J.body())  # type: ignore[attr-defined]
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
            self.meshcat.SetLineSegments(path, points, 2.0, color)  # type: ignore[attr-defined, arg-type]

    def _update_ellipsoids(self) -> None:
        """Compute and draw ellipsoids."""
        if not (self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked()):  # type: ignore[attr-defined]
            return

        if not self.plant or not self.context:  # type: ignore[attr-defined]
            return

        plant_context = self.plant.GetMyContextFromRoot(self.context)  # type: ignore[attr-defined]

        body_names = ["clubhead", "club_body", "wrist", "hand", "link_7"]
        target_body = None
        for name in body_names:
            if self.plant.HasBodyNamed(name):  # type: ignore[attr-defined]
                target_body = self.plant.GetBodyByName(name)  # type: ignore[attr-defined]
                break

        if target_body is None:
            target_body = self.plant.get_body(BodyIndex(self.plant.num_bodies() - 1))  # type: ignore[attr-defined]

        if target_body.name() == "world":
            return

        frame_W = self.plant.world_frame()  # type: ignore[attr-defined]
        frame_B = target_body.body_frame()

        J_spatial = self.plant.CalcJacobianSpatialVelocity(  # type: ignore[attr-defined]
            plant_context,
            JacobianWrtVariable.kV,
            frame_B,
            np.array([0, 0, 0]),
            frame_W,
            frame_W,  # type: ignore[arg-type]
        )
        J = J_spatial[3:, :]  # Translational

        M = self.plant.CalcMassMatrix(plant_context)  # type: ignore[attr-defined]

        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
            self.lbl_cond.setText(f"{cond:.2f}")  # type: ignore[attr-defined]

            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.plant.num_velocities()}")  # type: ignore[attr-defined]

            Minv = np.linalg.inv(M)
            Lambda_inv = J @ Minv @ J.T

            eigvals, eigvecs = np.linalg.eigh(Lambda_inv)

            X_WB = self.plant.EvalBodyPoseInWorld(plant_context, target_body)  # type: ignore[attr-defined]
            pos = X_WB.translation()

            if self.meshcat:  # type: ignore[attr-defined]
                if self.chk_mobility.isChecked():  # type: ignore[attr-defined]
                    radii = np.sqrt(np.maximum(eigvals, 1e-6))
                    path = "overlays/ellipsoids/mobility"

                    ellipsoid = Ellipsoid(radii[0], radii[1], radii[2])
                    self.meshcat.SetObject(path, ellipsoid, Rgba(0, 1, 0, 0.3))  # type: ignore[attr-defined]

                    R = RotationMatrix(eigvecs)
                    T = RigidTransform(R, pos)
                    self.meshcat.SetTransform(path, T)  # type: ignore[attr-defined]

                if self.chk_force_ellip.isChecked():  # type: ignore[attr-defined]
                    radii_f = 1.0 / np.sqrt(np.maximum(eigvals, 1e-6))
                    radii_f = np.clip(radii_f, 0.01, 5.0)
                    path = "overlays/ellipsoids/force"

                    ellipsoid_f = Ellipsoid(radii_f[0], radii_f[1], radii_f[2])
                    self.meshcat.SetObject(path, ellipsoid_f, Rgba(1, 0, 0, 0.3))  # type: ignore[attr-defined]

                    R = RotationMatrix(eigvecs)
                    T = RigidTransform(R, pos)
                    self.meshcat.SetTransform(path, T)  # type: ignore[attr-defined]

        except (ValueError, TypeError, RuntimeError) as e:
            LOGGER.warning(f"Ellipsoid calc error: {e}")

    def _draw_ellipsoids(self) -> None:
        """Draw force/mobility ellipsoids using Meshcat."""
        if (
            not self.meshcat  # type: ignore[attr-defined]
            or not self.manip_analyzer  # type: ignore[attr-defined]
            or not self.context  # type: ignore[attr-defined]
            or not self.plant  # type: ignore[attr-defined]
        ):
            return

        prefix = "ellipsoids"

        # Check if enabled
        show_m = self.chk_mobility.isChecked()  # type: ignore[attr-defined]
        show_f = self.chk_force_ellip.isChecked()  # type: ignore[attr-defined]

        if not (show_m or show_f):
            self.meshcat.Delete(prefix)  # type: ignore[attr-defined]
            return

        # Get selected
        selected = [n for n, c in self.manip_checkboxes.items() if c.isChecked()]  # type: ignore[attr-defined]
        if not selected:
            self.meshcat.Delete(prefix)  # type: ignore[attr-defined]
            return

        # Compute
        results = self.manip_analyzer.compute_metrics(self.context, selected)  # type: ignore[attr-defined]

        # Draw
        for res in results:
            name = res.body_name
            # Mobility
            if show_m and res.mobility_ellipsoid:
                path = f"{prefix}/{name}/mobility"
                radii = res.mobility_ellipsoid.radii
                scale = 0.5
                radii_viz = radii * scale

                if np.any(radii_viz <= 1e-9) or np.any(np.isnan(radii_viz)):
                    continue

                shape = Ellipsoid(radii_viz[0], radii_viz[1], radii_viz[2])
                color = Rgba(0.0, 1.0, 0.0, 0.5)

                R_matrix = RotationMatrix(res.mobility_ellipsoid.axes)
                X_WE = RigidTransform(R_matrix, res.mobility_ellipsoid.center)

                self.meshcat.SetObject(path, shape, color)  # type: ignore[attr-defined]
                self.meshcat.SetTransform(path, X_WE)  # type: ignore[attr-defined]
            else:
                self.meshcat.Delete(f"{prefix}/{name}/mobility")  # type: ignore[attr-defined]

            # Force
            if show_f and res.force_ellipsoid:
                path = f"{prefix}/{name}/force"
                radii = res.force_ellipsoid.radii
                scale = 0.1
                radii_viz = radii * scale

                if np.any(radii_viz <= 1e-9) or np.any(np.isnan(radii_viz)):
                    continue

                shape = Ellipsoid(radii_viz[0], radii_viz[1], radii_viz[2])
                color = Rgba(1.0, 0.0, 0.0, 0.5)

                R_matrix = RotationMatrix(res.force_ellipsoid.axes)
                X_WE = RigidTransform(R_matrix, res.force_ellipsoid.center)

                self.meshcat.SetObject(path, shape, color)  # type: ignore[attr-defined]
                self.meshcat.SetTransform(path, X_WE)  # type: ignore[attr-defined]
            else:
                self.meshcat.Delete(f"{prefix}/{name}/force")  # type: ignore[attr-defined]
