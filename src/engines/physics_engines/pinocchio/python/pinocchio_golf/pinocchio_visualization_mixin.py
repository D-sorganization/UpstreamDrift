"""Pinocchio visualization mixin.

Extracts viewer updates, ellipsoid drawing, vector drawing, frame/COM
overlays, and toggle handlers from PinocchioGUI (gui.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore
from PyQt6 import QtWidgets

from src.shared.python.logging_config import get_logger

# Check meshcat availability
try:
    import meshcat.geometry as g

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    g = None  # type: ignore

logger = get_logger(__name__)

# Constants
COM_SPHERE_RADIUS = 0.02
COM_COLOR = 0xFFFF00


class PinocchioVisualizationMixin:
    """Mixin for Pinocchio GUI visualization, ellipsoids, vectors, overlays.

    Provides:
    - ``_update_viewer``: Full viewer refresh with overlays
    - ``_compute_analysis``: Jacobian/mass matrix analysis
    - ``_draw_ellipsoids``: Mobility/force ellipsoid rendering
    - ``_draw_vectors``: Force/torque vector visualization
    - ``_draw_induced_vectors``: Induced acceleration vectors
    - ``_draw_cf_vectors``: Counterfactual vectors
    - ``_draw_frames`` / ``_draw_coms``: Frame/COM overlays
    - Toggle handlers for frames, COMs, forces, torques
    """

    def _update_viewer(self: Any) -> None:
        if (
            self.model is None
            or self.data is None
            or self.q is None
            or self.viz is None
        ):
            return

        # Update Visuals via Pinocchio Visualizer
        self.viz.display(self.q)

        # Kinematics Logic for frames (needed for custom overlays)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        # Calculate matrices for analysis
        self._compute_analysis()

        # Overlays
        if self.chk_frames.isChecked():
            self._draw_frames()
        if self.chk_coms.isChecked():
            self._draw_coms()
        if self.chk_forces.isChecked() or self.chk_torques.isChecked():
            self._draw_vectors()

        if self.chk_induced.isChecked():
            self._draw_induced_vectors()
        if self.chk_cf.isChecked():
            self._draw_cf_vectors()

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
            self._draw_ellipsoids()
        else:
            if self.viewer:
                self.viewer["overlays/ellipsoids"].delete()

    def _compute_analysis(self: Any) -> None:
        """Compute Jacobian and Mass matrix analysis."""
        if self.model is None or self.data is None or self.q is None:
            return

        joint_id = self.model.njoints - 1
        pin.computeJointJacobians(self.model, self.data, self.q)
        J = pin.getJointJacobian(
            self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL
        )

        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = s[0] / s[-1] if s[-1] > 1e-9 else float("inf")
            self.lbl_cond.setText(f"{cond:.2f}")
        except (ValueError, TypeError, RuntimeError):
            self.lbl_cond.setText("Error")

        M = pin.crba(self.model, self.data, self.q)
        try:
            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.model.nv}")
        except (ValueError, TypeError, RuntimeError):
            self.lbl_rank.setText("Error")

    def _draw_ellipsoids(self: Any) -> None:
        """Draw mobility/force ellipsoids for selected bodies."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.manip_analyzer is None
        ):
            return

        # Clear previous ellipsoids to prevent ghosting
        try:
            self.viewer["overlays/ellipsoids"].delete()
        except (RuntimeError, ValueError, AttributeError):
            pass

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
            # Get selected bodies
            selected_bodies = [
                name for name, chk in self.manip_checkboxes.items() if chk.isChecked()
            ]

            if selected_bodies:
                for body_name in selected_bodies:
                    res = self.manip_analyzer.compute_metrics(body_name, self.q)
                    if not res:
                        continue

                    pos = res.velocity_ellipsoid.center

                    if (
                        self.chk_mobility.isChecked()
                        and res.mobility_matrix is not None
                    ):
                        path_name = f"{res.body_name}/mobility"
                        radii = res.velocity_ellipsoid.radii
                        self._draw_ellipsoid_meshcat(
                            path_name,
                            pos,
                            res.velocity_ellipsoid.axes,
                            radii * 0.5,
                            0x00FF00,
                        )

                    if (
                        self.chk_force_ellip.isChecked()
                        and res.force_matrix is not None
                    ):
                        path_name = f"{res.body_name}/force"
                        radii = res.force_ellipsoid.radii
                        self._draw_ellipsoid_meshcat(
                            path_name,
                            pos,
                            res.force_ellipsoid.axes,
                            radii * 0.2,
                            0xFF0000,
                        )

    def _draw_ellipsoid_meshcat(
        self: Any,
        name: str,
        pos: np.ndarray,
        rot: np.ndarray,
        radii: np.ndarray,
        color: int,
    ) -> None:
        """Draw ellipsoid using Meshcat."""
        if self.viewer is None:
            return

        path = f"overlays/ellipsoids/{name}"

        self.viewer[path].set_object(
            g.Sphere(1.0),
            g.MeshLambertMaterial(color=color, opacity=0.5, transparent=True),
        )

        T = np.eye(4)
        T[:3, :3] = rot @ np.diag(radii)
        T[:3, 3] = pos

        self.viewer[path].set_transform(T)

    def _draw_vectors(self: Any) -> None:
        """Draw force and torque vectors at joints."""
        if self.model is None or self.data is None or self.viewer is None:
            return

        v = self.v if self.v is not None else np.zeros(self.model.nv)
        a = pin.aba(self.model, self.data, self.q, v, np.zeros(self.model.nv))

        pin.rnea(self.model, self.data, self.q, v, a)

        force_scale = self.spin_force_scale.value()
        torque_scale = self.spin_torque_scale.value()

        for i in range(1, self.model.njoints):
            joint_placement = self.data.oMi[i]
            f_local = self.data.f[i]

            f_world = joint_placement.rotation @ f_local.linear
            t_world = joint_placement.rotation @ f_local.angular

            joint_name = self.model.names[i]

            if self.chk_forces.isChecked() and np.linalg.norm(f_world) > 1e-3:
                self._draw_arrow(
                    f"overlays/forces/{joint_name}",
                    joint_placement.translation,
                    f_world * force_scale,
                    0xFF0000,
                )

            if self.chk_torques.isChecked() and np.linalg.norm(t_world) > 1e-3:
                self._draw_arrow(
                    f"overlays/torques/{joint_name}",
                    joint_placement.translation,
                    t_world * torque_scale,
                    0x0000FF,
                )

    def _draw_induced_vectors(self: Any) -> None:
        """Draw induced acceleration vectors."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.latest_induced is None
        ):
            return

        source = self.combo_induced.currentText()
        accels = np.zeros(self.model.nv)

        if source in ["gravity", "velocity", "total"]:
            if source in self.latest_induced:
                accels = self.latest_induced[source]
        else:
            if source in self.latest_induced:
                accels = self.latest_induced[source]
            else:
                txt = source
                if txt and self.analyzer and self.q is not None:
                    try:
                        parts = [float(x) for x in txt.split(",")]
                        tau = np.zeros(self.model.nv)
                        min_len = min(len(parts), len(tau))
                        tau[:min_len] = parts[:min_len]
                        accels = self.analyzer.compute_specific_control(self.q, tau)
                    except ValueError:
                        pass

        scale = self.spin_torque_scale.value()

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            idx_v = joint.idx_v
            nv = joint.nv
            if nv != 1:
                continue

            alpha = accels[idx_v]
            if abs(alpha) < 1e-3:
                continue

            oMi = self.data.oMi[i]
            S = joint.S
            a_local = S * alpha
            a_world = oMi.act(a_local)

            vec = a_world.angular
            if np.linalg.norm(vec) < 1e-6:
                vec = a_world.linear

            self._draw_arrow(
                f"overlays/induced/{self.model.names[i]}",
                oMi.translation,
                vec * scale,
                0xFF00FF,  # Magenta
            )

    def _draw_cf_vectors(self: Any) -> None:
        """Draw Counterfactual vectors."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.latest_cf is None
        ):
            return

        cf_type = self.combo_cf.currentText()
        if cf_type not in self.latest_cf:
            return

        vals = self.latest_cf[cf_type]
        scale = self.spin_torque_scale.value()

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            idx_v = joint.idx_v
            nv = joint.nv
            if nv != 1:
                continue

            val = vals[idx_v]
            if abs(val) < 1e-3:
                continue

            oMi = self.data.oMi[i]
            S = joint.S
            spatial_vec = oMi.act(S * val)

            vec = spatial_vec.angular
            if np.linalg.norm(vec) < 1e-6:
                vec = spatial_vec.linear

            self._draw_arrow(
                f"overlays/cf/{self.model.names[i]}",
                oMi.translation,
                vec * scale,
                0xFFFF00,  # Yellow
            )

    def _draw_arrow(
        self: Any, path: str, start: np.ndarray, vector: np.ndarray, color: int
    ) -> None:
        """Helper to draw an arrow in Meshcat."""
        if self.viewer is None:
            return

        points = np.vstack([start, start + vector]).T.astype(np.float32)
        self.viewer[path].set_object(
            g.Line(g.PointsGeometry(points), g.LineBasicMaterial(color=color))
        )

    def _draw_frames(self: Any) -> None:
        if self.model is None or self.data is None or self.viewer is None:
            return

        for i, frame in enumerate(self.model.frames):
            if frame.name == "universe":
                continue

            transform = self.data.oMf[i]
            homogeneous_matrix = transform.homogeneous
            self.viewer[f"overlays/frames/{frame.name}"].set_transform(
                homogeneous_matrix
            )

    def _draw_coms(self: Any) -> None:
        if self.model is None or self.data is None or self.viewer is None:
            return

        for i in range(1, self.model.njoints):
            inertia = self.model.inertias[i]
            joint_transform = self.data.oMi[i]
            com_world = joint_transform.act(inertia.lever)

            self.viewer[f"overlays/coms/{self.model.names[i]}"].set_transform(
                pin.SE3(np.eye(3), com_world).homogeneous
            )

    # --- Vis Helpers ---
    def _toggle_frames(self: Any, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return

        if not checked:
            self.viewer["overlays/frames"].delete()
        else:
            if self.model:
                for frame in self.model.frames:
                    if frame.name == "universe":
                        continue
                    self.viewer[f"overlays/frames/{frame.name}"].set_object(
                        g.triad(scale=0.1)
                    )
            self._update_viewer()

    def _toggle_coms(self: Any, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return

        if not checked:
            self.viewer["overlays/coms"].delete()
        else:
            if self.model:
                for i in range(1, self.model.njoints):
                    self.viewer[f"overlays/coms/{self.model.names[i]}"].set_object(
                        g.Sphere(COM_SPHERE_RADIUS),
                        g.MeshLambertMaterial(color=COM_COLOR),
                    )
            self._update_viewer()

    def _toggle_forces(self: Any, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/forces"].delete()
        self._update_viewer()

    def _toggle_torques(self: Any, checked: bool) -> None:  # noqa: FBT001
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/torques"].delete()
        self._update_viewer()

    def _populate_manipulability_checkboxes(self: Any) -> None:
        """Populate checkboxes for manipulability analysis body selection."""
        if self.manip_analyzer is None:
            return

        # Clear existing checkboxes
        while self.manip_body_layout.count():
            item = self.manip_body_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.manip_checkboxes.clear()

        # Get potential bodies from analyzer
        bodies = self.manip_analyzer.find_potential_bodies()

        cols = 3
        for i, name in enumerate(bodies):
            chk = QtWidgets.QCheckBox(name)
            chk.toggled.connect(self._update_viewer)
            self.manip_checkboxes[name] = chk
            self.manip_body_layout.addWidget(chk, i // cols, i % cols)

            # Default check relevant parts
            if any(x in name.lower() for x in ["club", "hand", "wrist"]):
                chk.setChecked(True)
