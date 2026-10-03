"""Pinocchio visualization mixin.

Extracts viewer updates, ellipsoid drawing, vector drawing, frame/COM
overlays, and toggle handlers from PinocchioGUI (gui.py).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pinocchio as pin
from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger

# Check meshcat availability
try:
    import meshcat.geometry as g

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    g = None  # type: ignore[assignment]

try:
    import pinocchio.visualize as viz
except (ImportError, AttributeError):
    viz = None  # type: ignore[assignment]

logger = get_logger(__name__)

# Constants
COM_SPHERE_RADIUS = 0.02
COM_COLOR = 0xFFFF00

__all__ = [
    "COM_COLOR",
    "COM_SPHERE_RADIUS",
    "MESHCAT_AVAILABLE",
    "PinocchioVisualizationMixin",
]


class PinocchioVisualizationMixin:
    """Mixin for Pinocchio GUI visualization, ellipsoids, vectors, overlays.

    Provides:
    - ``_init_meshcat_viewer``: Meshcat visualizer initialization
    - ``_log_meshcat_url``: URL logging to UI and console
    - ``_update_viewer``: Full viewer refresh with overlays
    - ``_compute_analysis``: Jacobian/mass matrix analysis
    - ``_draw_ellipsoids``: Mobility/force ellipsoid rendering
    - ``_draw_vectors``: Force/torque vector visualization
    - ``_draw_induced_vectors``: Induced acceleration vectors
    - ``_draw_cf_vectors``: Counterfactual vectors
    - ``_draw_frames`` / ``_draw_coms``: Frame/COM overlays
    - Toggle handlers for frames, COMs, forces, torques
    """

    def _init_meshcat_viewer(self: Any) -> None:
        """Initialize the Meshcat viewer."""
        self.viewer: Any | None = None
        if not MESHCAT_AVAILABLE or viz is None:
            if hasattr(self, "log_write"):
                self.log_write(
                    "Warning: Meshcat not available. Visualization disabled."
                )
            logger.warning("Meshcat module not found.")
            return

        try:
            try:
                self.viewer = viz.Visualizer(server_args=["--port", "7000"])
            except TypeError:
                logger.warning(
                    "Meshcat Visualizer: server_args not supported. Using default."
                )
                self.viewer = viz.Visualizer()

            url = self.viewer.url() if callable(self.viewer.url) else self.viewer.url
            logger.info("Internal Meshcat URL: %s", url)

            self._log_meshcat_url(url)
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error("Failed to initialize Meshcat viewer: %s", exc)
            if hasattr(self, "log_write"):
                self.log_write(f"Error: Failed to initialize Meshcat viewer: {exc}")
                self.log_write("Please ensure meshcat-server is running or try again.")

    def _log_meshcat_url(self: Any, url: str) -> None:
        """Log viewer URL to UI and console."""
        try:
            port = url.split(":")[-1].split("/")[0]
            host_url = f"http://127.0.0.1:{port}/static/"
            logger.info("Host Access URL: %s", host_url)
            if hasattr(self, "log_write"):
                self.log_write("=" * 40)
                self.log_write("VISUALIZER READY")
                self.log_write("Open this URL in your browser:")
                self.log_write(f"{host_url}")
                self.log_write("=" * 40)
        except (PermissionError, OSError, IndexError):
            logger.info("Could not determine host URL from: %s", url)

    def _update_viewer(self: Any) -> None:
        if (
            self.model is None
            or self.data is None
            or self.q is None
            or self.viz is None
        ):  # noqa: E501
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
        if (
            self.chk_forces.isChecked()
            or self.chk_torques.isChecked()
            or self.chk_cf.isChecked()
            or self._get_force_overlay_toggles()["shading"]
        ):
            self._draw_vectors()
        elif (
            hasattr(self, "force_overlay_view") and self.force_overlay_view is not None
        ):
            self.force_overlay_view.update(
                {"forces": False, "torques": False, "shading": False, "ztcf": False}
            )

        if self.chk_induced.isChecked():
            self._draw_induced_vectors()

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
        )  # noqa: E501

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
        with contextlib.suppress(RuntimeError, ValueError, AttributeError):
            self.viewer["overlays/ellipsoids"].delete()

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
                    ):  # noqa: E501
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
                    ):  # noqa: E501
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
        if name is None:
            raise ValueError("name must be provided")
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

    def _ensure_force_overlay_view(self: Any) -> Any:
        if not hasattr(self, "force_overlay_view") or self.force_overlay_view is None:
            from .force_overlay_view import PinocchioForceOverlayView

            color_session = getattr(self, "segment_force_colors", None)
            viz = getattr(self, "viz", None) or getattr(self, "viewer", None)
            self.force_overlay_view = PinocchioForceOverlayView(
                self, viz, color_session
            )
        return self.force_overlay_view

    def _get_force_overlay_toggles(self: Any) -> dict[str, Any]:
        color_session = getattr(self, "segment_force_colors", None)
        scale_enabled = (
            getattr(color_session, "_scale", None) is not None
            and color_session._scale.enabled
        )
        forces_enabled = (
            bool(self.chk_forces.isChecked()) if hasattr(self, "chk_forces") else True
        )
        torques_enabled = (
            bool(self.chk_torques.isChecked()) if hasattr(self, "chk_torques") else True
        )
        ztcf_enabled = (
            bool(self.chk_cf.isChecked()) if hasattr(self, "chk_cf") else False
        )
        shading_enabled = scale_enabled or (
            hasattr(self, "chk_shading") and bool(self.chk_shading.isChecked())
        )
        force_scale = (
            float(self.spin_force_scale.value())
            if hasattr(self, "spin_force_scale")
            else 0.001
        )
        torque_scale = (
            float(self.spin_torque_scale.value())
            if hasattr(self, "spin_torque_scale")
            else 0.005
        )
        return {
            "forces": forces_enabled,
            "torques": torques_enabled,
            "shading": shading_enabled,
            "ztcf": ztcf_enabled,
            "force_scale": force_scale,
            "torque_scale": torque_scale,
        }

    def _draw_vectors(self: Any) -> None:
        """Draw force and torque vectors via PinocchioForceOverlayView."""
        view = self._ensure_force_overlay_view()
        if view is not None:
            view.update(self._get_force_overlay_toggles())

    def _draw_induced_vectors(self: Any) -> None:  # noqa: C901
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

            if g is not None and self.viewer is not None:
                points = np.vstack(
                    [oMi.translation, oMi.translation + vec * scale]
                ).T.astype(np.float32)
                self.viewer[f"overlays/induced/{self.model.names[i]}"].set_object(
                    g.Line(
                        g.PointsGeometry(points), g.LineBasicMaterial(color=0xFF00FF)
                    )
                )

    def _draw_cf_vectors(self: Any) -> None:
        """Draw Counterfactual vectors via PinocchioForceOverlayView."""
        view = self._ensure_force_overlay_view()
        if view is not None:
            view.update(self._get_force_overlay_toggles())

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

            com_transform = np.eye(4)
            com_transform[:3, 3] = com_world
            self.viewer[f"overlays/coms/{self.model.names[i]}"].set_transform(
                com_transform
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
        if checked is None:
            raise ValueError("checked must be provided")
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/forces"].delete()
        self._update_viewer()

    def _toggle_torques(self: Any, checked: bool) -> None:  # noqa: FBT001
        if checked is None:
            raise ValueError("checked must be provided")
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
