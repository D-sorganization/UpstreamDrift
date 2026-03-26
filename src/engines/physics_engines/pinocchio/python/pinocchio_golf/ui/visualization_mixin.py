"""Visualization and overlay mixin for Pinocchio GUI."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pinocchio as pin

if TYPE_CHECKING:
    from ..gui import PinocchioGUI

logger = logging.getLogger(__name__)

# Constants (mirrored from main gui for availability)
COM_SPHERE_RADIUS = 0.02
COM_COLOR = 0xFFFF00


class VisualizationMixin:
    """Mixin containing visualization, viewer, and overlay logic."""

    def _init_meshcat_viewer(self: PinocchioGUI) -> None:
        """Initialize the viewer."""
        from . import MESHCAT_AVAILABLE, viz

        self.viewer: Any | None = None
        if not MESHCAT_AVAILABLE:
            self.log_write("Warning: Meshcat not available. Visualization disabled.")
            logger.warning("Meshcat module not found.")
            return

        try:
            try:
                self.viewer = viz.Visualizer(server_args=["--port", "7000"])
            except TypeError:
                logger.warning(
                    "Meshcat Visualizer: server_args not supported. Using default."
                )  # noqa: E501
                self.viewer = viz.Visualizer()

            url = self.viewer.url() if callable(self.viewer.url) else self.viewer.url
            logger.info("Internal Meshcat URL: %s", url)

            self._log_meshcat_url(url)
        except (ConnectionError, OSError, RuntimeError) as exc:
            logger.error(f"Failed to initialize Meshcat viewer: {exc}")
            self.log_write(f"Error: Failed to initialize Meshcat viewer: {exc}")
            self.log_write("Please ensure meshcat-server is running or try again.")

    def _log_meshcat_url(self: PinocchioGUI, url: str) -> None:
        """Log viewer URL to UI and console."""
        try:
            port = url.split(":")[-1].split("/")[0]
            host_url = f"http://127.0.0.1:{port}/static/"
            logger.info(f"Host Access URL: {host_url}")
            self.log_write("=" * 40)
            self.log_write("VISUALIZER READY")
            self.log_write("Open this URL in your browser:")
            self.log_write(f"{host_url}")
            self.log_write("=" * 40)
        except (PermissionError, OSError):
            logger.info("Could not determine host URL from: %s", url)

    def _update_viewer(self: PinocchioGUI) -> None:
        """Update the 3D viewer state and overlays."""
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
        if self.chk_forces.isChecked() or self.chk_torques.isChecked():
            self._draw_vectors()

        if hasattr(self, "chk_induced") and self.chk_induced.isChecked():
            self._draw_induced_vectors()
        if hasattr(self, "chk_cf") and self.chk_cf.isChecked():
            self._draw_cf_vectors()

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
            self._draw_ellipsoids()
        else:
            if self.viewer:
                self.viewer["overlays/ellipsoids"].delete()

    def _compute_analysis(self: PinocchioGUI) -> None:
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
        except (ValueError, TypeError, RuntimeError, IndexError):
            self.lbl_cond.setText("Error")

        M = pin.crba(self.model, self.data, self.q)
        try:
            rank = np.linalg.matrix_rank(M)
            self.lbl_rank.setText(f"{rank} / {self.model.nv}")
        except (ValueError, TypeError, RuntimeError):
            self.lbl_rank.setText("Error")

    def _draw_ellipsoids(self: PinocchioGUI) -> None:
        """Draw mobility/force ellipsoids for selected bodies."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or self.manip_analyzer is None
        ):
            return

        with contextlib.suppress(RuntimeError, ValueError, AttributeError):
            self.viewer["overlays/ellipsoids"].delete()

        if self.chk_mobility.isChecked() or self.chk_force_ellip.isChecked():
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
                            radii * 0.05,
                            0xFF0000,
                        )

    def _draw_ellipsoid_meshcat(
        self: PinocchioGUI,
        path: str,
        pos: np.ndarray,
        axes: np.ndarray,
        radii: np.ndarray,
        color: int,
    ) -> None:
        """Internal helper to render an ellipsoid in Meshcat."""
        if not (path is not None):
            raise ValueError("path must be provided")
        if not (path is not None):
            raise ValueError("path must be provided")
        import meshcat.geometry as g

        if self.viewer is None:
            return

        full_path = f"overlays/ellipsoids/{path}"
        self.viewer[full_path].set_object(
            g.Ellipsoid(radii),
            g.MeshPhongMaterial(color=color, opacity=0.4, transparent=True),
        )

        T = np.eye(4)
        T[:3, :3] = axes
        T[:3, 3] = pos
        self.viewer[full_path].set_transform(T)

    def _draw_vectors(self: PinocchioGUI) -> None:
        """Draw force and torque vectors."""
        if self.model is None or self.data is None or self.viewer is None:
            return

        self.viewer["overlays/vectors"].delete()

        if not self.chk_forces.isChecked() and not self.chk_torques.isChecked():
            return

        f_scale = self.spn_f_scale.value()
        t_scale = self.spn_t_scale.value()

        for i in range(1, self.model.njoints):
            pos = self.data.oMi[i].translation

            if self.chk_forces.isChecked():
                f = self.data.f[i].linear
                if np.linalg.norm(f) > 1e-4:
                    self._draw_arrow(f"joint_{i}_force", pos, f * f_scale, 0xFF0000)

            if self.chk_torques.isChecked():
                t = self.data.f[i].angular
                if np.linalg.norm(t) > 1e-4:
                    self._draw_arrow(f"joint_{i}_torque", pos, t * t_scale, 0x0000FF)

    def _draw_induced_vectors(self: PinocchioGUI) -> None:
        """Render induced acceleration components as vectors in the viewer."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or not self.chk_induced.isChecked()
        ):
            if self.viewer:
                self.viewer["overlays/induced"].delete()
            return

        self.viewer["overlays/induced"].delete()

        # Determine frame to visualize at - use end effector (last joint)
        joint_idx = self.model.njoints - 1
        pos = self.data.oMi[joint_idx].translation

        # Get latest induced data
        induced = getattr(self, "latest_induced", None) or {}

        scale = self.spn_f_scale.value() * 0.1

        colors = {
            "gravity": 0xFF0000,
            "velocity": 0x00FF00,
            "total": 0xFFFFFF,
            "specific_control": 0x0000FF,
        }

        for name, a in induced.items():
            if name in colors and np.linalg.norm(a) > 1e-6:
                # We need to map joint acceleration (NV) to spatial acceleration (6D)
                # or just visualize a representative component.
                # For the GUI, display first 3 spatial components.
                self._draw_arrow(f"induced/{name}", pos, a[:3] * scale, colors[name])

    def _draw_cf_vectors(self: PinocchioGUI) -> None:
        """Render Counterfactual vectors."""
        if (
            self.model is None
            or self.data is None
            or self.viewer is None
            or not self.chk_cf.isChecked()
        ):
            if self.viewer:
                self.viewer["overlays/cf"].delete()
            return

        self.viewer["overlays/cf"].delete()

        joint_idx = self.model.njoints - 1
        pos = self.data.oMi[joint_idx].translation

        cf = getattr(self, "latest_cf", None) or {}
        scale = self.spn_t_scale.value() * 0.1

        if "ztcf" in cf:
            self._draw_arrow("cf/ztcf", pos, cf["ztcf"][:3] * scale, 0xFFFF00)
        if "zvcf" in cf:
            self._draw_arrow("cf/zvcf", pos, cf["zvcf"][:3] * scale, 0x00FFFF)

    def _draw_arrow(
        self: PinocchioGUI, path: str, start: np.ndarray, vector: np.ndarray, color: int
    ) -> None:
        """Helper to draw an arrow in Meshcat."""
        if not (path is not None):
            raise ValueError("path must be provided")
        if not (path is not None):
            raise ValueError("path must be provided")
        import meshcat.geometry as g

        if self.viewer is None:
            return

        points = np.vstack([start, start + vector]).T.astype(np.float32)
        self.viewer[path].set_object(
            g.Line(g.PointsGeometry(points), g.LineBasicMaterial(color=color))
        )

    def _draw_frames(self: PinocchioGUI) -> None:
        """Render coordinate frames."""
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

    def _draw_coms(self: PinocchioGUI) -> None:
        """Render individual joint COMs."""
        if self.model is None or self.data is None or self.viewer is None:
            return

        for i in range(1, self.model.njoints):
            inertia = self.model.inertias[i]
            joint_transform = self.data.oMi[i]
            com_world = joint_transform.act(inertia.lever)

            self.viewer[f"overlays/coms/{self.model.names[i]}"].set_transform(
                pin.SE3(np.eye(3), com_world).homogeneous
            )

    def _toggle_frames(self: PinocchioGUI, checked: bool) -> None:
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/frames"].delete()
        else:
            import meshcat.geometry as g

            if self.model:
                for frame in self.model.frames:
                    if frame.name == "universe":
                        continue
                    self.viewer[f"overlays/frames/{frame.name}"].set_object(
                        g.triad(scale=0.1)
                    )
            self._update_viewer()

    def _toggle_coms(self: PinocchioGUI, checked: bool) -> None:
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/coms"].delete()
        else:
            import meshcat.geometry as g

            if self.model:
                for i in range(1, self.model.njoints):
                    self.viewer[f"overlays/coms/{self.model.names[i]}"].set_object(
                        g.Sphere(COM_SPHERE_RADIUS),
                        g.MeshLambertMaterial(color=COM_COLOR),
                    )
            self._update_viewer()

    def _toggle_forces(self: PinocchioGUI, checked: bool) -> None:
        if not (checked is not None):
            raise ValueError("checked must be provided")
        if not (checked is not None):
            raise ValueError("checked must be provided")
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/forces"].delete()
        self._update_viewer()

    def _toggle_torques(self: PinocchioGUI, checked: bool) -> None:
        if not (checked is not None):
            raise ValueError("checked must be provided")
        if not (checked is not None):
            raise ValueError("checked must be provided")
        if self.viewer is None:
            return
        if not checked:
            self.viewer["overlays/torques"].delete()
        self._update_viewer()
