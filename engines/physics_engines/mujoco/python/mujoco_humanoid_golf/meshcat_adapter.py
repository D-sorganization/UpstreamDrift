import logging
import os
import webbrowser
from typing import Any

import mujoco
import numpy as np

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
except ImportError:
    meshcat = None

logger = logging.getLogger(__name__)


class MuJoCoMeshcatAdapter:
    """
    Adapts MuJoCo model/data to Meshcat for web-based visualization.
    """

    def __init__(self, model: mujoco.MjModel | None = None):
        if meshcat is None:
            logger.warning("Meshcat not installed. Visualization disabled.")
            self.vis = None
            return

        # Initialize Visualizer
        # Note: We let meshcat find a free port if None, or use specific if provided.
        # But standard Visualizer doesn't take scalar port easily
        # without custom serverargs.
        # We'll rely on default behavior or pass zmq_url if needed?
        # For now, standard init.
        self.vis = meshcat.Visualizer()

        self.model = model
        self.is_open = True

        # Log URL
        self.url = self.vis.url()
        logger.info(f"Meshcat initialized at {self.url}")

        # Determine host-accessible URL if in Docker
        if os.environ.get("MESHCAT_HOST") == "0.0.0.0":
            try:
                port = self.url.split(":")[-1].split("/")[0]
                host_url = f"http://127.0.0.1:{port}/static/"
                logger.info(f"Host Meshcat URL: {host_url}")
            except (IndexError, ValueError):
                pass

        self.load_model_geometry()

    def open_browser(self) -> None:
        if self.vis is not None:
            webbrowser.open(self.vis.url())

    def load_model_geometry(self) -> None:
        """
        Parses MuJoCo model geoms and creates corresponding Meshcat objects.
        """
        if self.vis is None or self.model is None:
            return

        model = self.model
        self.vis["visuals"].delete()

        # Iterate over all geometries
        for i in range(model.ngeom):
            # geom properties
            gtype = model.geom_type[i]
            size = model.geom_size[i]
            rgba = model.geom_rgba[i]

            # Material/Color
            material = g.MeshPhongMaterial(
                color=self._rgba_to_hex(rgba), opacity=rgba[3]
            )

            shape = None

            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                shape = g.Sphere(radius=size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # Meshcat has no capsule, approximate with Cylinder + 2 Spheres?
                # Or just Cylinder for now to keep it fast.
                # size[0] = radius, size[1] = half-length
                shape = g.Cylinder(height=size[1] * 2, radius=size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                shape = g.Cylinder(height=size[1] * 2, radius=size[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                # size is half-extents
                shape = g.Box(lengths=size * 2)
            elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                # Infinite plane - approximate with large box/plane
                shape = g.Box([20, 20, 0.01])
            elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                # Loading meshes is complex (need to get vertices from model.mesh_*)
                # For now, approximate with Box or Sphere based on rbound
                shape = g.Sphere(radius=model.geom_rbound[i])
            else:
                # Fallback
                shape = g.Sphere(radius=0.1)

            if shape:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if not name:
                    name = f"geom_{i}"

                # If capsule/cylinder, MuJoCo defines them along Z axis.
                # If using Cylinder, we might need rotation correction depending on
                # meshcat convention (usually Y-axis aligned?)
                # Meshcat Cylinder is along Y axis by default? No, usually Y.
                # Let's check: Three.js cylinder is geometry aligned with the Y axis.
                # MuJoCo cylinder is Z axis.
                # So we need to rotate 90 deg around X.
                if gtype in [
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                ]:
                    # We will wrap it in a node that rotates
                    self.vis["visuals"][name]["geometry"].set_object(shape, material)
                    # Rotate geometry to align Y (meshcat) with Z (mujoco)
                    # Rotate -90 deg around X?
                    # Actually we update the PARENT transform in update(),
                    # which sets the Z-axis orientation.
                    # But the SHAPE itself needs to be pre-rotated if the local
                    # frame mismatch exists.
                    rotation_matrix = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
                    self.vis["visuals"][name]["geometry"].set_transform(rotation_matrix)
                else:
                    self.vis["visuals"][name].set_object(shape, material)

    def update(self, data: mujoco.MjData) -> None:
        """
        Updates geometry transforms from MuJoCo data.
        """
        if self.vis is None or data is None or self.model is None:
            return

        model = self.model

        # Update Geoms
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if not name:
                name = f"geom_{i}"

            pos = data.geom_xpos[i]
            mat = data.geom_xmat[i].reshape(3, 3)

            # Construct 4x4 matrix
            T = np.eye(4)
            T[:3, :3] = mat
            T[:3, 3] = pos

            self.vis["visuals"][name].set_transform(T)

    def draw_vectors(
        self,
        data: mujoco.MjData,
        show_force: bool,
        show_torque: bool,
        force_scale: float = 0.1,
        torque_scale: float = 0.1,
    ) -> None:
        """
        Draws force/torque vectors at joints.
        """
        if self.vis is None or self.model is None:
            return

        model = self.model

        # Clear previous vectors if not showing?
        # Meshcat efficiently updates if we overwrite the path.
        # But if we turn them off, we explicitly delete.
        if not show_force:
            self.vis["overlays/forces"].delete()
        if not show_torque:
            self.vis["overlays/torques"].delete()

        if not (show_force or show_torque):
            return

        # Iterate over joints (similar to Pinocchio logic)
        # MuJoCo data.qfrc_constraint / qfrc_actuator?
        # No, for reaction forces, we likely want `cfrc_ext` (com-based) or `cfrc_int`.
        # Or simply visualize contact forces?
        # User asked for "force and torque vectors... at joints"
        # (implied by Pinocchio parity).
        # Pinocchio visualized Joint Reaction Forces (`data.f` from RNEA).
        # In MuJoCo, joint reaction forces are not computed by default unless
        # inverse dynamics or `mj_rne` is used.
        # However, `data.cfrc_int` contains interaction forces at bodies.

        # Iterate over bodies (skipping world 0)
        for i in range(1, model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not body_name:
                body_name = f"body_{i}"

            # cfrc_int is (6,) vector: [torque(3), force(3)] at body frame/COM?
            # documentation: "interaction force/torque exerted by parent on this body"
            if data.cfrc_int is None:
                continue
            wrench = data.cfrc_int[i]  # type: ignore[index]
            f = wrench[3:]
            t = wrench[:3]

            pos = data.xpos[i]

            if show_force and np.linalg.norm(f) > 1e-3:
                self._draw_arrow(
                    f"overlays/forces/{body_name}", pos, f * force_scale, 0xFF0000
                )

            if show_torque and np.linalg.norm(t) > 1e-3:
                self._draw_arrow(
                    f"overlays/torques/{body_name}", pos, t * torque_scale, 0x0000FF
                )

    def draw_ellipsoid(
        self,
        name: str,
        position: np.ndarray,
        rotation: np.ndarray,
        radii: np.ndarray,
        color: int = 0x00FF00,
        opacity: float = 0.3,
    ) -> None:
        """
        Draws an ellipsoid at the specified position/orientation.

        Args:
            name: Unique name for the ellipsoid
            position: 3D position vector
            rotation: 3x3 rotation matrix (axes of ellipsoid)
            radii: Length of principal axes (x, y, z)
            color: Hex color
            opacity: Opacity (0-1)
        """
        if self.vis is None:
            return

        # Path for this ellipsoid
        path = f"overlays/ellipsoids/{name}"

        # Create unit sphere and scale it to radii
        # Meshcat doesn't have explicit Ellipsoid, so we use Scaled Sphere
        material = g.MeshPhongMaterial(color=color, opacity=opacity, transparent=True)
        shape = g.Sphere(radius=1.0)

        # Transformation: T = [Rot | Pos] * Scale
        # But Meshcat applies scale via object property or transform?
        # Usually transform matrix can include scale.

        T = np.eye(4)
        T[:3, :3] = rotation @ np.diag(radii)
        T[:3, 3] = position

        self.vis[path].set_object(shape, material)
        self.vis[path].set_transform(T)

    def clear_ellipsoids(self) -> None:
        """Clears all drawn ellipsoids."""
        if self.vis:
            self.vis["overlays/ellipsoids"].delete()

    def _draw_arrow(
        self, path: str, start: np.ndarray, vec: np.ndarray, color_hex: int
    ) -> None:
        if self.vis is None:
            return

        # Create a Line segment
        end = start + vec
        vertices = np.array([start, end]).T  # 3x2

        self.vis[path].set_object(
            g.Line(
                g.PointsGeometry(vertices),
                g.LineBasicMaterial(color=color_hex, linewidth=5),
            )
        )

    def _rgba_to_hex(self, rgba: Any) -> int:
        if rgba is None:
            return 0
        r, g, b = (int(c * 255) for c in rgba[:3])
        return (r << 16) + (g << 8) + b
