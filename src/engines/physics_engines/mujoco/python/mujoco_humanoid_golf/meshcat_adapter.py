"""MeshCat adapter for MuJoCo humanoid golf visualization.

Provides a browser-based 3D visualization backend via MeshCat,
wrapping MuJoCo model data for interactive display.
"""

import os
import webbrowser
from typing import Any

import mujoco
import numpy as np

from src.shared.python.biomechanics.biomechanics_data import BiomechanicalData
from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale
from src.shared.python.body_part_viz.meshcat_force_colors import MeshcatForceColors
from src.shared.python.body_part_viz.meshcat_scalar_colors import MeshcatScalarColors
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.force_overlay.glyphs import GlyphSet
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
    MeshcatPythonSink,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.plot_style.color_utils import rgba_to_hex

from .meshcat_geom_loader import build_meshcat_color_bindings

try:
    import meshcat
    import meshcat.geometry as g
except ImportError:
    meshcat = None

logger = get_logger(__name__)


class MuJoCoMeshcatAdapter:
    """
    Adapts MuJoCo model/data to Meshcat for web-based visualization.
    """

    def __init__(self, model: mujoco.MjModel | None = None) -> None:
        self._force_colors: MeshcatForceColors | None = None
        self._net_power_colors: MeshcatScalarColors | None = None
        self._glyph_renderer: MeshcatGlyphRenderer | None = None
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
        if os.environ.get("MESHCAT_HOST") == "0.0.0.0":  # nosec B104 - comparing env var value, not binding to an address
            try:
                port = self.url.split(":")[-1].split("/")[0]
                host_url = f"http://127.0.0.1:{port}/static/"
                logger.info(f"Host Meshcat URL: {host_url}")
            except (IndexError, ValueError):
                pass

        self.load_model_geometry()

    def open_browser(self) -> None:
        """Open the Meshcat visualization URL in the default browser."""
        if self.vis is not None:
            webbrowser.open(self.vis.url())

    def load_model_geometry(self) -> None:
        """Parse MuJoCo geoms into MeshCat visuals and colour-binding tables."""
        if self.vis is None or self.model is None:
            return
        self.vis["visuals"].delete()
        color_bindings = build_meshcat_color_bindings(
            self.vis, self.model, rgba_to_int=self._rgba_to_int
        )
        self._force_colors = MeshcatForceColors(
            self._set_color_property, color_bindings
        )
        self._net_power_colors = MeshcatScalarColors(
            self._set_color_property, color_bindings
        )

    def _set_color_property(self, path: str, prop: str, value: list[float]) -> None:
        if self.vis is not None:
            self.vis[path].set_property(prop, value)

    def update_force_colors(
        self, frame: AxialLoadFrame | None, scale: ForceColorScale
    ) -> None:
        """Share the native view's qualified loads and color settings."""
        if self._force_colors is not None:
            self._force_colors.apply(frame, scale)

    def update_net_power_colors(
        self, frame: Any | None, scale: ScalarColorScale
    ) -> None:
        if self._net_power_colors is not None:
            values = None if frame is None else getattr(frame, "values", None)
            self._net_power_colors.apply(values, scale)

    def update(self, data: mujoco.MjData) -> None:
        """
        Updates geometry transforms from MuJoCo data.
        """
        if data is None:
            raise ValueError("data must be provided")
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

    def draw_ke_globes(self, frame: Any | None, peaks: tuple[Any, ...]) -> None:
        """Draw COM-centered segment KE spheres (#15389)."""
        if self.vis is None:
            return
        self.vis["overlays/segment_ke"].delete()
        if frame is None:
            return
        from src.shared.python.segment_kinetic_energy.contracts import (
            PEAK_HIGHLIGHT_FRACTION,
        )
        from src.shared.python.segment_kinetic_energy.render_globes import (
            globe_rgba_for_ke,
            radius_for_ke,
        )

        peak_by_body = {p.body: p.peak_total_j for p in peaks}
        ref_peak = max(peak_by_body.values(), default=0.0)
        if ref_peak <= 0.0:
            ref_peak = max((s.total_j for s in frame.segments), default=0.0)
        for index, seg in enumerate(frame.segments):
            if seg.total_j <= 0.0:
                continue
            peak_j = peak_by_body.get(seg.body, seg.total_j)
            at_peak = seg.total_j >= PEAK_HIGHLIGHT_FRACTION * peak_j and peak_j > 0.0
            radius = radius_for_ke(seg.total_j, reference_peak_j=ref_peak)
            rgba = globe_rgba_for_ke(
                seg.total_j, reference_peak_j=ref_peak, at_peak=at_peak
            )
            path = f"overlays/segment_ke/{index}"
            material = g.MeshPhongMaterial(
                color=rgba_to_hex(rgba),
                opacity=float(rgba[3]),
                transparent=True,
            )
            self.vis[path].set_object(g.Sphere(radius), material)
            transform = np.eye(4)
            transform[:3, 3] = np.asarray(seg.com_m, dtype=float)
            self.vis[path].set_transform(transform)
            if at_peak:
                glow = f"overlays/segment_ke/glow_{index}"
                glow_mat = g.MeshPhongMaterial(
                    color=rgba_to_hex(rgba),
                    opacity=min(0.3, float(rgba[3])),
                    transparent=True,
                )
                self.vis[glow].set_object(g.Sphere(radius * 1.2), glow_mat)
                self.vis[glow].set_transform(transform)

    def draw_glyphs(self, glyphs: GlyphSet | None) -> None:
        """Draw force/torque glyphs through the shared MeshCat renderer.

        ``None`` removes every force overlay node. Labels that disappear from
        one call to the next are deleted by the renderer.
        """
        renderer = self._get_glyph_renderer()
        if renderer is None:
            return
        if glyphs is None:
            renderer.clear()
        else:
            renderer.update(glyphs)

    def _get_glyph_renderer(self) -> MeshcatGlyphRenderer | None:
        renderer = getattr(self, "_glyph_renderer", None)
        if renderer is None and self.vis is not None:
            renderer = MeshcatGlyphRenderer(MeshcatPythonSink(self.vis))
            self._glyph_renderer = renderer
        return renderer

    def draw_induced_vectors(  # noqa: C901
        self,
        data: mujoco.MjData,
        bio_data: BiomechanicalData | None,
        source: str,
        scale: float = 0.1,
    ) -> None:
        """
        Draws induced acceleration vectors.
        """
        if data is None:
            raise ValueError("data must be provided")
        if self.vis is None or self.model is None:
            return

        self.vis["overlays/induced"].delete()

        if bio_data is None:
            return

        # Determine key
        key = source
        # Custom name logic handled in sim_widget, usually passed as
        # 'selected_actuator' if the source string didn't match standard keys.
        # However, bio_data stores it under 'selected_actuator' if calculated
        # that way. We check if the key exists directly, else try
        # 'selected_actuator'
        if (
            key not in ["gravity", "velocity", "total", "actuator"]
            and key not in bio_data.induced_accelerations
        ):
            key = "selected_actuator"

        if key not in bio_data.induced_accelerations:
            return

        accels = bio_data.induced_accelerations[key]

        # Draw vectors at joints (angular acceleration mainly)
        for j in range(self.model.njnt):
            # Only visualize 1-DOF joints (Slide=2, Hinge=3)
            # 0=Free, 1=Ball have multiple DOFs and axes are different
            jtype = self.model.jnt_type[j]
            if jtype not in [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]:
                continue

            body_id = self.model.jnt_bodyid[j]
            qvel_adr = self.model.jnt_dofadr[j]

            if qvel_adr >= len(accels):
                continue

            acc = accels[qvel_adr]
            if abs(acc) < 1e-3:
                continue

            joint_pos = data.xpos[body_id]
            joint_axis = data.xaxis[j]

            arrow_len = acc * scale * 0.5
            arrow_dir = joint_axis * arrow_len

            # Magenta
            self._draw_arrow(
                f"overlays/induced/joint_{j}", joint_pos, arrow_dir, 0xFF00FF
            )  # noqa: E501

    def draw_cf_vectors(
        self,
        data: mujoco.MjData,
        bio_data: BiomechanicalData | None,
        cf_type: str,
        scale: float = 0.1,
    ) -> None:
        """
        Draws Counterfactual vectors.
        """
        if data is None:
            raise ValueError("data must be provided")
        if self.vis is None or self.model is None:
            return

        self.vis["overlays/cf"].delete()

        if bio_data is None or cf_type not in bio_data.counterfactuals:
            return

        values = bio_data.counterfactuals[cf_type]

        for j in range(self.model.njnt):
            # Only visualize 1-DOF joints
            jtype = self.model.jnt_type[j]
            if jtype not in [mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]:
                continue

            qvel_adr = self.model.jnt_dofadr[j]
            if qvel_adr >= len(values):
                continue

            val = values[qvel_adr]
            if abs(val) < 1e-3:
                continue

            body_id = self.model.jnt_bodyid[j]
            joint_pos = data.xpos[body_id]
            joint_axis = data.xaxis[j]

            arrow_len = val * scale * 0.5
            arrow_dir = joint_axis * arrow_len

            # Yellow
            self._draw_arrow(f"overlays/cf/joint_{j}", joint_pos, arrow_dir, 0xFFFF00)

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
        """
        if name is None:
            raise ValueError("name must be provided")
        if self.vis is None:
            return

        path = f"overlays/ellipsoids/{name}"
        material = g.MeshPhongMaterial(color=color, opacity=opacity, transparent=True)
        shape = g.Sphere(radius=1.0)

        T = np.eye(4)
        T[:3, :3] = rotation @ np.diag(radii)
        T[:3, 3] = position

        self.vis[path].set_object(shape, material)
        self.vis[path].set_transform(T)

    def clear_ellipsoids(self) -> None:
        """Clears all drawn ellipsoids."""
        if self.vis:
            self.vis["overlays/ellipsoids"].delete()

    def draw_swing_plane(
        self,
        name: str,
        vertices: np.ndarray,
        color: int = 0x4488FF,
        opacity: float = 0.25,
    ) -> None:
        """Draw a swing plane as a quad mesh.

        Args:
            name: Unique name for this plane object.
            vertices: (4, 3) corner positions in CCW order [m].
            color: Hex color for the plane surface.
            opacity: Transparency (0=invisible, 1=opaque).
        """
        if name is None:
            raise ValueError("name must be provided")
        if self.vis is None:
            return

        path = f"overlays/swing_plane/{name}"
        # Build two-triangle mesh from the 4 corners
        verts = np.array(vertices, dtype=np.float32)  # (4, 3)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

        material = g.MeshPhongMaterial(
            color=color,
            opacity=opacity,
            transparent=True,
            side=2,  # double-sided
        )
        mesh = g.TriangularMeshGeometry(vertices=verts, faces=faces)
        self.vis[path].set_object(mesh, material)

    def draw_trajectory(
        self,
        name: str,
        points: np.ndarray,
        color: int = 0x00FF00,
    ) -> None:
        """Draw a trajectory as a polyline with point markers.

        Args:
            name: Unique name for this trajectory object.
            points: (N, 3) trajectory positions [m].
            color: Hex color for the line.
        """
        if name is None:
            raise ValueError("name must be provided")
        if self.vis is None or len(points) < 2:
            return

        path = f"overlays/trajectories/{name}"
        pts = np.array(points, dtype=np.float32).T  # (3, N)
        self.vis[path].set_object(
            g.Line(
                g.PointsGeometry(pts),
                g.LineBasicMaterial(color=color, linewidth=3),
            )
        )

    def draw_arrow_line(
        self,
        name: str,
        start: np.ndarray,
        end: np.ndarray,
        color: int = 0x4488FF,
    ) -> None:
        """Draw a single directional line (e.g. plane normal arrow).

        Args:
            name: Unique name for this arrow.
            start: Start position [m] (3,).
            end: End position [m] (3,).
            color: Hex color.
        """
        if name is None:
            raise ValueError("name must be provided")
        if self.vis is None:
            return

        path = f"overlays/arrows/{name}"
        vertices = np.array([start, end], dtype=np.float32).T  # (3, 2)
        self.vis[path].set_object(
            g.Line(
                g.PointsGeometry(vertices),
                g.LineBasicMaterial(color=color, linewidth=4),
            )
        )

    def clear_swing_plane(self) -> None:
        """Clear all swing plane and trajectory overlays."""
        if self.vis:
            self.vis["overlays/swing_plane"].delete()
            self.vis["overlays/trajectories"].delete()
            self.vis["overlays/arrows"].delete()

    def _draw_arrow(
        self, path: str, start: np.ndarray, vec: np.ndarray, color_hex: int
    ) -> None:
        if path is None:
            raise ValueError("path must be provided")
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

    def _rgba_to_int(self, rgba: Any) -> int:
        if rgba is None:
            return 0
        hex_str = rgba_to_hex(rgba, include_alpha=False)
        return int(hex_str[1:], 16)
