"""Backend classes for the MuJoCo URDF viewer.

Contains:
- VisualizationFlags: visualization toggle configuration
- URDFToMJCFConverter: converts URDF XML to MuJoCo MJCF format
- MuJoCoOffscreenRenderer: off-screen renderer using MuJoCo
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import defusedxml.ElementTree as ET
import numpy as np

from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.engine_core.engine_availability import MUJOCO_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from typing import Any

logger = get_logger(__name__)

# MuJoCo is optional - gracefully handle missing
if MUJOCO_AVAILABLE:
    import mujoco
else:
    mujoco = None  # type: ignore[assignment]


@dataclass
class VisualizationFlags:
    """Configuration for visualization options.

    Controls what elements are displayed in the MuJoCo 3D view.
    """

    show_collision: bool = False
    show_frames: bool = True
    show_joint_limits: bool = False
    show_contacts: bool = False
    show_com: bool = False  # Center of mass visualization

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary for serialization."""
        return {
            "collision": self.show_collision,
            "frames": self.show_frames,
            "joint_limits": self.show_joint_limits,
            "contacts": self.show_contacts,
            "com": self.show_com,
        }


class URDFToMJCFConverter:
    """Convert URDF to MJCF for MuJoCo visualization.

    This is a simplified converter for preview purposes only.
    For production use, consider the official mujoco URDF import.
    """

    @staticmethod
    def convert(urdf_content: str) -> str:
        """Convert URDF XML to MJCF XML.

        Args:
            urdf_content: URDF XML string.

        Returns:
            MJCF XML string suitable for MuJoCo.

        Note:
            This is a simplified converter for visualization only.
            Complex URDF features may not be fully supported.
        """
        try:
            root = ET.fromstring(urdf_content)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse URDF: {e}")
            return URDFToMJCFConverter._get_default_mjcf()

        robot_name = root.get("name", "robot")

        # Build MJCF
        gravity_val = float(GRAVITY_M_S2)
        mjcf_parts = [
            f'<mujoco model="{robot_name}">',
            f'  <option gravity="0 0 -{gravity_val}" timestep="0.002"/>',
            '  <compiler angle="radian" inertiafromgeom="auto"/>',
            "",
            "  <worldbody>",
            '    <light name="light" pos="0 0 3" dir="0 0 -1"/>',
            '    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>',
        ]

        # Process links
        links = root.findall(".//link")
        for link in links:
            link_name = link.get("name", "unnamed")
            mjcf_parts.append(f'    <body name="{link_name}" pos="0 0 0.5">')

            # Process visual
            visual = link.find("visual")
            if visual is not None:
                geometry = visual.find("geometry")
                if geometry is not None:
                    geom_xml = URDFToMJCFConverter._convert_geometry(geometry)
                    if geom_xml:
                        mjcf_parts.append(f"      {geom_xml}")

            # Process inertial
            inertial = link.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    mass = mass_elem.get("value", "1.0")
                    mjcf_parts.append(
                        f'      <inertial pos="0 0 0" mass="{mass}" '
                        f'diaginertia="0.1 0.1 0.1"/>'
                    )
            else:
                # Default inertial
                mjcf_parts.append(
                    '      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>'
                )

            mjcf_parts.append("    </body>")

        mjcf_parts.extend(
            [
                "  </worldbody>",
                "</mujoco>",
            ]
        )

        return "\n".join(mjcf_parts)

    @staticmethod
    def _convert_geometry(geometry: ET.Element) -> str | None:
        """Convert URDF geometry to MJCF geom."""
        box = geometry.find("box")
        if box is not None:
            size = box.get("size", "0.1 0.1 0.1")
            # Box size in URDF is full dimensions, MJCF uses half-sizes
            parts = [float(x) / 2 for x in size.split()]
            return f'<geom type="box" size="{parts[0]} {parts[1]} {parts[2]}"/>'

        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = cylinder.get("radius", "0.05")
            length = cylinder.get("length", "0.1")
            half_len = float(length) / 2
            return f'<geom type="cylinder" size="{radius} {half_len}"/>'

        sphere = geometry.find("sphere")
        if sphere is not None:
            radius = sphere.get("radius", "0.05")
            return f'<geom type="sphere" size="{radius}"/>'

        mesh = geometry.find("mesh")
        if mesh is not None:
            # Mesh files require external assets - use sphere placeholder
            return '<geom type="sphere" size="0.05" rgba="0.5 0.5 0.5 1"/>'

        return None

    @staticmethod
    def _get_default_mjcf() -> str:
        """Return a default MJCF scene for empty/invalid URDF."""
        gravity_val = float(GRAVITY_M_S2)
        return f"""
<mujoco model="default">
  <option gravity="0 0 -{gravity_val}" timestep="0.002"/>
  <worldbody>
    <light name="light" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="placeholder" pos="0 0 0.5">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class MuJoCoOffscreenRenderer:
    """Offscreen renderer for MuJoCo scenes.

    Renders to a numpy array that can be displayed in Qt.
    Supports visualization toggles for collision, frames, joints, and contacts.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        """Initialize the offscreen renderer.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
        """
        if width is None:
            raise ValueError("width must be provided")
        self.width = width
        self.height = height
        self._model: Any | None = None
        self._data: Any | None = None
        self._renderer: Any | None = None
        self._scene: Any | None = None
        self._camera: Any | None = None
        self._scene_option: Any | None = None  # mjvOption for visualization flags

        # Camera parameters
        self.azimuth = 90.0
        self.elevation = -20.0
        self.distance = 3.0
        self.lookat = np.array([0.0, 0.0, 0.5])

        # Visualization flags
        self.vis_flags = VisualizationFlags()

    def load_urdf_file(self, urdf_path: str) -> bool:
        """Load URDF model from file path.

        This allows MuJoCo to resolve relative mesh paths correctly.
        Preprocesses the URDF to fix zero/small mass and inertia values
        that MuJoCo rejects.

        Args:
            urdf_path: Path to URDF file.

        Returns:
            True if loaded successfully.
        """
        if urdf_path is None:
            raise ValueError("urdf_path must be provided")
        if not MUJOCO_AVAILABLE:
            logger.warning("MuJoCo not available")
            return False

        try:
            from pathlib import Path

            # Read and preprocess URDF to fix small masses/inertias
            urdf_content = Path(urdf_path).read_text(encoding="utf-8")
            fixed_content = self._fix_urdf_inertials(urdf_content)

            # Save to a uniquely-named temp file in the same directory so that
            # relative mesh paths resolve correctly and parallel runs don't race.
            urdf_dir = Path(urdf_path).parent
            fd, temp_path_str = tempfile.mkstemp(suffix=".urdf", dir=urdf_dir)
            temp_urdf = Path(temp_path_str)
            try:
                import os

                os.close(fd)
            except OSError:
                pass
            temp_urdf.write_text(fixed_content, encoding="utf-8")

            try:
                self._model = mujoco.MjModel.from_xml_path(str(temp_urdf))
                self._data = mujoco.MjData(self._model)

                # Use model's offscreen dimensions to avoid framebuffer mismatch
                render_width = min(self.width, self._model.vis.global_.offwidth)
                render_height = min(self.height, self._model.vis.global_.offheight)

                # Create renderer with compatible dimensions
                # Note: MuJoCo Renderer takes (model, height, width) not (model, width, height)
                self._renderer = mujoco.Renderer(
                    self._model, render_height, render_width
                )

                # Initialize persistent camera for efficiency
                self._camera = mujoco.MjvCamera()

                # Initialize scene options for visualization toggles
                self._scene_option = mujoco.MjvOption()
                self._apply_visualization_flags()

                # Forward kinematics to set initial positions
                mujoco.mj_forward(self._model, self._data)

                logger.info(
                    f"MuJoCo model loaded from file: {urdf_path} "
                    f"(render size: {render_width}x{render_height})"
                )
                return True
            finally:
                # Clean up temp file
                if temp_urdf.exists():
                    temp_urdf.unlink()

        except ImportError as e:
            logger.error(f"Failed to load URDF file: {e}")
            self._model = None
            self._data = None
            return False

    def _fix_urdf_inertials(self, urdf_content: str) -> str:
        """Fix zero/small mass and inertia values in URDF.

        MuJoCo requires minimum mass and inertia values (mjMINVAL).
        Also adds MuJoCo visual settings for offscreen rendering.

        Args:
            urdf_content: Original URDF content.

        Returns:
            Fixed URDF content.
        """
        if urdf_content is None:
            raise ValueError("urdf_content must be provided")
        import re

        min_mass = 0.001  # 1 gram minimum
        min_inertia = 0.0001  # Minimum inertia value

        # Fix small mass values
        urdf_content = re.sub(
            r'<mass\s+value="([^"]+)"',
            lambda m: f'<mass value="{max(float(m.group(1)), min_mass)}"',
            urdf_content,
        )

        # Fix zero inertia values
        def fix_inertia_attr(attr: str, content: str) -> str:
            """Replace near-zero diagonal inertia values with the minimum."""
            if attr is None:
                raise ValueError("attr must be provided")
            pattern = rf'{attr}="([^"]+)"'

            def replace(m: re.Match) -> str:
                """Substitute the matched inertia value if below threshold."""
                val = float(m.group(1))
                # Only fix diagonal elements (ixx, iyy, izz)
                if attr in ("ixx", "iyy", "izz") and val < min_inertia:
                    return f'{attr}="{min_inertia}"'
                return m.group(0)

            return re.sub(pattern, replace, content)

        for attr in ("ixx", "iyy", "izz"):
            urdf_content = fix_inertia_attr(attr, urdf_content)

        # Add MuJoCo extension for larger offscreen framebuffer
        # This allows rendering at higher resolutions
        mujoco_extension = """
  <mujoco>
    <visual>
      <global offwidth="1024" offheight="1024"/>
    </visual>
  </mujoco>
"""
        # Insert mujoco extension after robot tag opening
        urdf_content = re.sub(
            r"(<robot[^>]*>)",
            r"\1" + mujoco_extension,
            urdf_content,
            count=1,
        )

        return urdf_content

    def load_mjcf(self, mjcf_content: str) -> bool:
        """Load MJCF model from string.

        Args:
            mjcf_content: MJCF XML string.

        Returns:
            True if loaded successfully.
        """
        if mjcf_content is None:
            raise ValueError("mjcf_content must be provided")
        if not MUJOCO_AVAILABLE:
            logger.warning("MuJoCo not available")
            return False

        try:
            self._model = mujoco.MjModel.from_xml_string(mjcf_content)
            self._data = mujoco.MjData(self._model)

            # Create renderer (args: model, height, width)
            self._renderer = mujoco.Renderer(self._model, self.height, self.width)

            # Initialize persistent camera for efficiency
            self._camera = mujoco.MjvCamera()

            # Initialize scene options for visualization toggles
            self._scene_option = mujoco.MjvOption()
            self._apply_visualization_flags()

            # Forward kinematics to set initial positions
            mujoco.mj_forward(self._model, self._data)

            logger.info("MuJoCo model loaded successfully")
            return True

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to load MJCF: {e}")
            self._model = None
            self._data = None
            return False

    def _apply_visualization_flags(self) -> None:
        """Apply visualization flags to MuJoCo scene options.

        Maps VisualizationFlags to MuJoCo's mjvOption flags.
        """
        if not MUJOCO_AVAILABLE or self._scene_option is None:
            return

        # MuJoCo visualization flags reference:
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvoption

        # Frame visualization (coordinate frames at bodies)
        self._scene_option.frame = (
            mujoco.mjtFrame.mjFRAME_BODY.value
            if self.vis_flags.show_frames
            else mujoco.mjtFrame.mjFRAME_NONE.value
        )

        # Collision geometry vs visual geometry
        # In MuJoCo, geomgroup controls visibility of geometry groups
        # Group 0 = collision, Group 1 = visual (typically)
        # flags.geomgroup is a 6-element array where each element toggles a group
        if self.vis_flags.show_collision:
            # Show collision geoms (group 0)
            self._scene_option.geomgroup[0] = 1
        else:
            # Hide collision geoms by default, show visual
            self._scene_option.geomgroup[0] = 0

        # Contact point visualization
        contact_flag_index = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT.value
        self._scene_option.flags[contact_flag_index] = self.vis_flags.show_contacts

        # Contact force visualization (arrows)
        contact_force_index = mujoco.mjtVisFlag.mjVIS_CONTACTFORCE.value
        self._scene_option.flags[contact_force_index] = self.vis_flags.show_contacts

        # Joint visualization
        joint_flag_index = mujoco.mjtVisFlag.mjVIS_JOINT.value
        self._scene_option.flags[joint_flag_index] = self.vis_flags.show_joint_limits

        # Center of mass visualization (if enabled)
        com_flag_index = mujoco.mjtVisFlag.mjVIS_COM.value
        self._scene_option.flags[com_flag_index] = self.vis_flags.show_com

        logger.debug(f"Applied visualization flags: {self.vis_flags.to_dict()}")

    def set_visualization_flags(self, flags: VisualizationFlags) -> None:
        """Update visualization flags and re-apply to scene.

        Args:
            flags: New visualization flags configuration.
        """
        if flags is None:
            raise ValueError("flags must be provided")
        self.vis_flags = flags
        self._apply_visualization_flags()

    def render(self) -> np.ndarray | None:
        """Render the current scene.

        Returns:
            RGB image as numpy array (H, W, 3), or None if rendering fails.
        """
        if not MUJOCO_AVAILABLE or self._model is None:
            return None

        try:
            # Configure camera parameters before scene update
            self._camera.azimuth = self.azimuth  # type: ignore[union-attr]
            self._camera.elevation = self.elevation  # type: ignore[union-attr]
            self._camera.distance = self.distance  # type: ignore[union-attr]
            self._camera.lookat[:] = self.lookat  # type: ignore[union-attr]

            # Update scene with configured camera and visualization options
            if self._scene_option is not None:
                self._renderer.update_scene(  # type: ignore[union-attr]
                    self._data,
                    camera=self._camera,
                    scene_option=self._scene_option,
                )
            else:
                self._renderer.update_scene(  # type: ignore[union-attr]
                    self._data,
                    camera=self._camera,
                )

            # Render to RGB array
            image = self._renderer.render()  # type: ignore[union-attr]
            return image

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Render failed: {e}")
            return None

    def rotate_camera(self, d_azimuth: float, d_elevation: float) -> None:
        """Rotate camera by delta angles."""
        if d_azimuth is None:
            raise ValueError("d_azimuth must be provided")
        self.azimuth += d_azimuth
        self.elevation = max(-89, min(89, self.elevation + d_elevation))

    def zoom_camera(self, factor: float) -> None:
        """Zoom camera by factor."""
        if factor is None:
            raise ValueError("factor must be provided")
        self.distance *= factor
        self.distance = max(0.5, min(20.0, self.distance))
