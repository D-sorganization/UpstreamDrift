# mypy: ignore-errors
# MuJoCo types are dynamically imported and mypy cannot resolve them statically
"""MuJoCo-based 3D visualization for URDF preview.

Implements Task 2.1: MuJoCo Visualization Embed per Phase 2 roadmap.
Provides real-time URDF preview via MJCF conversion.

Issue #755: Enhanced visualization toggles for collision, frames, joints, and contacts.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import defusedxml.ElementTree as ET
import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.constants import GRAVITY_M_S2  # DRY: Use centralized constant
from src.shared.python.engine_availability import MUJOCO_AVAILABLE
from src.shared.python.logging_config import get_logger

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
        if not MUJOCO_AVAILABLE:
            logger.warning("MuJoCo not available")
            return False

        try:
            from pathlib import Path

            # Read and preprocess URDF to fix small masses/inertias
            urdf_content = Path(urdf_path).read_text(encoding="utf-8")
            fixed_content = self._fix_urdf_inertials(urdf_content)

            # Save to temp file in same directory for mesh resolution
            urdf_dir = Path(urdf_path).parent
            temp_urdf = urdf_dir / "_temp_fixed_model.urdf"
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
            pattern = rf'{attr}="([^"]+)"'

            def replace(m: re.Match) -> str:
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
            self._camera.azimuth = self.azimuth
            self._camera.elevation = self.elevation
            self._camera.distance = self.distance
            self._camera.lookat[:] = self.lookat

            # Update scene with configured camera and visualization options
            if self._scene_option is not None:
                self._renderer.update_scene(
                    self._data,
                    camera=self._camera,
                    scene_option=self._scene_option,
                )
            else:
                self._renderer.update_scene(
                    self._data,
                    camera=self._camera,
                )

            # Render to RGB array
            image = self._renderer.render()
            return image

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Render failed: {e}")
            return None

    def rotate_camera(self, d_azimuth: float, d_elevation: float) -> None:
        """Rotate camera by delta angles."""
        self.azimuth += d_azimuth
        self.elevation = max(-89, min(89, self.elevation + d_elevation))

    def zoom_camera(self, factor: float) -> None:
        """Zoom camera by factor."""
        self.distance *= factor
        self.distance = max(0.5, min(20.0, self.distance))


class MuJoCoViewerWidget(QWidget):
    """Qt widget for MuJoCo-based URDF visualization.

    Features:
    - Real-time URDF preview via MJCF conversion
    - Mouse-based camera control (rotate, zoom)
    - Visualization toggles (collision, frames, joints, contacts)
    - Physics sanity checks
    - Clear headless fallback messaging

    Issue #755: Enhanced with working toggles and contacts visualization.
    """

    # Signals
    validation_error = pyqtSignal(str)
    model_loaded = pyqtSignal(bool)
    visualization_changed = pyqtSignal(dict)  # Emitted when toggles change

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the MuJoCo viewer widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._urdf_content = ""
        self._urdf_path: str | None = None
        self._renderer: MuJoCoOffscreenRenderer | None = None
        self._last_mouse_pos: QPointF | None = None
        self._current_image: QImage | None = None

        # Visualization flags (using dataclass)
        self._vis_flags = VisualizationFlags()

        self._setup_ui()
        self._setup_renderer()

        # Render timer
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._update_render)
        self._render_timer.start(50)  # 20 FPS

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Toolbar with visualization toggles
        toolbar = QHBoxLayout()

        # Create toggle group with visual separator
        toggle_frame = QFrame()
        toggle_frame.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border-radius: 4px;
                padding: 2px;
            }
            QCheckBox {
                color: #ddd;
                padding: 4px 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border-radius: 2px;
            }
        """)
        toggle_layout = QHBoxLayout(toggle_frame)
        toggle_layout.setContentsMargins(4, 2, 4, 2)
        toggle_layout.setSpacing(8)

        self._collision_checkbox = QCheckBox("Collision")
        self._collision_checkbox.setToolTip("Show collision geometry (red wireframe)")
        self._collision_checkbox.toggled.connect(self._on_collision_toggled)
        toggle_layout.addWidget(self._collision_checkbox)

        self._frames_checkbox = QCheckBox("Frames")
        self._frames_checkbox.setChecked(True)
        self._frames_checkbox.setToolTip("Show coordinate frames at each body")
        self._frames_checkbox.toggled.connect(self._on_frames_toggled)
        toggle_layout.addWidget(self._frames_checkbox)

        self._joints_checkbox = QCheckBox("Joints")
        self._joints_checkbox.setToolTip("Show joint axes and limits")
        self._joints_checkbox.toggled.connect(self._on_joints_toggled)
        toggle_layout.addWidget(self._joints_checkbox)

        self._contacts_checkbox = QCheckBox("Contacts")
        self._contacts_checkbox.setToolTip("Show contact points and forces")
        self._contacts_checkbox.toggled.connect(self._on_contacts_toggled)
        toggle_layout.addWidget(self._contacts_checkbox)

        toolbar.addWidget(toggle_frame)
        toolbar.addStretch()

        self._launch_btn = QPushButton("Launch Full Viewer")
        self._launch_btn.setToolTip("Open in MuJoCo's interactive viewer")
        self._launch_btn.clicked.connect(self._launch_external_viewer)
        toolbar.addWidget(self._launch_btn)

        layout.addLayout(toolbar)

        # Viewport
        self._viewport = QLabel()
        self._viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viewport.setMinimumSize(320, 240)
        self._viewport.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        self._viewport.setMouseTracking(True)
        layout.addWidget(self._viewport, stretch=1)

        # Status bar
        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_label)

        # Headless fallback with clear messaging
        if not MUJOCO_AVAILABLE:
            self._status_label.setText(
                "⚠️ MuJoCo not installed - running in headless mode"
            )
            self._disable_toggles()
            self._update_headless_placeholder()

    def _setup_renderer(self) -> None:
        """Initialize the offscreen renderer."""
        if MUJOCO_AVAILABLE:
            # Use larger framebuffer to avoid dimension mismatch errors
            self._renderer = MuJoCoOffscreenRenderer(800, 800)
            # Sync initial flags
            self._renderer.vis_flags = self._vis_flags

    def _disable_toggles(self) -> None:
        """Disable all visualization toggles (for headless mode)."""
        self._collision_checkbox.setEnabled(False)
        self._frames_checkbox.setEnabled(False)
        self._joints_checkbox.setEnabled(False)
        self._contacts_checkbox.setEnabled(False)
        self._launch_btn.setEnabled(False)

    def _update_headless_placeholder(self) -> None:
        """Show a clear headless fallback message."""
        self._viewport.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px dashed #4a4a6a;
                border-radius: 8px;
                color: #8888aa;
                font-size: 14px;
            }
        """)
        self._viewport.setText(
            "🖥️ Headless Mode\n\n"
            "MuJoCo is not installed.\n"
            "3D preview is unavailable.\n\n"
            "To enable 3D visualization:\n"
            "  pip install mujoco\n\n"
            "Model data is still being processed\n"
            "and exported correctly."
        )

    def _update_placeholder(self, message: str) -> None:
        """Show a placeholder message."""
        self._viewport.setText(message)

    def update_visualization(
        self, urdf_content: str, urdf_path: str | None = None
    ) -> None:
        """Update visualization with new URDF content.

        Args:
            urdf_content: URDF XML string.
            urdf_path: Optional path to URDF file for mesh resolution.
        """
        self._urdf_content = urdf_content
        self._urdf_path = urdf_path

        if not MUJOCO_AVAILABLE or not self._renderer:
            # Count elements for display
            link_count = urdf_content.count("<link")
            joint_count = urdf_content.count("<joint")
            self._update_placeholder(
                f"URDF Preview\n\nLinks: {link_count}\nJoints: {joint_count}\n\n"
                "(Install MuJoCo for 3D preview)"
            )
            return

        # Validate URDF
        validation_errors = self._validate_urdf(urdf_content)
        if validation_errors:
            self.validation_error.emit("\n".join(validation_errors))

        # Try to load from file path first (for mesh resolution)
        success = False
        if urdf_path:
            logger.info(f"Loading URDF from path: {urdf_path}")
            success = self._renderer.load_urdf_file(urdf_path)

        # Fallback to MJCF conversion if file loading fails
        if not success:
            logger.info("Falling back to MJCF conversion")
            mjcf_content = URDFToMJCFConverter.convert(urdf_content)
            success = self._renderer.load_mjcf(mjcf_content)

        self.model_loaded.emit(success)

        link_count = urdf_content.count("<link")
        joint_count = urdf_content.count("<joint")
        if success:
            self._status_label.setText(
                f"✓ Model loaded: {link_count} links, {joint_count} joints"
            )
        else:
            self._status_label.setText("⚠️ Failed to load model")

    def _validate_urdf(self, urdf_content: str) -> list[str]:
        """Validate URDF for physics sanity.

        Args:
            urdf_content: URDF XML string.

        Returns:
            List of validation error messages.
        """
        errors = []

        try:
            root = ET.fromstring(urdf_content)
        except ET.ParseError as e:
            return [f"XML Parse Error: {e}"]

        # Check inertial properties
        for link in root.findall(".//link"):
            link_name = link.get("name", "unnamed")
            inertial = link.find("inertial")

            if inertial is not None:
                inertia = inertial.find("inertia")
                if inertia is not None:
                    # Check positive definiteness (diagonal elements)
                    ixx = float(inertia.get("ixx", "0"))
                    iyy = float(inertia.get("iyy", "0"))
                    izz = float(inertia.get("izz", "0"))
                    # Off-diagonal elements read but not checked in simple validation
                    _ = inertia.get("ixy", "0")
                    _ = inertia.get("ixz", "0")
                    _ = inertia.get("iyz", "0")

                    # Simple check: diagonal elements should be positive
                    if ixx <= 0 or iyy <= 0 or izz <= 0:
                        errors.append(
                            f"Link '{link_name}': Non-positive inertia diagonal"
                        )

        # Check joint axes
        for joint in root.findall(".//joint"):
            joint_name = joint.get("name", "unnamed")
            axis_elem = joint.find("axis")

            if axis_elem is not None:
                xyz = axis_elem.get("xyz", "0 0 1")
                axis = [float(x) for x in xyz.split()]
                norm = sum(x * x for x in axis) ** 0.5

                if abs(norm - 1.0) > 0.01:
                    errors.append(
                        f"Joint '{joint_name}': Axis not normalized (|axis|={norm:.3f})"
                    )

        return errors

    def _update_render(self) -> None:
        """Update the rendered image."""
        if not self._renderer:
            return

        image = self._renderer.render()
        if image is not None:
            # Convert numpy array to QImage
            h, w, c = image.shape
            bytes_per_line = c * w
            q_image = QImage(
                image.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_image)

            # Scale to fit viewport
            scaled = pixmap.scaled(
                self._viewport.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._viewport.setPixmap(scaled)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse press for camera control."""
        if event and event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse move for camera rotation."""
        if event and self._last_mouse_pos and self._renderer:
            dx = event.position().x() - self._last_mouse_pos.x()
            dy = event.position().y() - self._last_mouse_pos.y()

            self._renderer.rotate_camera(dx * 0.5, dy * 0.5)
            self._last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse release."""
        self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Handle mouse wheel for zoom."""
        if event and self._renderer:
            delta = event.angleDelta().y()
            factor = 0.9 if delta > 0 else 1.1
            self._renderer.zoom_camera(factor)

    def _on_collision_toggled(self, checked: bool) -> None:
        """Handle collision visualization toggle.

        Args:
            checked: Whether collision geometry should be shown.
        """
        self._vis_flags.show_collision = checked
        self._update_renderer_flags()
        logger.info(f"Collision visualization: {checked}")

    def _on_frames_toggled(self, checked: bool) -> None:
        """Handle frames visualization toggle.

        Args:
            checked: Whether coordinate frames should be shown.
        """
        self._vis_flags.show_frames = checked
        self._update_renderer_flags()
        logger.info(f"Frame visualization: {checked}")

    def _on_joints_toggled(self, checked: bool) -> None:
        """Handle joint limits visualization toggle.

        Args:
            checked: Whether joint axes and limits should be shown.
        """
        self._vis_flags.show_joint_limits = checked
        self._update_renderer_flags()
        logger.info(f"Joint limits visualization: {checked}")

    def _on_contacts_toggled(self, checked: bool) -> None:
        """Handle contacts visualization toggle.

        Args:
            checked: Whether contact points and forces should be shown.
        """
        self._vis_flags.show_contacts = checked
        self._update_renderer_flags()
        logger.info(f"Contacts visualization: {checked}")

    def _update_renderer_flags(self) -> None:
        """Sync visualization flags to the renderer."""
        if self._renderer:
            self._renderer.set_visualization_flags(self._vis_flags)
            self.visualization_changed.emit(self._vis_flags.to_dict())

    def _launch_external_viewer(self) -> None:
        """Launch MuJoCo's standalone viewer."""
        if not self._urdf_content:
            logger.warning("No URDF content to view")
            return

        try:
            # Convert to MJCF and save to temp file
            mjcf_content = URDFToMJCFConverter.convert(self._urdf_content)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", delete=False
            ) as f:
                f.write(mjcf_content)
                temp_path = f.name

            # Launch viewer subprocess
            cmd = [
                sys.executable,
                "-c",
                f"import mujoco; import mujoco.viewer; "
                f"m=mujoco.MjModel.from_xml_path(r'{temp_path}'); "
                f"mujoco.viewer.launch(m)",
            ]
            subprocess.Popen(cmd)
            logger.info("Launched external MuJoCo viewer")

        except ImportError as e:
            logger.error(f"Failed to launch viewer: {e}")

    def clear(self) -> None:
        """Clear the visualization."""
        self._urdf_content = ""
        self._update_placeholder("No URDF content")
        self._status_label.setText("")

    def reset_view(self) -> None:
        """Reset camera to default position."""
        if self._renderer:
            self._renderer.azimuth = 90.0
            self._renderer.elevation = -20.0
            self._renderer.distance = 3.0
            self._renderer.lookat = np.array([0.0, 0.0, 0.5])

    def get_visualization_flags(self) -> VisualizationFlags:
        """Get current visualization flags.

        Returns:
            Current visualization configuration.
        """
        return self._vis_flags

    def set_visualization_flags(self, flags: VisualizationFlags) -> None:
        """Set visualization flags programmatically.

        Args:
            flags: New visualization configuration.
        """
        self._vis_flags = flags

        # Update checkboxes to match
        self._collision_checkbox.setChecked(flags.show_collision)
        self._frames_checkbox.setChecked(flags.show_frames)
        self._joints_checkbox.setChecked(flags.show_joint_limits)
        self._contacts_checkbox.setChecked(flags.show_contacts)

        self._update_renderer_flags()

    def highlight_body(self, body_name: str | None) -> None:
        """Highlight a specific body in the visualization.

        Args:
            body_name: Name of body to highlight, or None to clear.
        """
        # Future enhancement: implement body highlighting in MuJoCo
        # This would require modifying geom colors in the scene
        logger.debug(f"Body highlight requested: {body_name}")

    def is_mujoco_available(self) -> bool:
        """Check if MuJoCo rendering is available.

        Returns:
            True if MuJoCo is installed and renderer is initialized.
        """
        return MUJOCO_AVAILABLE and self._renderer is not None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently loaded model.

        Returns:
            Dictionary with model statistics.
        """
        info: dict[str, Any] = {
            "mujoco_available": MUJOCO_AVAILABLE,
            "model_loaded": False,
            "link_count": 0,
            "joint_count": 0,
        }

        if self._urdf_content:
            info["link_count"] = self._urdf_content.count("<link")
            info["joint_count"] = self._urdf_content.count("<joint")
            info["model_loaded"] = True

        if self._renderer and self._renderer._model is not None:
            info["bodies"] = self._renderer._model.nbody
            info["joints"] = self._renderer._model.njnt
            info["geoms"] = self._renderer._model.ngeom

        return info
