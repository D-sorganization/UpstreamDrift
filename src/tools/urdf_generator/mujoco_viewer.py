# mypy: ignore-errors
# MuJoCo types are dynamically imported and mypy cannot resolve them statically
"""MuJoCo-based 3D visualization for URDF preview.

Implements Task 2.1: MuJoCo Visualization Embed per Phase 2 roadmap.
Provides real-time URDF preview via MJCF conversion.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.810

# MuJoCo is optional - gracefully handle missing
try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore[assignment]
    MUJOCO_AVAILABLE = False


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
        mjcf_parts = [
            f'<mujoco model="{robot_name}">',
            f'  <option gravity="0 0 -{GRAVITY_M_S2}" timestep="0.002"/>',
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
        return f"""
<mujoco model="default">
  <option gravity="0 0 -{GRAVITY_M_S2}" timestep="0.002"/>
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

        # Camera parameters
        self.azimuth = 90.0
        self.elevation = -20.0
        self.distance = 3.0
        self.lookat = np.array([0.0, 0.0, 0.5])

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

            # Create renderer (args: model, width, height)
            self._renderer = mujoco.Renderer(self._model, self.width, self.height)

            # Initialize persistent camera for efficiency
            self._camera = mujoco.MjvCamera()

            # Forward kinematics to set initial positions
            mujoco.mj_forward(self._model, self._data)

            logger.info("MuJoCo model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load MJCF: {e}")
            self._model = None
            self._data = None
            return False

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

            # Update scene with configured camera
            self._renderer.update_scene(
                self._data,
                camera=self._camera,
            )

            # Render to RGB array
            image = self._renderer.render()
            return image

        except Exception as e:
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
    - Visualization toggles (collision, frames, joints)
    - Physics sanity checks
    """

    # Signals
    validation_error = pyqtSignal(str)
    model_loaded = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the MuJoCo viewer widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._urdf_content = ""
        self._renderer: MuJoCoOffscreenRenderer | None = None
        self._last_mouse_pos: QPointF | None = None
        self._current_image: QImage | None = None

        # Visualization options
        self._show_collision = False
        self._show_frames = True
        self._show_joint_limits = False

        self._setup_ui()
        self._setup_renderer()

        # Render timer
        self._render_timer = QTimer()
        self._render_timer.timeout.connect(self._update_render)
        self._render_timer.start(50)  # 20 FPS

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()

        self._collision_checkbox = QCheckBox("Collision")
        self._collision_checkbox.toggled.connect(self._on_collision_toggled)
        toolbar.addWidget(self._collision_checkbox)

        self._frames_checkbox = QCheckBox("Frames")
        self._frames_checkbox.setChecked(True)
        self._frames_checkbox.toggled.connect(self._on_frames_toggled)
        toolbar.addWidget(self._frames_checkbox)

        self._joints_checkbox = QCheckBox("Joint Limits")
        self._joints_checkbox.toggled.connect(self._on_joints_toggled)
        toolbar.addWidget(self._joints_checkbox)

        toolbar.addStretch()

        self._launch_btn = QPushButton("Launch Full Viewer")
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

        if not MUJOCO_AVAILABLE:
            self._status_label.setText("⚠️ MuJoCo not installed")
            self._update_placeholder("MuJoCo not installed.\n\npip install mujoco")

    def _setup_renderer(self) -> None:
        """Initialize the offscreen renderer."""
        if MUJOCO_AVAILABLE:
            self._renderer = MuJoCoOffscreenRenderer(640, 480)

    def _update_placeholder(self, message: str) -> None:
        """Show a placeholder message."""
        self._viewport.setText(message)

    def update_visualization(self, urdf_content: str) -> None:
        """Update visualization with new URDF content.

        Args:
            urdf_content: URDF XML string.
        """
        self._urdf_content = urdf_content

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

        # Convert to MJCF
        mjcf_content = URDFToMJCFConverter.convert(urdf_content)

        # Load model
        success = self._renderer.load_mjcf(mjcf_content)
        self.model_loaded.emit(success)

        if success:
            link_count = urdf_content.count("<link")
            joint_count = urdf_content.count("<joint")
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
        """Handle collision visualization toggle."""
        self._show_collision = checked
        logger.info(f"Collision visualization: {checked}")

    def _on_frames_toggled(self, checked: bool) -> None:
        """Handle frames visualization toggle."""
        self._show_frames = checked
        logger.info(f"Frame visualization: {checked}")

    def _on_joints_toggled(self, checked: bool) -> None:
        """Handle joint limits visualization toggle."""
        self._show_joint_limits = checked
        logger.info(f"Joint limits visualization: {checked}")

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

        except Exception as e:
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
