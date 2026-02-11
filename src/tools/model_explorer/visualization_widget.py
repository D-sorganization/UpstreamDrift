"""3D visualization widget for URDF preview."""

import math

import defusedxml.ElementTree as ET
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.shared.python.engine_core.engine_availability import (
    MUJOCO_AVAILABLE,  # noqa: E402
)
from src.shared.python.logging_pkg.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


# Import MuJoCo viewer if available

if MUJOCO_AVAILABLE:

    try:

        from .mujoco_viewer import MuJoCoViewerWidget

        logger.info("MuJoCo 3D viewer available")

    except ImportError as e:

        logger.info(f"MuJoCo viewer widget not available: {e}")

        MuJoCoViewerWidget = None  # type: ignore[misc, assignment]

else:

    logger.info("MuJoCo not available, using fallback grid view")

    MuJoCoViewerWidget = None  # type: ignore[misc, assignment]


class VisualizationWidget(QWidget):
    """Widget for 3D visualization of URDF models.



    This widget provides a container for the visualization backend.

    Uses MuJoCo viewer when available, falls back to simple grid view.



    Signals:

        object_clicked: Emitted when a link/joint is clicked in the 3D view.

            Args: (component_type: str, component_name: str)

        selection_changed: Emitted when selection changes.

            Args: (component_name: str or None)

    """

    # Signals for click-to-highlight feature

    object_clicked = pyqtSignal(str, str)  # component_type, component_name

    selection_changed = pyqtSignal(object)  # component_name or None

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the visualization widget.



        Args:

            parent: Parent widget, if any.

        """

        super().__init__(parent)

        self.urdf_content = ""

        self.urdf_path: str | None = None

        self.use_mujoco = MUJOCO_AVAILABLE

        self.mujoco_widget: MuJoCoViewerWidget | None = None  # type: ignore[assignment]

        self._link_names: list[str] = []  # Cache of link names for selection

        self._joint_names: list[str] = []  # Cache of joint names

        self._selected_object: str | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""

        layout = QVBoxLayout(self)

        # 3D Visualization Widget - use MuJoCo if available

        if self.use_mujoco:

            try:

                self.mujoco_widget = MuJoCoViewerWidget()

                layout.addWidget(self.mujoco_widget)

                logger.info("Using MuJoCo 3D viewer")

            except (RuntimeError, ValueError, OSError) as e:

                logger.warning(f"Failed to create MuJoCo widget: {e}")

                self.use_mujoco = False

        if not self.use_mujoco:

            self.gl_widget = Simple3DVisualizationWidget()

            layout.addWidget(self.gl_widget)

        # Info Label (below the 3D view)

        self.info_label = QLabel("No URDF content loaded")

        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.info_label.setStyleSheet("""

            QLabel {

                background-color: #f5f5f5;

                color: #333;

                padding: 5px;

                border-top: 1px solid #ddd;

            }

        """)

        layout.addWidget(self.info_label)

    def update_visualization(
        self, urdf_content: str, urdf_path: str | None = None
    ) -> None:
        """Update the 3D visualization with new URDF content.



        Args:

            urdf_content: URDF XML content to visualize.

            urdf_path: Optional path to URDF file for mesh resolution.

        """

        self.urdf_content = urdf_content

        self.urdf_path = urdf_path

        # Parse component names for click-to-highlight feature

        self._parse_urdf_components(urdf_content)

        # Update the status text

        if urdf_content.strip():

            # Count links and joints in the URDF

            link_count = urdf_content.count("<link")

            joint_count = urdf_content.count("<joint")

            if self.use_mujoco:

                self.info_label.setText(
                    f"Links: {link_count} | Joints: {joint_count} (MuJoCo 3D View)"
                )

                # Update MuJoCo viewer - pass path for mesh resolution

                if self.mujoco_widget:

                    try:

                        self.mujoco_widget.update_visualization(urdf_content, urdf_path)

                    except (RuntimeError, ValueError, OSError) as e:

                        logger.warning(f"Failed to render in MuJoCo: {e}")

                        self.info_label.setText(
                            f"Links: {link_count} | Joints: {joint_count} (MuJoCo render failed)"
                        )

            else:

                self.info_label.setText(
                    f"Links: {link_count} | Joints: {joint_count} (Grid View)"
                )

                self.gl_widget.update()

        else:

            self.info_label.setText("No URDF content loaded")

        logger.info(
            f"Visualization updated with URDF content ({len(urdf_content)} characters)"
        )

    def clear(self) -> None:
        """Clear the visualization."""

        self.urdf_content = ""

        self.info_label.setText("No URDF content loaded")

        if self.use_mujoco and self.mujoco_widget:

            self.mujoco_widget.clear()

        elif hasattr(self, "gl_widget"):

            self.gl_widget.update()

        logger.info("Visualization cleared")

    def reset_view(self) -> None:
        """Reset the 3D view to default position."""

        if self.use_mujoco and self.mujoco_widget:

            self.mujoco_widget.reset_view()

        elif hasattr(self, "gl_widget"):

            self.gl_widget.reset_view()

        logger.info("View reset requested")

    def _parse_urdf_components(self, urdf_content: str) -> None:
        """Parse URDF content to extract link and joint names for selection.



        Args:

            urdf_content: URDF XML string to parse.

        """

        self._link_names = []

        self._joint_names = []

        if not urdf_content.strip():

            return

        try:

            root = ET.fromstring(urdf_content)

            for child in root:

                name = child.get("name", "")

                if child.tag == "link" and name:

                    self._link_names.append(name)

                elif child.tag == "joint" and name:

                    self._joint_names.append(name)

            logger.debug(
                f"Parsed {len(self._link_names)} links and {len(self._joint_names)} joints"
            )

        except ET.ParseError as e:

            logger.warning(f"Failed to parse URDF for component names: {e}")

    def select_object(self, name: str | None) -> None:
        """Select an object by name and highlight it.



        Args:

            name: Name of the link or joint to select, or None to clear selection.

        """

        self._selected_object = name

        self.selection_changed.emit(name)

        if name:

            # Determine if it's a link or joint

            if name in self._link_names:

                self.object_clicked.emit("link", name)

                logger.info(f"Selected link: {name}")

            elif name in self._joint_names:

                self.object_clicked.emit("joint", name)

                logger.info(f"Selected joint: {name}")

        # Update visualization to show highlight

        if self.use_mujoco and self.mujoco_widget:

            # MuJoCo viewer can highlight specific geoms/bodies

            if hasattr(self.mujoco_widget, "highlight_body"):

                self.mujoco_widget.highlight_body(name)

        elif hasattr(self, "gl_widget"):

            self.gl_widget.set_highlighted_object(name)

            self.gl_widget.update()

    def get_link_names(self) -> list[str]:
        """Get list of link names in the current URDF.



        Returns:

            List of link names.

        """

        return self._link_names.copy()

    def get_joint_names(self) -> list[str]:
        """Get list of joint names in the current URDF.



        Returns:

            List of joint names.

        """

        return self._joint_names.copy()


class Simple3DVisualizationWidget(QOpenGLWidget):
    """Simple OpenGL-based 3D visualization widget using QPainter.



    This serves as a lightweight fallback viewer when full 3D engines

    (like MuJoCo) are not available. It renders a 3D grid and axes

    using 2D QPainter operations with manual projection.



    Signals:

        object_clicked: Emitted when user clicks to select an object.

    """

    # Signal for click-to-highlight

    object_clicked = pyqtSignal(str)  # object name

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the 3D visualization widget.



        Args:

            parent: Parent widget, if any.

        """

        super().__init__(parent)

        # Camera parameters

        self.camera_distance = 5.0

        self.camera_rotation_x = 0.0

        self.camera_rotation_y = 0.0

        # Mouse interaction

        self.last_mouse_pos: QPointF | None = None

        self._is_dragging = False

        # Highlighted object for click-to-highlight

        self._highlighted_object: str | None = None

        # Timer for animation

        self.timer = QTimer()

        self.timer.timeout.connect(self.update)

        self.timer.start(16)  # ~60 FPS

    def set_highlighted_object(self, name: str | None) -> None:
        """Set the currently highlighted object.



        Args:

            name: Name of object to highlight, or None to clear.

        """

        self._highlighted_object = name

        self.update()

    def initializeGL(self) -> None:
        """Initialize OpenGL.



        Note: Not used in this implementation as we use QPainter

        in paintGL for maximum compatibility.

        """

    def resizeGL(self, width: int, height: int) -> None:
        """Handle OpenGL resize.



        Args:

            width: New width.

            height: New height.



        Note: Not used in this implementation.

        """

    def project_point(self, x: float, y: float, z: float) -> tuple[float, float]:
        """Project 3D point to 2D screen coordinates."""

        # 1. Rotate around Y (yaw)

        rad_y = math.radians(self.camera_rotation_y)

        x_r1 = x * math.cos(rad_y) - z * math.sin(rad_y)

        z_r1 = x * math.sin(rad_y) + z * math.cos(rad_y)

        y_r1 = y

        # 2. Rotate around X (pitch)

        rad_x = math.radians(self.camera_rotation_x)

        y_r2 = y_r1 * math.cos(rad_x) - z_r1 * math.sin(rad_x)

        x_r2 = x_r1

        # 3. Project to screen

        # Use simple orthographic-like projection scaled by distance

        scale = self.camera_distance * 40.0

        screen_x = x_r2 * scale

        screen_y = -y_r2 * scale  # Flip Y for screen coordinates (up is negative Y)

        return screen_x, screen_y

    def paintGL(self) -> None:
        """Paint the OpenGL scene using QPainter for fallback visualization."""

        painter = QPainter(self)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background

        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # Center of the widget

        center_x = self.width() / 2

        center_y = self.height() / 2

        painter.translate(center_x, center_y)

        # Draw Grid

        painter.setPen(QPen(QColor(80, 80, 80), 1))

        grid_size = 5

        grid_step = 1.0

        # Draw lines parallel to X and Z

        for i in range(-grid_size, grid_size + 1):

            val = i * grid_step

            # Line parallel to Z (varying Z, fixed X)

            x1, y1 = self.project_point(val, 0, -grid_size * grid_step)

            x2, y2 = self.project_point(val, 0, grid_size * grid_step)

            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # Line parallel to X (varying X, fixed Z)

            x1, y1 = self.project_point(-grid_size * grid_step, 0, val)

            x2, y2 = self.project_point(grid_size * grid_step, 0, val)

            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Draw Axes

        origin_x, origin_y = self.project_point(0, 0, 0)

        # X Axis - Red

        painter.setPen(QPen(QColor(255, 100, 100), 2))

        ax_x, ax_y = self.project_point(1.5, 0, 0)

        painter.drawLine(int(origin_x), int(origin_y), int(ax_x), int(ax_y))

        painter.drawText(int(ax_x), int(ax_y), "X")

        # Y Axis - Green (Up)

        painter.setPen(QPen(QColor(100, 255, 100), 2))

        ay_x, ay_y = self.project_point(0, 1.5, 0)

        painter.drawLine(int(origin_x), int(origin_y), int(ay_x), int(ay_y))

        painter.drawText(int(ay_x), int(ay_y), "Y")

        # Z Axis - Blue

        painter.setPen(QPen(QColor(100, 100, 255), 2))

        az_x, az_y = self.project_point(0, 0, 1.5)

        painter.drawLine(int(origin_x), int(origin_y), int(az_x), int(az_y))

        painter.drawText(int(az_x), int(az_y), "Z")

        # Draw overlay info (reset transform)

        painter.resetTransform()

        painter.setPen(QColor(255, 255, 255))

        painter.drawText(10, 20, f"Zoom: {self.camera_distance:.1f}x")

        painter.drawText(
            10,
            35,
            f"Rot: {self.camera_rotation_x:.1f}, {self.camera_rotation_y:.1f}",
        )

        painter.drawText(10, 50, "Grid View (Fallback)")

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse press events.



        Args:

            event: Mouse event.

        """

        if event is not None and event.button() == Qt.MouseButton.LeftButton:

            self.last_mouse_pos = event.position()

            self._is_dragging = False  # Will become True if mouse moves significantly

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse move events.



        Args:

            event: Mouse event.

        """

        if event is not None and self.last_mouse_pos is not None:

            dx = event.position().x() - self.last_mouse_pos.x()

            dy = event.position().y() - self.last_mouse_pos.y()

            # If moved more than 3 pixels, consider it a drag

            if abs(dx) > 3 or abs(dy) > 3:

                self._is_dragging = True

            self.camera_rotation_y += dx * 0.5

            self.camera_rotation_x += dy * 0.5

            # Clamp vertical rotation

            self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))

            self.last_mouse_pos = event.position()

            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse release events.



        Emits object_clicked signal if this was a click (not a drag).



        Args:

            event: Mouse event.

        """

        if event is not None:

            # If not dragging, treat as a click for selection

            if not self._is_dragging and event.button() == Qt.MouseButton.LeftButton:

                # In the fallback grid view, we can't do proper ray-casting

                # so just emit a general click event for UI feedback

                self.object_clicked.emit("")

            self.last_mouse_pos = None

            self._is_dragging = False

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        """Handle wheel events for zooming.



        Args:

            event: Wheel event.

        """

        if event is not None:

            delta = event.angleDelta().y()

            # Inverse logic for more intuitive zoom (scroll up = zoom in)

            zoom_factor = 1.1 if delta > 0 else 0.9

            self.camera_distance *= zoom_factor

            self.camera_distance = max(0.1, min(10.0, self.camera_distance))

            self.update()

    def reset_view(self) -> None:
        """Reset camera view."""

        self.camera_distance = 1.0

        self.camera_rotation_x = 0.0

        self.camera_rotation_y = 0.0

        self.update()
