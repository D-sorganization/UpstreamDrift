"""3D visualization widget for URDF preview."""

import logging
import math

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class VisualizationWidget(QWidget):
    """Widget for 3D visualization of URDF models.

    This widget provides a container for the visualization backend.
    It currently uses a simple 2.5D grid visualization as a fallback.
    For full 3D rendering, the MuJoCo viewer is recommended.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the visualization widget.

        Args:
            parent: Parent widget, if any.
        """
        super().__init__(parent)
        self.urdf_content = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # 3D Visualization Widget
        self.gl_widget = Simple3DVisualizationWidget()
        layout.addWidget(self.gl_widget)

        # Info Label (below the 3D view)
        self.info_label = QLabel("No URDF content loaded")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet(
            """
            QLabel {
                background-color: #f5f5f5;
                color: #333;
                padding: 5px;
                border-top: 1px solid #ddd;
            }
        """
        )
        layout.addWidget(self.info_label)

    def update_visualization(self, urdf_content: str) -> None:
        """Update the 3D visualization with new URDF content.

        Args:
            urdf_content: URDF XML content to visualize.
        """
        self.urdf_content = urdf_content

        # Update the status text
        if urdf_content.strip():
            # Count links and joints in the URDF
            link_count = urdf_content.count("<link")
            joint_count = urdf_content.count("<joint")

            self.info_label.setText(
                f"Links: {link_count} | Joints: {joint_count} (Grid View - Install MuJoCo for 3D)"
            )
        else:
            self.info_label.setText("No URDF content loaded")

        # Force update of the GL widget
        self.gl_widget.update()

        logger.info(
            f"Visualization updated with URDF content ({len(urdf_content)} characters)"
        )

    def clear(self) -> None:
        """Clear the visualization."""
        self.urdf_content = ""
        self.info_label.setText("No URDF content loaded")
        self.gl_widget.update()
        logger.info("Visualization cleared")

    def reset_view(self) -> None:
        """Reset the 3D view to default position."""
        self.gl_widget.reset_view()
        logger.info("View reset requested")


class Simple3DVisualizationWidget(QOpenGLWidget):
    """Simple OpenGL-based 3D visualization widget using QPainter.

    This serves as a lightweight fallback viewer when full 3D engines
    (like MuJoCo) are not available. It renders a 3D grid and axes
    using 2D QPainter operations with manual projection.
    """

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

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS

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

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse move events.

        Args:
            event: Mouse event.
        """
        if event is not None and self.last_mouse_pos is not None:
            dx = event.position().x() - self.last_mouse_pos.x()
            dy = event.position().y() - self.last_mouse_pos.y()

            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5

            # Clamp vertical rotation
            self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))

            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Handle mouse release events.

        Args:
            event: Mouse event.
        """
        if event is not None:
            self.last_mouse_pos = None

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
