"""OpenGL widget for 3D golf swing visualization.

Extracted from golf_gui_application.py for Single Responsibility Principle.
"""

from __future__ import annotations

import logging
import traceback

import moderngl as mgl
import numpy as np
import pandas as pd
from golf_data_core import FrameData, FrameProcessor, RenderConfig
from golf_opengl_renderer import OpenGLRenderer
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

logger = logging.getLogger(__name__)


class GolfVisualizerWidget(QOpenGLWidget):
    """OpenGL widget for 3D golf swing visualization."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.renderer = None
        self.frame_processor = None
        self.current_frame_data = None
        self.current_render_config = None

        # Camera state - Fixed for proper golf views
        self.camera_distance = 3.0
        self.camera_azimuth = 0.0  # Face-on view (looking at golfer from front)
        self.camera_elevation = 15.0  # Slightly elevated for better view
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Ground level tracking
        self.ground_level = 0.0

        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_pressed = False

        # Set focus policy for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def initializeGL(self) -> None:
        """Initialize OpenGL context."""
        try:
            # Create moderngl context
            self.ctx = mgl.create_context()

            # Initialize renderer
            self.renderer = OpenGLRenderer()
            self.renderer.initialize(self.ctx)

            # Set viewport
            self.renderer.set_viewport(self.width(), self.height())

            logger.info("OpenGL context initialized")
            logger.info("   Version: %s", self.ctx.info["GL_VERSION"])
            logger.info("   Vendor: %s", self.ctx.info["GL_VENDOR"])
            logger.info("   Renderer: %s", self.ctx.info["GL_RENDERER"])

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("OpenGL initialization failed: %s", e)
            traceback.print_exc()

    def resizeGL(self, w: int, h: int) -> None:
        """Handle OpenGL widget resize."""
        if self.renderer:
            self.renderer.set_viewport(w, h)

    def paintGL(self) -> None:
        """Render the OpenGL scene."""
        if not self.renderer or not self.current_frame_data:
            return

        try:
            # Calculate view and projection matrices
            view_matrix = self._calculate_view_matrix()
            proj_matrix = self._calculate_projection_matrix()
            view_position = self._calculate_view_position()

            # Pass ground level to renderer
            if hasattr(self.renderer, "ground_level"):
                self.renderer.ground_level = self.ground_level

            # Render frame
            self.renderer.render_frame(
                self.current_frame_data,
                {},  # Empty dynamics data for now
                self.current_render_config or RenderConfig(),
                view_matrix,
                proj_matrix,
                view_position,
            )

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Render error: %s", e)

    def _calculate_view_matrix(self) -> np.ndarray:
        """Calculate view matrix from camera parameters."""
        # Convert spherical coordinates to Cartesian
        x = (
            self.camera_distance
            * np.cos(np.radians(self.camera_elevation))
            * np.cos(np.radians(self.camera_azimuth))
        )
        y = self.camera_distance * np.sin(np.radians(self.camera_elevation))
        z = (
            self.camera_distance
            * np.cos(np.radians(self.camera_elevation))
            * np.sin(np.radians(self.camera_azimuth))
        )

        camera_pos = np.array([x, y, z], dtype=np.float32) + self.camera_target

        # Look-at matrix
        forward = self.camera_target - camera_pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward = forward / forward_norm
        else:
            forward = np.array(
                [0, 0, -1], dtype=np.float32
            )  # Default forward direction

        right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            right = np.array([1, 0, 0], dtype=np.float32)  # Default right direction

        up = np.cross(right, forward)

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = -camera_pos

        return view_matrix

    def _calculate_projection_matrix(self) -> np.ndarray:
        """Calculate projection matrix."""
        aspect = self.width() / max(self.height(), 1)
        fov = 45.0
        near = 0.1
        far = 100.0

        f = 1.0 / np.tan(np.radians(fov) / 2.0)

        proj_matrix = np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

        return proj_matrix

    def _calculate_view_position(self) -> np.ndarray:
        """Calculate view position."""
        x = (
            self.camera_distance
            * np.cos(np.radians(self.camera_elevation))
            * np.cos(np.radians(self.camera_azimuth))
        )
        y = self.camera_distance * np.sin(np.radians(self.camera_elevation))
        z = (
            self.camera_distance
            * np.cos(np.radians(self.camera_elevation))
            * np.sin(np.radians(self.camera_azimuth))
        )

        return np.array([x, y, z], dtype=np.float32) + self.camera_target

    def load_data_from_dataframes(
        self, dataframes: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Load data from pandas DataFrames."""
        try:
            baseq_df, ztcfq_df, deltaq_df = dataframes

            # Create frame processor with config
            config = RenderConfig()
            self.frame_processor = FrameProcessor(
                (baseq_df, ztcfq_df, deltaq_df), config
            )

            # Get first frame
            if len(self.frame_processor.time_vector) > 0:
                self.current_frame_data = self.frame_processor.get_frame_data(0)
                self.current_render_config = RenderConfig()

                # Frame camera to data
                self._frame_camera_to_data()

                # Trigger redraw
                self.update()

                logger.info("Loaded %s frames", len(self.frame_processor.time_vector))

        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Data loading failed: %s", e)
            traceback.print_exc()

    def update_frame(self, frame_data: FrameData, render_config: RenderConfig) -> None:
        """Update the current frame data and render config."""
        self.current_frame_data = frame_data
        self.current_render_config = render_config
        self.update()

    def _frame_camera_to_data(self) -> None:
        """Frame camera to show all data and set proper ground level."""
        if not self.current_frame_data:
            return

        # Calculate bounding box of data
        positions = [
            self.current_frame_data.left_wrist,
            self.current_frame_data.left_elbow,
            self.current_frame_data.left_shoulder,
            self.current_frame_data.right_wrist,
            self.current_frame_data.right_elbow,
            self.current_frame_data.right_shoulder,
            self.current_frame_data.hub,
            self.current_frame_data.butt,
            self.current_frame_data.clubhead,
        ]

        positions = [pos for pos in positions if np.isfinite(pos).all()]

        if not positions:
            return

        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        max_distance = np.max(np.linalg.norm(positions - center, axis=1))

        # Set ground level to lowest Z point in the data
        self.ground_level = np.min(positions[:, 2])

        # Update camera target to be centered horizontally but at ground level
        self.camera_target = np.array(
            [center[0], center[1], self.ground_level], dtype=np.float32
        )
        self.camera_distance = max_distance * 2.5

        logger.info(
            "Camera framed: center=%s, ground_level=%.3f, distance=%.2f",
            center,
            self.ground_level,
            self.camera_distance,
        )

    def set_face_on_view(self) -> None:
        """Set camera to face-on view (looking at golfer from front)."""
        self.camera_azimuth = 0.0
        self.camera_elevation = 15.0
        self.update()
        logger.info("Camera: Face-on view")

    def set_down_the_line_view(self) -> None:
        """Set camera to down-the-line view (90 deg from face-on)."""
        self.camera_azimuth = 90.0  # 90 deg from face-on, not 180 deg
        self.camera_elevation = 15.0
        self.update()
        logger.info("Camera: Down-the-line view")

    def set_behind_view(self) -> None:
        """Set camera to behind view (180 deg from face-on)."""
        self.camera_azimuth = 180.0
        self.camera_elevation = 15.0
        self.update()
        logger.info("Camera: Behind view")

    def set_above_view(self) -> None:
        """Set camera to overhead view."""
        self.camera_azimuth = 0.0
        self.camera_elevation = 80.0
        self.update()
        logger.info("Camera: Overhead view")

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events."""
        self.last_mouse_pos = event.pos()
        self.mouse_pressed = True

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release events."""
        self.mouse_pressed = False

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move events."""
        if not self.mouse_pressed or not self.last_mouse_pos:
            return

        delta = event.pos() - self.last_mouse_pos

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Rotate camera
            self.camera_azimuth += delta.x() * 0.5
            self.camera_elevation += delta.y() * 0.5
            self.camera_elevation = np.clip(self.camera_elevation, -89, 89)

        elif event.buttons() & Qt.MouseButton.RightButton:
            # Pan camera
            pan_speed = self.camera_distance * 0.001
            right = np.array(
                [
                    np.cos(np.radians(self.camera_azimuth - 90)),
                    0,
                    np.sin(np.radians(self.camera_azimuth - 90)),
                ],
                dtype=np.float32,
            )
            up = np.array([0, 1, 0], dtype=np.float32)

            self.camera_target += (right * delta.x() - up * delta.y()) * pan_speed

        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel events."""
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.camera_distance *= zoom_factor
        self.camera_distance = np.clip(self.camera_distance, 0.1, 50.0)
        self.update()

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        key = event.key()

        if key == Qt.Key.Key_1:
            self.set_face_on_view()
        elif key == Qt.Key.Key_2:
            self.set_down_the_line_view()
        elif key == Qt.Key.Key_3:
            self.set_behind_view()
        elif key == Qt.Key.Key_4:
            self.set_above_view()
        elif key == Qt.Key.Key_R:
            self._frame_camera_to_data()
            self.update()
        elif key == Qt.Key.Key_Space:
            # Toggle playback if parent has this functionality
            parent = self.parent()
            if parent and hasattr(parent, "toggle_playback"):
                parent.toggle_playback()
        else:
            super().keyPressEvent(event)
