# mypy: disable-error-code="no-redef,var-annotated,assignment"
"""OpenGL widget for the legacy golf swing visualizer."""

from __future__ import annotations

import logging
import time

import moderngl as mgl
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

try:
    from .golf_visualizer_data import DataProcessor
    from .golf_visualizer_models import RenderConfig
    from .golf_visualizer_renderer import OpenGLRenderer
except ImportError:
    from golf_visualizer_data import DataProcessor
    from golf_visualizer_models import RenderConfig
    from golf_visualizer_renderer import OpenGLRenderer

logger = logging.getLogger(__name__)


class ModernGolfVisualizerWidget(QOpenGLWidget):
    """Modern OpenGL widget for golf swing visualization"""

    def __init__(self) -> None:
        super().__init__()
        self.renderer = OpenGLRenderer()
        self.data_processor = DataProcessor()
        self.datasets = None
        self.current_frame = 0
        self.num_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.camera_distance = 3.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 20.0
        self.camera_target = np.array([0, 0, 0], dtype=np.float32)
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5
        self.render_config = RenderConfig(
            show_forces={"BASEQ": True, "ZTCFQ": True, "DELTAQ": True},
            show_torques={"BASEQ": True, "ZTCFQ": True, "DELTAQ": True},
            show_body_segments={
                "left_forearm": True,
                "left_upper_arm": True,
                "right_forearm": True,
                "right_upper_arm": True,
                "left_shoulder_neck": True,
                "right_shoulder_neck": True,
            },
        )
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.next_frame)
        self.frame_times = []
        self.fps = 0.0

    def initializeGL(self) -> None:
        """Initialize OpenGL context"""
        self.ctx = mgl.create_context()
        self.renderer.initialize(self.ctx)
        self.ctx.viewport = (0, 0, self.width(), self.height())
        logger.info("OpenGL initialized successfully")
        logger.info(f"   OpenGL Version: {self.ctx.info['GL_VERSION']}")
        logger.info(f"   Vendor: {self.ctx.info['GL_VENDOR']}")
        logger.info(f"   Renderer: {self.ctx.info['GL_RENDERER']}")

    def paintGL(self) -> None:
        """Render the current frame"""
        start_time = time.time()
        if self.datasets is None or self.num_frames == 0:
            self.ctx.clear(0.1, 0.2, 0.3)
            return
        frame_data = self.data_processor.extract_frame_data(
            self.current_frame, self.datasets
        )  # noqa: E501
        view_matrix = self._calculate_view_matrix()
        proj_matrix = self._calculate_projection_matrix()
        self.renderer.render_frame(
            frame_data, self.render_config, view_matrix, proj_matrix
        )  # noqa: E501
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self.fps = (
            len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        )  # noqa: E501

    def resizeGL(self, width, height) -> None:
        """Handle window resize"""
        self.ctx.viewport = (0, 0, width, height)

    def load_data(self, baseq_file: str, ztcfq_file: str, delta_file: str) -> None:
        """Load golf swing data"""
        try:
            datasets = self.data_processor.load_matlab_data(
                baseq_file, ztcfq_file, delta_file
            )  # noqa: E501
            self.datasets = {
                "BASEQ": datasets[0],
                "ZTCFQ": datasets[1],
                "DELTAQ": datasets[2],
            }
            self.num_frames = len(datasets[0])
            self.current_frame = 0
            logger.info(f"Data loaded: {self.num_frames} frames")
            self.update()
        except (RuntimeError, ValueError, OSError) as e:
            logger.info(f"Failed to load data: {e}")

    def play_animation(self) -> None:
        """Start animation playback"""
        if not self.is_playing and self.num_frames > 0:
            self.is_playing = True
            interval = int(33 / self.playback_speed)
            self.animation_timer.start(interval)

    def pause_animation(self) -> None:
        """Pause animation playback"""
        self.is_playing = False
        self.animation_timer.stop()

    def next_frame(self) -> None:
        """Advance to next frame"""
        if self.num_frames > 0:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.update()

    def set_frame(self, frame_idx: int) -> None:
        """Jump to specific frame"""
        if 0 <= frame_idx < self.num_frames:
            self.current_frame = frame_idx
            self.update()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for camera control"""
        self.last_mouse_pos = (event.x(), event.y())

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse movement for camera control"""
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos[0]
            dy = event.y() - self.last_mouse_pos[1]
            self.camera_azimuth += dx * self.mouse_sensitivity
            self.camera_elevation = np.clip(
                self.camera_elevation - dy * self.mouse_sensitivity, -89, 89
            )
            self.last_mouse_pos = (event.x(), event.y())
            self.update()

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for camera zoom"""
        if not (event is not None):
            raise ValueError("event must be provided")
        delta = event.angleDelta().y() / 120
        self.camera_distance = np.clip(self.camera_distance - delta * 0.2, 0.5, 10.0)
        self.update()

    def _calculate_view_matrix(self) -> np.ndarray:
        """Calculate camera view matrix"""
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        self.camera_distance * np.sin(elevation_rad)
        self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        view_matrix = np.eye(4, dtype=np.float32)
        return view_matrix

    def _calculate_projection_matrix(self) -> np.ndarray:
        """Calculate perspective projection matrix"""
        proj_matrix = np.eye(4, dtype=np.float32)
        return proj_matrix
