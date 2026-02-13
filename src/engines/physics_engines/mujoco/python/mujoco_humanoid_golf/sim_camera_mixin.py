"""MuJoCo sim widget camera and mouse interaction mixin.

Extracts camera manipulation (mouse drag, wheel zoom, camera presets),
context menu, and frame/COM toggle logic from MuJoCoSimWidget.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class SimCameraMixin:
    """Mixin for MuJoCoSimWidget camera and mouse interaction.

    Provides:
    - ``mousePressEvent`` / ``mouseMoveEvent`` / ``mouseReleaseEvent``
    - ``wheelEvent``
    - ``set_camera_azimuth`` / ``set_camera_elevation`` / etc.
    - ``reset_camera``
    - ``show_context_menu``
    - ``toggle_frame_visibility`` / ``toggle_com_visibility``
    """

    def mousePressEvent(self: Any, event: QtGui.QMouseEvent | None) -> None:
        """Handle mouse press for camera rotation, translation, or body selection."""
        if event is None:
            return
        modifiers = event.modifiers()
        button = event.button()
        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        if button == QtCore.Qt.MouseButton.MiddleButton or (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            x = int(event.position().x())
            y = int(event.position().y())
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "translate"
            return

        if (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers == QtCore.Qt.KeyboardModifier.NoModifier
            and self.manipulator is not None
            and self.model is not None
        ):
            body_id = self.manipulator.select_body(
                x,
                y,
                self.frame_width,
                self.frame_height,
                self.camera,
            )

            if body_id is not None:
                body_name = self.manipulator.get_body_name(body_id)
                logger.debug("Selected body via mouse: %s (id=%s)", body_name, body_id)
                self._render_once()
            else:
                self.last_mouse_pos = (x, y)
                self.is_dragging = True
                self.camera_mode = "rotate"

        if button == QtCore.Qt.MouseButton.RightButton:
            if self.manipulator is not None and self.model is not None:
                body_id = self.manipulator.select_body(
                    x,
                    y,
                    self.frame_width,
                    self.frame_height,
                    self.camera,
                )
                if body_id is not None:
                    self.show_context_menu(event.globalPosition().toPoint(), body_id)
                    return

            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "rotate"
            return

        if (
            button == QtCore.Qt.MouseButton.LeftButton
            and modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.last_mouse_pos = (x, y)
            self.is_dragging = True
            self.camera_mode = "rotate"
            return

        super().mousePressEvent(event)  # type: ignore[misc]

    def mouseMoveEvent(self: Any, event: QtGui.QMouseEvent | None) -> None:
        """Handle mouse drag for camera orbit, pan, or body manipulation."""
        if event is None:
            return
        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        if self.is_dragging and self.last_mouse_pos is not None:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            if self.camera_mode == "rotate":
                sensitivity = 0.5
                self.camera.azimuth -= dx * sensitivity
                self.camera.elevation = np.clip(
                    self.camera.elevation + dy * sensitivity,
                    -90.0,
                    90.0,
                )
                self._render_once()
            elif self.camera_mode == "translate":
                sensitivity = 0.01 * self.camera.distance
                az_rad = np.deg2rad(self.camera.azimuth)
                el_rad = np.deg2rad(self.camera.elevation)
                forward = np.array(
                    [
                        np.cos(el_rad) * np.sin(az_rad),
                        np.cos(el_rad) * np.cos(az_rad),
                        np.sin(el_rad),
                    ],
                )
                up_world = np.array([0, 0, 1])
                right = np.cross(up_world, forward)
                right = right / (np.linalg.norm(right) + 1e-8)
                up = np.cross(forward, right)

                self.camera.lookat[:] += (
                    -dx * sensitivity * right + dy * sensitivity * up
                )
                self._render_once()

            self.last_mouse_pos = (x, y)
            return

        if self.manipulator is not None and self.model is not None:
            if self.manipulator.selected_body_id is not None:
                success = self.manipulator.drag_to(
                    x,
                    y,
                    self.frame_width,
                    self.frame_height,
                    self.camera,
                )

                if success:
                    self._render_once()

        super().mouseMoveEvent(event)  # type: ignore[misc]

    def mouseReleaseEvent(self: Any, event: QtGui.QMouseEvent | None) -> None:
        """Handle mouse release to end dragging or body manipulation."""
        if event is None:
            return
        if self.is_dragging:
            self.is_dragging = False
            self.last_mouse_pos = None

        if (
            self.manipulator is not None
            and self.model is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.manipulator.selected_body_id is not None
        ):
            body_name = self.manipulator.get_body_name(
                self.manipulator.selected_body_id,
            )
            logger.debug("Released body via mouse: %s", body_name)
            self.manipulator.deselect_body()
            self._render_once()

        super().mouseReleaseEvent(event)  # type: ignore[misc]

    def wheelEvent(self: Any, event: QtGui.QWheelEvent | None) -> None:
        """Handle scroll wheel for camera zoom."""
        if event is None:
            return
        if self.camera is not None:
            delta = event.angleDelta().y()
            zoom_factor = 1.0 + (delta / 1200.0)
            self.camera.distance *= zoom_factor
            self.camera.distance = max(0.1, min(50.0, self.camera.distance))
            self._render_once()

        super().wheelEvent(event)  # type: ignore[misc]

    def set_camera_azimuth(self: Any, azimuth: float) -> None:
        """Set the camera azimuth angle in degrees."""
        if self.camera is not None:
            self.camera.azimuth = azimuth
            self._render_once()

    def set_camera_elevation(self: Any, elevation: float) -> None:
        """Set the camera elevation angle in degrees."""
        if self.camera is not None:
            self.camera.elevation = np.clip(elevation, -90.0, 90.0)
            self._render_once()

    def set_camera_distance(self: Any, distance: float) -> None:
        """Set the camera orbit distance in meters."""
        if self.camera is not None:
            self.camera.distance = np.clip(distance, 0.1, 50.0)
            self._render_once()

    def set_camera_lookat(self: Any, x: float, y: float, z: float) -> None:
        """Set the camera look-at target position."""
        if self.camera is not None:
            self.camera.lookat[:] = [x, y, z]
            self._render_once()

    def reset_camera(self: Any) -> None:
        """Restore camera to the default viewpoint."""
        if self.camera is not None:
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.azimuth = 90.0
            self.camera.elevation = -20.0
            self.camera.distance = 3.0
            self.camera.lookat[:] = [0, 0, 1]
            self._render_once()

    def show_context_menu(self: Any, global_pos: QtCore.QPoint, body_id: int) -> None:
        """Display a right-click context menu for a selected body."""
        if self.manipulator is None:
            return

        body_name = self.manipulator.get_body_name(body_id)
        menu = QtWidgets.QMenu(self)
        menu.setTitle(f"Body: {body_name}")

        header = menu.addAction(f"Selected: {body_name}")
        if header is not None:
            header.setEnabled(False)
        menu.addSeparator()

        action_frame = menu.addAction("Show Coordinate System")
        if action_frame is not None:
            action_frame.setCheckable(True)
            action_frame.setChecked(body_id in self.visible_frames)
            action_frame.triggered.connect(
                lambda: self.toggle_frame_visibility(body_id)
            )

        action_com = menu.addAction("Show Center of Mass")
        if action_com is not None:
            action_com.setCheckable(True)
            action_com.setChecked(body_id in self.visible_coms)
            action_com.triggered.connect(lambda: self.toggle_com_visibility(body_id))

        menu.exec(global_pos)

    def toggle_frame_visibility(self: Any, body_id: int) -> None:
        """Toggle coordinate frame overlay for a body."""
        if body_id in self.visible_frames:
            self.visible_frames.remove(body_id)
        else:
            self.visible_frames.add(body_id)
        self._render_once()

    def toggle_com_visibility(self: Any, body_id: int) -> None:
        """Toggle center-of-mass overlay for a body."""
        if body_id in self.visible_coms:
            self.visible_coms.remove(body_id)
        else:
            self.visible_coms.add(body_id)
        self._render_once()
