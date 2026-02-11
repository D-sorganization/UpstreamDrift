"""MuJoCo sim widget rendering mixin.

Extracts frame rendering, overlay compositing, force/torque/induced/CF
vector drawing, manipulation overlays, swing plane overlays, and
frame/COM overlays from MuJoCoSimWidget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from PyQt6 import QtGui

from src.shared.python.logging_config import get_logger

# Lazy loading globals for OpenCV
CV2_LIB = None
INVALID_CV2 = False


def get_cv2() -> Any:
    """Lazy import of OpenCV to speed up initial load."""
    global CV2_LIB, INVALID_CV2  # noqa: PLW0603
    if CV2_LIB is None and not INVALID_CV2:
        try:
            import cv2

            CV2_LIB = cv2
        except ImportError:
            INVALID_CV2 = True
    return CV2_LIB


logger = get_logger(__name__)
MIN_CAMERA_DEPTH = 0.1
FORCE_VISUALIZATION_THRESHOLD = 1e-5


class SimRenderingMixin:
    """Mixin for MuJoCoSimWidget rendering and overlay compositing.

    Provides:
    - ``render`` / ``_render_once``: Frame rendering pipeline
    - ``_add_force_torque_overlays``: Force/torque/induced/CF screen overlays
    - ``_draw_torque_vectors`` / ``_draw_force_vectors``
    - ``_draw_induced_vectors`` / ``_draw_cf_vectors``
    - ``_add_manipulation_overlays``: Selected body / constraint overlays
    - ``_add_swing_plane_overlays``: Club trajectory / swing plane overlays
    - ``_add_frame_and_com_overlays``: Per-body frame/COM overlays
    - ``_world_to_screen``: World → screen-space projection
    - ``_update_background_colors`` / ``set_background_color``
    """

    def render(self: Any) -> None:  # type: ignore[override]
        """Render the scene immediately."""
        self._render_once()

    def _render_once(self: Any) -> None:  # noqa: PLR0912
        """Render one frame of the simulation."""
        if self.renderer is None or self.model is None or self.data is None:
            return

        if self.scene is not None:
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.scene_option,
                None,
                self.camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            self._update_background_colors()

        if hasattr(self, "camera") and self.camera is not None:
            self.renderer.update_scene(
                self.data,
                camera=self.camera,
                scene_option=self.scene_option,
            )
        else:
            self.renderer.update_scene(
                self.data,
                camera=self.camera_name,
                scene_option=self.scene_option,
            )

        rgb = self.renderer.render()

        # Add force/torque/accel overlays
        rgb = self._add_force_torque_overlays(rgb)

        if self.manipulator is not None and (
            self.show_selected_body or self.show_constraints
        ):
            rgb = self._add_manipulation_overlays(rgb)

        if self.show_club_trajectory or self.show_swing_plane:
            rgb = self._add_swing_plane_overlays(rgb)

        if self.visible_frames or self.visible_coms:
            rgb = self._add_frame_and_com_overlays(rgb)

        if self.meshcat_adapter:
            try:
                self.meshcat_adapter.update(self.data)
                self.meshcat_adapter.draw_vectors(
                    self.data,
                    self.show_force_vectors,
                    self.show_torque_vectors,
                    self.force_scale,
                    self.torque_scale,
                )

                if self.show_induced_vectors:
                    self.meshcat_adapter.draw_induced_vectors(
                        self.data,
                        self.latest_bio_data,
                        self.induced_vector_source,
                        self.torque_scale,
                    )
                else:
                    self.meshcat_adapter.draw_induced_vectors(self.data, None, "")

                if self.show_cf_vectors:
                    self.meshcat_adapter.draw_cf_vectors(
                        self.data,
                        self.latest_bio_data,
                        self.cf_vector_type,
                        self.torque_scale,
                    )
                else:
                    self.meshcat_adapter.draw_cf_vectors(self.data, None, "")

            except (RuntimeError, ValueError, AttributeError):
                pass

        if rgb is None or rgb.size == 0 or len(rgb.shape) < 3:
            return

        h, w, _ = rgb.shape
        image = QtGui.QImage(
            rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888
        )
        image = image.copy()
        pixmap = QtGui.QPixmap.fromImage(image)

        self.label.setPixmap(pixmap)

    def _update_background_colors(self: Any) -> None:
        if self.scene is not None:
            try:
                if hasattr(self.scene, "sky_rgba"):
                    self.scene.sky_rgba[:] = self.sky_color
                if hasattr(self.scene, "ground_rgba"):
                    self.scene.ground_rgba[:] = self.ground_color
            except (AttributeError, TypeError):
                pass

    def set_background_color(
        self: Any, sky_color: Any = None, ground_color: Any = None
    ) -> None:
        if sky_color is not None:
            self.sky_color = np.array(sky_color, dtype=np.float32)
        if ground_color is not None:
            self.ground_color = np.array(ground_color, dtype=np.float32)

        if self.scene is not None:
            self._update_background_colors()
            self._render_once()

    def _add_force_torque_overlays(self: Any, rgb: np.ndarray) -> np.ndarray:
        """Overlay torque/force/accel vectors using screen-space arrows."""
        if self.model is None or self.data is None:
            return rgb

        cv2 = get_cv2()
        if cv2 is None:
            logger.warning(
                "OpenCV not installed, cannot draw force/torque overlays."
            )
            return rgb

        img = rgb.copy()

        def draw_arrow(
            start: np.ndarray,
            end: np.ndarray,
            color: tuple[int, int, int],
        ) -> None:
            start_px = self._world_to_screen(start)
            end_px = self._world_to_screen(end)
            if start_px is None or end_px is None:
                return
            cv2.arrowedLine(
                img,
                start_px,
                end_px,
                color,
                thickness=2,
                tipLength=0.2,
            )

        if self.show_torque_vectors:
            self._draw_torque_vectors(draw_arrow)

        if self.show_force_vectors:
            self._draw_force_vectors(draw_arrow)

        if self.show_induced_vectors and self.latest_bio_data:
            self._draw_induced_vectors(draw_arrow)

        if self.show_cf_vectors and self.latest_bio_data:
            self._draw_cf_vectors(draw_arrow)

        return img

    def _draw_torque_vectors(self: Any, draw_arrow_func: Callable) -> None:
        if self.model is None or self.data is None:
            return

        selected_id = None
        if self.isolate_forces_visualization and self.manipulator:
            selected_id = self.manipulator.selected_body_id

        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if joint_id < 0 or joint_id >= self.model.njnt:
                continue
            body_id = self.model.jnt_bodyid[joint_id]
            if body_id < 0 or body_id >= self.model.nbody:
                continue

            if selected_id is not None and body_id != selected_id:
                continue

            torque = float(self.data.ctrl[i])
            if abs(torque) < 1e-6:
                continue
            joint_axis = self.data.xaxis[3 * joint_id : 3 * joint_id + 3]
            joint_pos = self.data.xpos[body_id].copy()
            arrow_length = abs(torque) * self.torque_scale
            arrow_dir = joint_axis * np.sign(torque) * arrow_length
            arrow_end = joint_pos + arrow_dir
            color = (255, 0, 0) if torque >= 0 else (0, 0, 255)
            draw_arrow_func(joint_pos, arrow_end, color)

    def _draw_force_vectors(self: Any, draw_arrow_func: Callable) -> None:
        if self.data is None or self.model is None:
            return

        selected_id = None
        if self.isolate_forces_visualization and self.manipulator:
            selected_id = self.manipulator.selected_body_id

        external_forces = self.data.cfrc_ext.reshape(-1, 6)
        for body_id in range(1, self.model.nbody):
            if selected_id is not None and body_id != selected_id:
                continue

            world_force = external_forces[body_id, 3:6]
            magnitude = float(np.linalg.norm(world_force))
            if magnitude < FORCE_VISUALIZATION_THRESHOLD:
                continue
            body_pos = self.data.xpos[body_id].copy()
            arrow_end = body_pos + world_force * self.force_scale
            draw_arrow_func(body_pos, arrow_end, (0, 255, 0))

        internal_forces = self.data.cfrc_int.reshape(-1, 6)
        for body_id in range(1, self.model.nbody):
            if selected_id is not None and body_id != selected_id:
                continue

            joint_force = internal_forces[body_id, 3:6]
            magnitude = float(np.linalg.norm(joint_force))
            if magnitude < FORCE_VISUALIZATION_THRESHOLD:
                continue
            body_pos = self.data.xpos[body_id].copy()
            arrow_end = body_pos + joint_force * self.force_scale
            draw_arrow_func(body_pos, arrow_end, (0, 255, 255))

    def _draw_induced_vectors(self: Any, draw_arrow_func: Callable) -> None:
        """Draw Induced Acceleration vectors (Magenta)."""
        if (
            self.model is None
            or self.data is None
            or self.latest_bio_data is None
        ):
            return

        selected_id = None
        if self.isolate_forces_visualization and self.manipulator:
            selected_id = self.manipulator.selected_body_id

        key = self.induced_vector_source
        if key not in ["gravity", "actuator"]:
            key = "selected_actuator"

        if key not in self.latest_bio_data.induced_accelerations:
            return

        accels = self.latest_bio_data.induced_accelerations[key]

        for j in range(self.model.njnt):
            body_id = self.model.jnt_bodyid[j]
            if selected_id is not None and body_id != selected_id:
                continue

            qvel_adr = self.model.jnt_dofadr[j]
            if qvel_adr >= len(accels):
                continue

            acc = accels[qvel_adr]
            if abs(acc) < 1e-3:
                continue

            joint_pos = self.data.xpos[body_id].copy()
            joint_axis = self.data.xaxis[3 * j : 3 * j + 3]

            arrow_len = acc * self.torque_scale * 0.5
            arrow_dir = joint_axis * arrow_len
            arrow_end = joint_pos + arrow_dir

            draw_arrow_func(joint_pos, arrow_end, (255, 0, 255))

    def _draw_cf_vectors(self: Any, draw_arrow_func: Callable) -> None:
        """Draw Counterfactual vectors (Yellow)."""
        if (
            self.model is None
            or self.data is None
            or self.latest_bio_data is None
        ):
            return
        if self.cf_vector_type not in self.latest_bio_data.counterfactuals:
            return

        values = self.latest_bio_data.counterfactuals[self.cf_vector_type]

        for j in range(self.model.njnt):
            qvel_adr = self.model.jnt_dofadr[j]
            if qvel_adr >= len(values):
                continue

            val = values[qvel_adr]
            if abs(val) < 1e-3:
                continue

            body_id = self.model.jnt_bodyid[j]
            joint_pos = self.data.xpos[body_id].copy()
            joint_axis = self.data.xaxis[3 * j : 3 * j + 3]

            arrow_len = val * self.torque_scale * 0.5
            arrow_dir = joint_axis * arrow_len
            arrow_end = joint_pos + arrow_dir

            draw_arrow_func(joint_pos, arrow_end, (0, 255, 255))

    def _add_manipulation_overlays(self: Any, rgb: np.ndarray) -> np.ndarray:
        cv2 = get_cv2()
        if cv2 is None:
            return rgb

        if self.model is None or self.data is None:
            return rgb

        img = rgb.copy()

        if (
            self.show_selected_body
            and self.manipulator is not None
            and self.manipulator.selected_body_id is not None
        ):
            body_pos = self.data.xpos[
                self.manipulator.selected_body_id
            ].copy()
            screen_pos = self._world_to_screen(body_pos)
            if screen_pos is not None:
                x, y = screen_pos
                cv2.circle(img, (x, y), 20, (0, 255, 255), 3)
                cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
                body_name = self.manipulator.get_body_name(
                    self.manipulator.selected_body_id,
                )
                cv2.putText(
                    img,
                    body_name,
                    (x + 25, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        if self.show_constraints and self.manipulator is not None:
            for body_id in self.manipulator.get_constrained_bodies():
                body_pos = self.data.xpos[body_id].copy()
                screen_pos = self._world_to_screen(body_pos)

                if screen_pos is not None:
                    x, y = screen_pos
                    cv2.rectangle(
                        img,
                        (x - 15, y - 15),
                        (x + 15, y + 15),
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        "FIXED",
                        (x + 20, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                    )

        return img

    def _world_to_screen(
        self: Any, world_pos: np.ndarray
    ) -> tuple[int, int] | None:
        cam_azimuth = np.deg2rad(self.camera.azimuth)
        cam_elevation = np.deg2rad(self.camera.elevation)
        cam_distance = self.camera.distance
        cam_lookat = self.camera.lookat.copy()

        forward = np.array(
            [
                np.cos(cam_elevation) * np.sin(cam_azimuth),
                np.cos(cam_elevation) * np.cos(cam_azimuth),
                np.sin(cam_elevation),
            ],
        )
        cam_pos = cam_lookat - forward * cam_distance

        up_world = np.array([0, 0, 1])
        right = np.cross(up_world, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)

        to_point = world_pos - cam_pos
        z = np.dot(to_point, forward)

        if z < MIN_CAMERA_DEPTH:
            return None

        fovy = 45.0
        aspect = self.frame_width / self.frame_height

        x_cam = np.dot(to_point, right)
        y_cam = np.dot(to_point, up)

        x_ndc = x_cam / (z * np.tan(np.deg2rad(fovy / 2)) * aspect)
        y_ndc = y_cam / (z * np.tan(np.deg2rad(fovy / 2)))

        x_screen = int((x_ndc + 1.0) * 0.5 * self.frame_width)
        y_screen = int((1.0 - y_ndc) * 0.5 * self.frame_height)

        if (
            0 <= x_screen < self.frame_width
            and 0 <= y_screen < self.frame_height
        ):
            return (x_screen, y_screen)

        return None

    def _add_swing_plane_overlays(
        self: Any, rgb: np.ndarray
    ) -> np.ndarray:
        """Overlay club trajectory and swing plane normal onto the pixel frame."""
        cv2 = get_cv2()
        if cv2 is None or self.model is None or self.data is None:
            return rgb

        img = rgb.copy()
        history = self.swing_plane_visualizer.trajectory_history

        if self.show_club_trajectory and len(history) >= 2:
            prev_px = None
            for pt in history:
                px = self._world_to_screen(np.asarray(pt))
                if px is not None:
                    cv2.circle(img, px, 2, (0, 255, 0), -1)
                    if prev_px is not None:
                        cv2.line(img, prev_px, px, (0, 255, 0), 1)
                    prev_px = px

        if (
            self.show_reference_trajectory
            and self.reference_trajectory is not None
            and len(self.reference_trajectory) >= 2
        ):
            prev_px = None
            for pt in self.reference_trajectory:
                px = self._world_to_screen(np.asarray(pt))
                if px is not None:
                    cv2.circle(img, px, 2, (0, 140, 255), -1)
                    if prev_px is not None:
                        cv2.line(img, prev_px, px, (0, 140, 255), 1)
                    prev_px = px

        if self.show_swing_plane:
            scene = self.swing_plane_visualizer.current_scene
            if scene.instantaneous_plane is not None:
                plane = scene.instantaneous_plane
                start_px = self._world_to_screen(plane.normal_arrow_start)
                end_px = self._world_to_screen(plane.normal_arrow_end)
                if start_px is not None and end_px is not None:
                    cv2.arrowedLine(
                        img,
                        start_px,
                        end_px,
                        (255, 136, 68),
                        2,
                        tipLength=0.3,
                    )

        return img

    def _add_frame_and_com_overlays(
        self: Any, rgb: np.ndarray
    ) -> np.ndarray:
        cv2 = get_cv2()
        if self.model is None or self.data is None or cv2 is None:
            return rgb

        img = rgb.copy()

        def draw_line(
            start_px: tuple[int, int],
            end_px: tuple[int, int],
            color: tuple[int, int, int],
            thickness: int = 2,
        ) -> None:
            cv2.line(img, start_px, end_px, color, thickness)

        axis_length = 0.2
        for body_id in self.visible_frames:
            pos = self.data.xpos[body_id].copy()
            rot = self.data.xmat[body_id].reshape(3, 3)

            origin = self._world_to_screen(pos)
            if origin is None:
                continue

            x_end = pos + rot[:, 0] * axis_length
            x_px = self._world_to_screen(x_end)
            if x_px:
                draw_line(origin, x_px, (255, 0, 0))

            y_end = pos + rot[:, 1] * axis_length
            y_px = self._world_to_screen(y_end)
            if y_px:
                draw_line(origin, y_px, (0, 255, 0))

            z_end = pos + rot[:, 2] * axis_length
            z_px = self._world_to_screen(z_end)
            if z_px:
                draw_line(origin, z_px, (0, 0, 255))

        for body_id in self.visible_coms:
            com_pos = self.data.xipos[body_id].copy()
            screen_pos = self._world_to_screen(com_pos)
            if screen_pos:
                cv2.circle(img, screen_pos, 5, (0, 255, 255), -1)
                cv2.circle(img, screen_pos, 7, (0, 0, 0), 1)

                cv2.putText(
                    img,
                    "COM",
                    (screen_pos[0] + 10, screen_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

        return img
