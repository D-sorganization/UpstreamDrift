from __future__ import annotations

import mujoco
import numpy as np

from ._constraint_manager import BodyConstraint, ConstraintManagerMixin, ConstraintType
from ._ik_solver import IKSolverMixin
from ._mouse_picking import MousePickingRay
from ._pose_library import PoseLibraryMixin, StoredPose

__all__ = [
    "BodyConstraint",
    "ConstraintType",
    "InteractiveManipulator",
    "MousePickingRay",
    "StoredPose",
]


class InteractiveManipulator(IKSolverMixin, ConstraintManagerMixin, PoseLibraryMixin):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not (model is not None):
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

        self.picker = MousePickingRay(model, data)

        self.ik_damping = 0.05
        self.ik_max_iterations = 20
        self.ik_tolerance = 1e-3
        self.ik_step_size = 0.3

        self.selected_body_id: int | None = None
        self.drag_offset: np.ndarray | None = None
        self.original_qpos: np.ndarray | None = None

        self.constraints: dict[int, BodyConstraint] = {}

        self.pose_library: dict[str, StoredPose] = {}

        self.drag_enabled = True
        self.maintain_orientation = False
        self.use_nullspace_posture = True

    def enable_drag(self, enabled: bool) -> None:
        self.drag_enabled = enabled

    def select_body(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
    ) -> int | None:
        if not (x is not None):
            raise ValueError("x must be provided")
        if not self.drag_enabled:
            return None

        result = self.picker.pick_body(x, y, width, height, camera)

        if result is not None:
            body_id, intersection_point, _ = result

            self.selected_body_id = body_id
            self.original_qpos = self.data.qpos.copy()

            body_pos = self.data.xpos[body_id].copy()
            self.drag_offset = intersection_point - body_pos

            return body_id

        return None

    def deselect_body(self) -> None:
        self.selected_body_id = None
        self.drag_offset = None
        self.original_qpos = None

    def drag_to(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        camera: mujoco.MjvCamera,
        plane_normal: np.ndarray | None = None,
    ) -> bool:
        if not (x is not None):
            raise ValueError("x must be provided")
        if self.selected_body_id is None or not self.drag_enabled:
            return False

        ray_origin, ray_dir = self.picker.screen_to_ray(x, y, width, height, camera)

        if plane_normal is None:
            cam_azimuth = np.deg2rad(camera.azimuth)
            cam_elevation = np.deg2rad(camera.elevation)
            plane_normal = np.array(
                [
                    np.cos(cam_elevation) * np.sin(cam_azimuth),
                    np.cos(cam_elevation) * np.cos(cam_azimuth),
                    np.sin(cam_elevation),
                ],
            )

        body_pos = self.data.xpos[self.selected_body_id].copy()
        plane_point = body_pos

        denom = np.dot(ray_dir, plane_normal)
        if abs(denom) < 1e-6:
            return False

        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        if t < 0:
            return False

        target_point = ray_origin + ray_dir * t

        if self.drag_offset is not None:
            target_point -= self.drag_offset

        mocap_id = self.model.body_mocapid[self.selected_body_id]
        if mocap_id != -1:
            self.data.mocap_pos[mocap_id] = target_point
            mujoco.mj_forward(self.model, self.data)
            return True

        success = self._solve_ik_for_body(
            self.selected_body_id,
            target_point,
            maintain_orientation=self.maintain_orientation,
        )

        if success:
            self._apply_constraints()

        return success

    def get_body_name(self, body_id: int) -> str:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name is not None:
            return str(name)
        return f"body_{body_id}"

    def find_body_by_name(self, name: str) -> int | None:
        if not (name is not None):
            raise ValueError("name must be provided")
        for body_id in range(self.model.nbody):
            body_name = self.get_body_name(body_id)
            if name.lower() in body_name.lower():
                return body_id
        return None

    def reset_to_original_pose(self) -> None:
        if self.original_qpos is not None:
            self.data.qpos[:] = self.original_qpos.copy()
            mujoco.mj_forward(self.model, self.data)
