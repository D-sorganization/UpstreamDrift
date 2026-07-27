from __future__ import annotations

import mujoco
import math
import numpy as np


class IKSolverMixin:
    model: mujoco.MjModel
    data: mujoco.MjData
    ik_damping: float
    ik_max_iterations: int
    ik_tolerance: float
    ik_step_size: float
    use_nullspace_posture: bool
    original_qpos: np.ndarray | None

    def _compute_ik_task_error(
        self,
        body_id: int,
        target_position: np.ndarray,
        target_quat: np.ndarray | None,
        maintain_orientation: bool,
    ) -> np.ndarray:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
        current_pos = self.data.xpos[body_id].copy()
        pos_error = target_position - current_pos

        if maintain_orientation and target_quat is not None:
            current_quat = self.data.xquat[body_id].copy()
            ori_error = self._orientation_error(current_quat, target_quat)
            return np.concatenate([pos_error, ori_error])
        return pos_error

    def _compute_ik_step(
        self,
        body_id: int,
        task_error: np.ndarray,
        task_dim: int,
        maintain_orientation: bool,
    ) -> np.ndarray | None:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

        J = np.vstack([jacp, jacr]) if maintain_orientation else jacp

        damping_matrix = self.ik_damping**2 * np.eye(task_dim)
        try:
            return J.T @ np.linalg.solve(J @ J.T + damping_matrix, task_error)
        except np.linalg.LinAlgError:
            return None

    def _apply_nullspace_posture(
        self,
        J_damped: np.ndarray,
        body_id: int,
        q: np.ndarray,
        maintain_orientation: bool,
    ) -> np.ndarray:
        if not (J_damped is not None):
            raise ValueError("J_damped must be provided")
        if not self.use_nullspace_posture or self.original_qpos is None:
            return J_damped

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        J = np.vstack([jacp, jacr]) if maintain_orientation else jacp

        J_pinv = np.linalg.pinv(J, rcond=self.ik_damping)
        nullspace_proj = np.eye(self.model.nv) - J_pinv @ J

        qpos_diff = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(
            self.model,
            qpos_diff,
            1.0,
            q,
            self.original_qpos,
        )

        nullspace_motion = nullspace_proj @ qpos_diff
        return J_damped + 0.05 * nullspace_motion

    def _solve_ik_for_body(
        self,
        body_id: int,
        target_position: np.ndarray,
        maintain_orientation: bool = False,
    ) -> bool:
        if not (body_id is not None):
            raise ValueError("body_id must be provided")
        q = self.data.qpos.copy()
        task_dim = 6 if maintain_orientation else 3
        target_quat = self.data.xquat[body_id].copy() if maintain_orientation else None

        for _iteration in range(self.ik_max_iterations):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)

            task_error = self._compute_ik_task_error(
                body_id,
                target_position,
                target_quat,
                maintain_orientation,
            )

            if math.sqrt(np.vdot(task_error, task_error)) < self.ik_tolerance:  # ⚡ Bolt: math.sqrt(np.vdot) is faster than np.linalg.norm for small 1D arrays
                return True

            J_damped = self._compute_ik_step(
                body_id,
                task_error,
                task_dim,
                maintain_orientation,
            )
            if J_damped is None:
                return False

            J_damped = self._apply_nullspace_posture(
                J_damped,
                body_id,
                q,
                maintain_orientation,
            )

            q_new = np.zeros_like(q)
            mujoco.mj_integratePos(
                self.model,
                q_new,
                J_damped * self.ik_step_size,
                1.0,
            )
            q = self._clamp_joint_limits(q_new)

        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return False

    def _orientation_error(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
    ) -> np.ndarray:
        if not (q_current is not None):
            raise ValueError("q_current must be provided")
        q_current_conj = np.array(
            [q_current[0], -q_current[1], -q_current[2], -q_current[3]],
        )

        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_current_conj

        q_error = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
        )

        return 2.0 * q_error[1:4]

    def _clamp_joint_limits(self, q: np.ndarray) -> np.ndarray:
        if not (q is not None):
            raise ValueError("q must be provided")
        q_clamped = q.copy()

        for i in range(min(self.model.njnt, len(q))):
            if self.model.jnt_limited[i]:
                q_min = self.model.jnt_range[i, 0]
                q_max = self.model.jnt_range[i, 1]
                qpos_addr = self.model.jnt_qposadr[i]

                if qpos_addr < len(q_clamped):
                    q_clamped[qpos_addr] = np.clip(q_clamped[qpos_addr], q_min, q_max)

        return q_clamped
