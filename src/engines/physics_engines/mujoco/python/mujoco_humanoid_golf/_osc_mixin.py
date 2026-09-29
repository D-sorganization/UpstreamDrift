from __future__ import annotations

import mujoco
import numpy as np


class OperationalSpaceControlMixin:
    """Mixin providing operational space control (OSC) for AdvancedController."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def compute_operational_space_control(
        self,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        target_acceleration: np.ndarray,
        body_id: int,
    ) -> np.ndarray:
        """Compute operational space control (OSC).

        OSC is an advanced task-space controller that accounts for
        the configuration-dependent inertia:

        F = Λ(q)(ẍ_d + K_d ė + K_p e) + μ(q,q̇) + p(q)
        τ = J^T F + N^T τ_posture

        where Λ is task-space inertia.

        Args:
            target_position: Desired end-effector position [3]
            target_velocity: Desired end-effector velocity [3]
            target_acceleration: Desired end-effector acceleration [3]
            body_id: Body ID for end-effector

        Returns:
            Control torques [nu]
        """
        # Compute Jacobian
        # MuJoCo 3.3+ may require reshaped arrays - try both approaches
        if not (target_position is not None):
            raise ValueError("target_position must be provided")
        try:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        except TypeError:
            # Fallback to flat array approach
            jacp_flat = np.zeros(3 * self.model.nv)
            jacr_flat = np.zeros(3 * self.model.nv)
            mujoco.mj_jacBody(self.model, self.data, jacp_flat, jacr_flat, body_id)
            jacp = jacp_flat.reshape(3, self.model.nv)

        # Current state
        current_pos = self.data.xpos[body_id].copy()
        current_vel = jacp @ self.data.qvel

        # Errors
        pos_error = target_position - current_pos
        vel_error = target_velocity - current_vel

        # Compute task-space inertia matrix
        # Λ = (J M^{-1} J^T)^{-1}

        # Calculate J M^{-1} efficiently using sparse factorization
        # mj_solveM solves M x = y. We pass J as y (shape 3 x nv)
        # Output will be J M^{-1}
        jac_m_inv = np.zeros((3, self.model.nv))
        mujoco.mj_solveM(self.model, self.data, jac_m_inv, jacp)

        lambda_matrix = np.linalg.inv(jac_m_inv @ jacp.T)

        # Compute dynamically consistent pseudoinverse
        # J_bar = M^{-1} J^T Λ
        # jac_m_inv.T is M^{-1} J^T (since M is symmetric)
        j_bar = jac_m_inv.T @ lambda_matrix

        # Task-space control law
        k_p = 100.0
        k_d = 20.0

        desired_acceleration = target_acceleration + k_d * vel_error + k_p * pos_error
        f_task = lambda_matrix @ desired_acceleration

        # Coriolis and gravity compensation in task space
        # μ = Λ J M^{-1} h - Λ J̇ q̇
        # For now, simplified version

        # Map to joint torques
        tau_task = jacp.T @ f_task

        # Nullspace control
        nullspace_proj = np.eye(self.model.nv) - jacp.T @ j_bar.T
        tau_null = nullspace_proj @ (-10.0 * self.data.qvel)  # Damping

        tau = tau_task + tau_null + self.data.qfrc_bias

        return tau[: self.model.nu]  # type: ignore[no-any-return]
