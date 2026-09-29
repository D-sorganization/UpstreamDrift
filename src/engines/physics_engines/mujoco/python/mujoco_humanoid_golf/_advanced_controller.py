from __future__ import annotations

import mujoco
import numpy as np

from ._control_types import ControlMode, HybridControlMask, ImpedanceParameters
from ._osc_mixin import OperationalSpaceControlMixin


class AdvancedController(OperationalSpaceControlMixin):
    """Advanced controller implementing multiple control strategies.

    This controller provides professional-grade control schemes used in
    industrial robotics and research applications.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.data = data

        # Control mode
        self.mode = ControlMode.TORQUE

        # Default impedance parameters (moderate stiffness/damping)
        self.impedance_params = ImpedanceParameters(
            stiffness=np.ones(model.nv) * 100.0,  # 100 N/m or Nm/rad
            damping=np.ones(model.nv) * 20.0,  # 20 Ns/m or Nms/rad
        )

        # Target for impedance/admittance control
        self.target_position: np.ndarray | None = None
        self.target_velocity: np.ndarray | None = None

        # For force control
        self.target_force: np.ndarray | None = None

        # Hybrid control mask (default: all position control)
        self.hybrid_mask = HybridControlMask(force_mask=np.zeros(model.nv, dtype=bool))

        # Gravity compensation flag
        self.enable_gravity_compensation = True

        # Find important body IDs
        self.club_head_id = self._find_body_id("club_head")

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern."""
        if name_pattern is None:
            raise ValueError("name_pattern must be provided")
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set control mode.

        Args:
            mode: Desired control mode
        """
        self.mode = mode

    def set_impedance_parameters(self, params: ImpedanceParameters) -> None:
        """Set impedance control parameters.

        Args:
            params: Impedance parameters
        """
        self.impedance_params = params

    def set_hybrid_mask(self, mask: HybridControlMask) -> None:
        """Set hybrid force-position control mask.

        Args:
            mask: Hybrid control mask
        """
        self.hybrid_mask = mask

    def compute_control(  # noqa: PLR0911 - Multiple return paths for different control modes
        self,
        target_position: np.ndarray | None = None,
        target_velocity: np.ndarray | None = None,
        target_force: np.ndarray | None = None,
        feedforward_torque: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute control torques based on current mode.

        Args:
            target_position: Desired position [nv] or task space [m]
            target_velocity: Desired velocity [nv] or task space [m]
            target_force: Desired force/torque [nv] or task space [m]
            feedforward_torque: Feedforward torque [nv]

        Returns:
            Control torques [nu]
        """
        if self.mode == ControlMode.TORQUE:
            return (
                feedforward_torque
                if feedforward_torque is not None
                else np.zeros(self.model.nu)
            )

        if self.mode == ControlMode.IMPEDANCE:
            return self._compute_impedance_control(target_position, target_velocity)

        if self.mode == ControlMode.ADMITTANCE:
            return self._compute_admittance_control(target_force)

        if self.mode == ControlMode.HYBRID:
            return self._compute_hybrid_control(
                target_position,
                target_velocity,
                target_force,
            )

        if self.mode == ControlMode.COMPUTED_TORQUE:
            return self._compute_computed_torque_control(
                target_position,
                target_velocity,
            )

        if self.mode == ControlMode.TASK_SPACE:
            return self._compute_task_space_control(target_position, target_velocity)

        return np.zeros(self.model.nu)

    def _compute_impedance_control(
        self,
        target_position: np.ndarray | None,
        target_velocity: np.ndarray | None,
    ) -> np.ndarray:
        """Compute impedance control torques.

        Impedance control creates a virtual spring-damper system:
        τ = K(q_d - q) + D(q̇_d - q̇) + g(q)

        Args:
            target_position: Desired position [nv]
            target_velocity: Desired velocity [nv]

        Returns:
            Control torques [nu]
        """
        if target_position is None:
            target_position = self.data.qpos.copy()
        if target_velocity is None:
            target_velocity = np.zeros(self.model.nv)

        # Get impedance matrices
        k_matrix, d_matrix, _m_matrix = self.impedance_params.as_matrices(self.model.nv)

        # Position error
        pos_error = target_position - self.data.qpos

        # Velocity error
        vel_error = target_velocity - self.data.qvel

        # Impedance control law
        tau = k_matrix @ pos_error + d_matrix @ vel_error

        # Add gravity compensation
        if self.enable_gravity_compensation:
            tau += self._compute_gravity_compensation()

        # Map to actuators (assuming 1-to-1 mapping for now)
        return np.asarray(tau[: self.model.nu])

    def _compute_admittance_control(
        self,
        target_force: np.ndarray | None,
    ) -> np.ndarray:
        """Compute admittance control torques.

        Admittance control modifies position based on force error:
        Δq̈ = M^{-1}(F_d - F)

        This is the dual of impedance control.

        Args:
            target_force: Desired force/torque [nv]

        Returns:
            Control torques [nu]
        """
        if target_force is None:
            target_force = np.zeros(self.model.nv)

        # Measured force (from constraint forces or sensors)
        measured_force = self.data.qfrc_constraint.copy()

        # Force error
        force_error = target_force - measured_force

        # Get impedance matrices
        _k_matrix, d_matrix, m_matrix = self.impedance_params.as_matrices(self.model.nv)

        # Admittance dynamics: compute desired acceleration
        m_inv = np.linalg.inv(m_matrix)
        desired_acceleration = m_inv @ force_error

        # Integrate to get desired velocity (simple Euler integration)
        dt = self.model.opt.timestep
        desired_velocity = self.data.qvel + desired_acceleration * dt

        # Use impedance control to track desired velocity
        tau = d_matrix @ (desired_velocity - self.data.qvel)

        # Add gravity compensation
        if self.enable_gravity_compensation:
            tau += self._compute_gravity_compensation()

        return tau[: self.model.nu]  # type: ignore[no-any-return]

    def _compute_hybrid_control(
        self,
        target_position: np.ndarray | None,
        target_velocity: np.ndarray | None,
        target_force: np.ndarray | None,
    ) -> np.ndarray:
        """Compute hybrid force-position control torques.

        Hybrid control combines position and force control:
        τ = S_p τ_p + S_f τ_f

        where S_p and S_f are selection matrices.

        Args:
            target_position: Desired position [nv]
            target_velocity: Desired velocity [nv]
            target_force: Desired force [nv]

        Returns:
            Control torques [nu]
        """
        if target_position is None:
            target_position = self.data.qpos.copy()
        if target_velocity is None:
            target_velocity = np.zeros(self.model.nv)
        if target_force is None:
            target_force = np.zeros(self.model.nv)

        # Get selection matrices
        s_p = self.hybrid_mask.get_position_selection_matrix()
        s_f = self.hybrid_mask.get_force_selection_matrix()

        # Position control component
        k_matrix, d_matrix, _m_matrix = self.impedance_params.as_matrices(self.model.nv)
        pos_error = target_position - self.data.qpos
        vel_error = target_velocity - self.data.qvel
        tau_position = k_matrix @ pos_error + d_matrix @ vel_error

        # Force control component
        measured_force = self.data.qfrc_constraint.copy()
        force_error = target_force - measured_force
        tau_force = force_error  # Simple force tracking

        # Combine using selection matrices
        tau = s_p @ tau_position + s_f @ tau_force

        # Add gravity compensation
        if self.enable_gravity_compensation:
            tau += self._compute_gravity_compensation()

        return tau[: self.model.nu]  # type: ignore[no-any-return]

    def _compute_computed_torque_control(
        self,
        target_position: np.ndarray | None,
        target_velocity: np.ndarray | None,
    ) -> np.ndarray:
        """Compute computed torque control (inverse dynamics control).

        This is a model-based feedforward control:
        τ = M(q)q̈_d + C(q,q̇)q̇ + g(q)

        where q̈_d = q̈_ref + K_d(q̇_d - q̇) + K_p(q_d - q)

        Args:
            target_position: Desired position [nv]
            target_velocity: Desired velocity [nv]

        Returns:
            Control torques [nu]
        """
        if target_position is None:
            target_position = self.data.qpos.copy()
        if target_velocity is None:
            target_velocity = np.zeros(self.model.nv)

        # Compute errors
        pos_error = target_position - self.data.qpos
        vel_error = target_velocity - self.data.qvel

        # PD gains
        k_p = self.impedance_params.stiffness
        k_d = self.impedance_params.damping

        # Desired acceleration (PD control)
        if k_p.ndim == 1:
            q_ddot_desired = k_p * pos_error + k_d * vel_error
        else:
            q_ddot_desired = k_p @ pos_error + k_d @ vel_error

        # Compute inverse dynamics
        # τ = M q̈ + C q̇ + g
        # In MuJoCo: qfrc_bias = C q̇ + g

        # Use efficient sparse multiplication for M @ q_ddot
        m_qddot = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, m_qddot, q_ddot_desired)

        tau = m_qddot + self.data.qfrc_bias

        return tau[: self.model.nu]  # type: ignore[no-any-return]

    def _compute_task_space_control(
        self,
        target_position: np.ndarray | None,
        target_velocity: np.ndarray | None,
    ) -> np.ndarray:
        """Compute task-space control with nullspace projection.

        This controls end-effector in Cartesian space:
        τ = J^T F + (I - J^T J_bar^T) τ_null

        where τ_null is a nullspace objective (e.g., joint centering).

        Args:
            target_position: Desired end-effector position [3 or 6]
            target_velocity: Desired end-effector velocity [3 or 6]

        Returns:
            Control torques [nu]
        """
        if self.club_head_id is None:
            # Fall back to joint-space control
            return self._compute_impedance_control(target_position, target_velocity)

        if target_position is None:
            target_position = self.data.xpos[self.club_head_id].copy()
        if target_velocity is None:
            target_velocity = np.zeros(3)

        # Compute Jacobian - fixed for MuJoCo 3.x API
        try:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.club_head_id)
        except TypeError:
            # Fallback to flat array approach for older MuJoCo versions
            jacp_flat = np.zeros(3 * self.model.nv)
            jacr_flat = np.zeros(3 * self.model.nv)
            mujoco.mj_jacBody(
                self.model, self.data, jacp_flat, jacr_flat, self.club_head_id
            )
            jacp = jacp_flat.reshape(3, self.model.nv)

        # Current end-effector state
        current_pos = self.data.xpos[self.club_head_id].copy()
        current_vel = jacp @ self.data.qvel

        # Task-space errors
        pos_error = target_position[:3] - current_pos
        vel_error = target_velocity[:3] - current_vel

        # Task-space PD control
        k_p = 100.0  # Cartesian stiffness
        k_d = 20.0  # Cartesian damping

        desired_force = k_p * pos_error + k_d * vel_error

        # Map to joint torques
        tau_task = jacp.T @ desired_force

        # Nullspace objective: joint centering
        joint_center = np.zeros(self.model.nv)  # Could use mid-range
        nullspace_error = joint_center - self.data.qpos

        # Nullspace projection
        j_pinv = np.linalg.pinv(jacp)
        nullspace_proj = np.eye(self.model.nv) - j_pinv @ jacp

        tau_null = nullspace_proj @ (10.0 * nullspace_error)  # Low gain

        # Combined control
        tau = tau_task + tau_null

        # Add gravity compensation
        if self.enable_gravity_compensation:
            tau += self._compute_gravity_compensation()

        return tau[: self.model.nu]  # type: ignore[no-any-return]

    def _compute_gravity_compensation(self) -> np.ndarray:
        """Compute gravity compensation torques.

        Returns:
            Gravity compensation torques [nv]
        """
        # In MuJoCo, gravity is included in qfrc_bias
        # We can extract it by computing with and without gravity
        # For now, use a simple approximation

        # Save current state
        qfrc_bias = self.data.qfrc_bias.copy()

        # Gravity compensation is the bias force without velocity terms
        # In quasi-static case: g(q) ≈ qfrc_bias
        return np.asarray(qfrc_bias.copy())
