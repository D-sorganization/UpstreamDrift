"""Advanced control schemes for robotics applications.

This module implements state-of-the-art control strategies including:
- Impedance control (position-based)
- Admittance control (force-based)
- Hybrid force-position control
- Computed torque control (inverse dynamics)
- Task-space control with nullspace projection
- Operational space control
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np


class ControlMode(Enum):
    """Control mode enumeration."""

    TORQUE = "torque"  # Direct torque control
    IMPEDANCE = "impedance"  # Impedance control
    ADMITTANCE = "admittance"  # Admittance control
    HYBRID = "hybrid"  # Hybrid force-position
    COMPUTED_TORQUE = "computed_torque"  # Computed torque
    TASK_SPACE = "task_space"  # Task-space control


@dataclass
class ImpedanceParameters:
    """Parameters for impedance control."""

    stiffness: np.ndarray  # Stiffness matrix K [n x n] or vector [n]
    damping: np.ndarray  # Damping matrix D [n x n] or vector [n]
    inertia: np.ndarray | None = None  # Inertia matrix M [n x n] (optional)

    def as_matrices(self, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to full matrices.

        Args:
            dim: Dimension of control space

        Returns:
            Tuple of (K_matrix, D_matrix, M_matrix)
        """
        # Stiffness
        k_matrix = (
            np.diag(self.stiffness) if self.stiffness.ndim == 1 else self.stiffness
        )

        # Damping
        d_matrix = np.diag(self.damping) if self.damping.ndim == 1 else self.damping

        # Inertia
        if self.inertia is None:
            m_matrix = np.eye(dim)
        elif self.inertia.ndim == 1:
            m_matrix = np.diag(self.inertia)
        else:
            m_matrix = self.inertia

        return k_matrix, d_matrix, m_matrix


@dataclass
class HybridControlMask:
    """Mask for hybrid force-position control.

    For each DOF: True = force control, False = position control
    """

    force_mask: np.ndarray  # Boolean mask [n]

    def get_position_mask(self) -> np.ndarray:
        """Get complementary position control mask."""
        return ~self.force_mask

    def get_force_selection_matrix(self) -> np.ndarray:
        """Get force selection matrix S_f."""
        return np.diag(self.force_mask.astype(float))

    def get_position_selection_matrix(self) -> np.ndarray:
        """Get position selection matrix S_p."""
        return np.diag(self.get_position_mask().astype(float))


class AdvancedController:
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
        return qfrc_bias.copy()

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


class TrajectoryGenerator:
    """Generate smooth trajectories for control.

    Useful for generating reference trajectories for controllers.
    """

    @staticmethod
    def minimum_jerk_trajectory(
        start: np.ndarray,
        goal: np.ndarray,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate minimum jerk trajectory.

        Minimum jerk trajectories are smooth and human-like.

        Args:
            start: Starting position [n]
            goal: Goal position [n]
            duration: Trajectory duration [s]
            dt: Time step [s]

        Returns:
            Tuple of (positions, velocities, accelerations)
            Each is [num_steps x n]
        """
        num_steps = int(duration / dt)
        t = np.linspace(0, duration, num_steps)

        # Minimum jerk polynomial
        # s(t) = a_0 + a_1 t + ... + a_5 t^5
        # with boundary conditions:
        # s(0) = 0, ṡ(0) = 0, s̈(0) = 0
        # s(T) = 1, ṡ(T) = 0, s̈(T) = 0

        tau = t / duration  # Normalized time [0, 1]
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2

        # Interpolate
        positions = (
            start[np.newaxis, :] + (goal - start)[np.newaxis, :] * s[:, np.newaxis]
        )
        velocities = (goal - start)[np.newaxis, :] * s_dot[:, np.newaxis]
        accelerations = (goal - start)[np.newaxis, :] * s_ddot[:, np.newaxis]

        return positions, velocities, accelerations

    @staticmethod
    def quintic_spline(
        waypoints: np.ndarray,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate quintic spline through waypoints.

        Args:
            waypoints: Waypoints [num_waypoints x n]
            duration: Total trajectory duration [s]
            dt: Time step [s]

        Returns:
            Tuple of (positions, velocities, accelerations)
        """
        # Simplified: use minimum jerk between consecutive waypoints
        all_positions = []
        all_velocities = []
        all_accelerations = []

        num_segments = len(waypoints) - 1
        segment_duration = duration / num_segments

        for i in range(num_segments):
            pos, vel, acc = TrajectoryGenerator.minimum_jerk_trajectory(
                waypoints[i],
                waypoints[i + 1],
                segment_duration,
                dt,
            )

            all_positions.append(pos)
            all_velocities.append(vel)
            all_accelerations.append(acc)

        positions = np.vstack(all_positions)
        velocities = np.vstack(all_velocities)
        accelerations = np.vstack(all_accelerations)

        return positions, velocities, accelerations
