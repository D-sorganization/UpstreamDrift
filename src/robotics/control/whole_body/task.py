"""Task definitions for whole-body control.

This module provides task descriptors that define objectives
for the whole-body controller optimization.

Design by Contract:
    Tasks must have consistent dimensions.
    Jacobians must be finite and have correct shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class TaskType(Enum):
    """Type of task constraint."""

    EQUALITY = auto()  # A @ x = b (hard equality)
    INEQUALITY = auto()  # lb <= A @ x <= ub
    SOFT = auto()  # Minimize ||A @ x - b||^2_W (soft objective)


@dataclass
class Task:
    """Task descriptor for whole-body control.

    A task defines a control objective as:
        - Equality: J @ ddq = x_ddot_des
        - Inequality: lb <= J @ ddq <= ub
        - Soft: minimize ||J @ ddq - x_ddot_des||^2_W

    Attributes:
        name: Human-readable task name.
        task_type: Type of constraint (equality, inequality, soft).
        priority: Priority level (0 = highest).
        jacobian: Task Jacobian (task_dim, n_v).
        target: Target task-space acceleration.
        weight: Diagonal weight matrix for soft tasks.
        lower_bound: Lower bound for inequality tasks.
        upper_bound: Upper bound for inequality tasks.
        gain_p: Proportional gain for error feedback.
        gain_d: Derivative gain for error feedback.
    """

    name: str
    task_type: TaskType
    priority: int
    jacobian: NDArray[np.float64]
    target: NDArray[np.float64]
    weight: NDArray[np.float64] | None = None
    lower_bound: NDArray[np.float64] | None = None
    upper_bound: NDArray[np.float64] | None = None
    gain_p: float = 100.0
    gain_d: float = 20.0
    _task_dim: int = field(init=False, repr=False)
    _config_dim: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and initialize task dimensions."""
        self.jacobian = np.asarray(self.jacobian, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)

        if self.jacobian.ndim != 2:
            raise ValueError(f"Jacobian must be 2D, got {self.jacobian.ndim}D")

        self._task_dim = self.jacobian.shape[0]
        self._config_dim = self.jacobian.shape[1]

        if self.target.shape != (self._task_dim,):
            raise ValueError(
                f"Target shape {self.target.shape} doesn't match "
                f"Jacobian rows {self._task_dim}"
            )

        # Validate weight if provided
        if self.weight is not None:
            self.weight = np.asarray(self.weight, dtype=np.float64)
            if self.weight.shape != (self._task_dim,):
                raise ValueError(
                    f"Weight shape {self.weight.shape} doesn't match "
                    f"task dimension {self._task_dim}"
                )

        # Validate bounds for inequality tasks
        if (
            self.task_type == TaskType.INEQUALITY
            and self.lower_bound is None
            and self.upper_bound is None
        ):
            raise ValueError("Inequality task must have at least one bound")

        # Validate finite values
        if not np.all(np.isfinite(self.jacobian)):
            raise ValueError("Jacobian contains non-finite values")
        if not np.all(np.isfinite(self.target)):
            raise ValueError("Target contains non-finite values")

    @property
    def task_dim(self) -> int:
        """Get task space dimension."""
        return self._task_dim

    @property
    def config_dim(self) -> int:
        """Get configuration space dimension."""
        return self._config_dim

    def get_weight_matrix(self) -> NDArray[np.float64]:
        """Get diagonal weight matrix.

        Returns:
            Diagonal weight matrix (task_dim, task_dim).
        """
        if self.weight is not None:
            return np.diag(self.weight)
        return np.eye(self._task_dim)

    def compute_error_feedback(
        self,
        position_error: NDArray[np.float64],
        velocity_error: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute desired acceleration from PD error feedback.

        Args:
            position_error: Position error (task_dim,).
            velocity_error: Velocity error (task_dim,).

        Returns:
            Desired task-space acceleration for error correction.
        """
        return self.target + self.gain_p * position_error + self.gain_d * velocity_error


def create_com_task(
    jacobian_com: NDArray[np.float64],
    com_current: NDArray[np.float64],
    com_target: NDArray[np.float64],
    com_velocity: NDArray[np.float64],
    com_velocity_target: NDArray[np.float64] | None = None,
    weight: float = 1.0,
    priority: int = 2,
    gain_p: float = 100.0,
    gain_d: float = 20.0,
) -> Task:
    """Create a center-of-mass tracking task.

    Args:
        jacobian_com: CoM Jacobian (3, n_v).
        com_current: Current CoM position (3,).
        com_target: Target CoM position (3,).
        com_velocity: Current CoM velocity (3,).
        com_velocity_target: Target CoM velocity (3,). Zero if None.
        weight: Task weight.
        priority: Task priority.
        gain_p: Proportional gain.
        gain_d: Derivative gain.

    Returns:
        Task configured for CoM tracking.
    """
    com_current = np.asarray(com_current, dtype=np.float64)
    com_target = np.asarray(com_target, dtype=np.float64)
    com_velocity = np.asarray(com_velocity, dtype=np.float64)

    if com_velocity_target is None:
        com_velocity_target = np.zeros(3)
    else:
        com_velocity_target = np.asarray(com_velocity_target, dtype=np.float64)

    # Compute desired CoM acceleration with PD feedback
    error_p = com_target - com_current
    error_v = com_velocity_target - com_velocity

    target_accel = gain_p * error_p + gain_d * error_v

    return Task(
        name="com_tracking",
        task_type=TaskType.SOFT,
        priority=priority,
        jacobian=jacobian_com,
        target=target_accel,
        weight=np.full(3, weight),
        gain_p=gain_p,
        gain_d=gain_d,
    )


def create_posture_task(
    n_v: int,
    q_current: NDArray[np.float64],
    q_target: NDArray[np.float64],
    v_current: NDArray[np.float64],
    weight: float = 0.1,
    priority: int = 4,
    gain_p: float = 50.0,
    gain_d: float = 10.0,
    mask: NDArray[np.bool_] | None = None,
) -> Task:
    """Create a posture regularization task.

    Args:
        n_v: Number of velocity coordinates.
        q_current: Current joint positions.
        q_target: Target joint positions.
        v_current: Current joint velocities.
        weight: Task weight.
        priority: Task priority.
        gain_p: Proportional gain.
        gain_d: Derivative gain.
        mask: Boolean mask for which joints to include.

    Returns:
        Task configured for posture regularization.
    """
    q_current = np.asarray(q_current, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)
    v_current = np.asarray(v_current, dtype=np.float64)

    if mask is None:
        mask = np.ones(n_v, dtype=bool)

    n_active = int(np.sum(mask))

    # Identity Jacobian for joint space
    jacobian = np.eye(n_v)[mask]

    # Handle quaternion joints (n_q > n_v case)
    q_error = (
        q_target[:n_v] - q_current[:n_v]
        if len(q_current) > n_v
        else q_target - q_current
    )
    v_target = np.zeros(n_v)

    error_p = q_error[mask]
    error_v = (v_target - v_current)[mask]

    target_accel = gain_p * error_p + gain_d * error_v

    return Task(
        name="posture",
        task_type=TaskType.SOFT,
        priority=priority,
        jacobian=jacobian,
        target=target_accel,
        weight=np.full(n_active, weight),
        gain_p=gain_p,
        gain_d=gain_d,
    )


def create_ee_task(
    jacobian_ee: NDArray[np.float64],
    ee_current: NDArray[np.float64],
    ee_target: NDArray[np.float64],
    ee_velocity: NDArray[np.float64],
    ee_velocity_target: NDArray[np.float64] | None = None,
    weight: float = 1.0,
    priority: int = 3,
    gain_p: float = 100.0,
    gain_d: float = 20.0,
    position_only: bool = False,
) -> Task:
    """Create an end-effector tracking task.

    Args:
        jacobian_ee: End-effector Jacobian (6, n_v) or (3, n_v).
        ee_current: Current EE pose (position (3,) or pose (7,)).
        ee_target: Target EE pose.
        ee_velocity: Current EE velocity (twist (6,) or linear (3,)).
        ee_velocity_target: Target EE velocity. Zero if None.
        weight: Task weight.
        priority: Task priority.
        gain_p: Proportional gain.
        gain_d: Derivative gain.
        position_only: Use only position, ignore orientation.

    Returns:
        Task configured for end-effector tracking.
    """
    jacobian_ee = np.asarray(jacobian_ee, dtype=np.float64)
    ee_current = np.asarray(ee_current, dtype=np.float64)
    ee_target = np.asarray(ee_target, dtype=np.float64)
    ee_velocity = np.asarray(ee_velocity, dtype=np.float64)

    if position_only:
        # Use only position components
        if jacobian_ee.shape[0] == 6:
            jacobian_ee = jacobian_ee[:3]
        ee_current = ee_current[:3]
        ee_target = ee_target[:3]
        ee_velocity = ee_velocity[:3]
        task_dim = 3
    else:
        task_dim = jacobian_ee.shape[0]

    if ee_velocity_target is None:
        ee_velocity_target = np.zeros(task_dim)
    else:
        ee_velocity_target = np.asarray(ee_velocity_target, dtype=np.float64)
        if position_only and len(ee_velocity_target) > 3:
            ee_velocity_target = ee_velocity_target[:3]

    # Compute error
    error_p = ee_target - ee_current
    error_v = ee_velocity_target - ee_velocity

    target_accel = gain_p * error_p + gain_d * error_v

    return Task(
        name="end_effector",
        task_type=TaskType.SOFT,
        priority=priority,
        jacobian=jacobian_ee,
        target=target_accel,
        weight=np.full(task_dim, weight),
        gain_p=gain_p,
        gain_d=gain_d,
    )


def create_contact_constraint(
    jacobian_contact: NDArray[np.float64],
    contact_velocity: NDArray[np.float64],
    priority: int = 0,
) -> Task:
    """Create a contact constraint (zero velocity at contact).

    Args:
        jacobian_contact: Contact point Jacobian (3 or 6, n_v).
        contact_velocity: Current contact velocity (should be ~0).
        priority: Task priority (typically 0 for highest).

    Returns:
        Task configured as equality constraint for contact.
    """
    jacobian_contact = np.asarray(jacobian_contact, dtype=np.float64)
    task_dim = jacobian_contact.shape[0]

    # Contact constraint: J @ ddq = -J_dot @ v ≈ 0 for stationary contact
    # We want the contact point to maintain zero velocity
    # ddx = J @ ddq + J_dot @ v = 0
    # So J @ ddq = -J_dot @ v
    # For simplicity, target zero acceleration (assumes J_dot @ v ≈ 0)
    target = np.zeros(task_dim)

    return Task(
        name="contact",
        task_type=TaskType.EQUALITY,
        priority=priority,
        jacobian=jacobian_contact,
        target=target,
    )


def create_joint_limit_task(
    n_v: int,
    q_current: NDArray[np.float64],
    v_current: NDArray[np.float64],
    q_min: NDArray[np.float64],
    q_max: NDArray[np.float64],
    margin: float = 0.1,
    priority: int = 1,
    gain: float = 100.0,
) -> Task | None:
    """Create joint limit avoidance task.

    Only creates task when close to limits.

    Args:
        n_v: Number of velocity coordinates.
        q_current: Current joint positions.
        v_current: Current joint velocities.
        q_min: Minimum joint limits.
        q_max: Maximum joint limits.
        margin: Distance from limit to activate [rad].
        priority: Task priority.
        gain: Repulsion gain.

    Returns:
        Task for limit avoidance, or None if not near limits.
    """
    q_current = np.asarray(q_current, dtype=np.float64)[:n_v]
    v_current = np.asarray(v_current, dtype=np.float64)[:n_v]
    q_min = np.asarray(q_min, dtype=np.float64)[:n_v]
    q_max = np.asarray(q_max, dtype=np.float64)[:n_v]

    # Find joints near limits
    near_lower = q_current < (q_min + margin)
    near_upper = q_current > (q_max - margin)
    near_limit = near_lower | near_upper

    if not np.any(near_limit):
        return None

    # Select active joints
    active_indices = np.where(near_limit)[0]
    n_active = len(active_indices)

    # Build Jacobian for active joints only
    jacobian = np.zeros((n_active, n_v))
    for i, idx in enumerate(active_indices):
        jacobian[i, idx] = 1.0

    # Compute repulsive acceleration
    target = np.zeros(n_active)
    for i, idx in enumerate(active_indices):
        if near_lower[idx]:
            # Push away from lower limit
            dist = q_current[idx] - q_min[idx]
            target[i] = gain * (margin - dist) - 10.0 * v_current[idx]
        else:
            # Push away from upper limit
            dist = q_max[idx] - q_current[idx]
            target[i] = -gain * (margin - dist) - 10.0 * v_current[idx]

    return Task(
        name="joint_limits",
        task_type=TaskType.INEQUALITY,
        priority=priority,
        jacobian=jacobian,
        target=target,
        lower_bound=-np.inf * np.ones(n_active),
        upper_bound=np.inf * np.ones(n_active),
    )
