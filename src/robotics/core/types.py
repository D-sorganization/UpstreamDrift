"""Core type definitions for robotics module.

This module defines the fundamental data structures used throughout
the robotics system. All types are immutable dataclasses where possible
to ensure thread safety and predictable behavior.

Design by Contract:
    All dataclasses validate their invariants on construction.
    Invalid states are not representable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class ContactType(Enum):
    """Type of contact between bodies."""

    POINT = auto()  # Single point contact
    LINE = auto()  # Edge/line contact
    PATCH = auto()  # Surface/patch contact (e.g., foot)
    SOFT = auto()  # Soft/deformable contact


class FrictionConeType(Enum):
    """Friction cone approximation method."""

    EXACT = auto()  # Second-order cone (exact)
    LINEARIZED_4 = auto()  # 4-sided pyramid approximation
    LINEARIZED_8 = auto()  # 8-sided pyramid approximation
    LINEARIZED_16 = auto()  # 16-sided pyramid approximation


class TaskPriority(Enum):
    """Priority levels for whole-body control tasks."""

    HARD_CONSTRAINT = 0  # Must be satisfied (contact, dynamics)
    SAFETY = 1  # Safety constraints (joint limits)
    PRIMARY = 2  # Primary objective (CoM tracking)
    SECONDARY = 3  # Secondary objective (posture)
    TERTIARY = 4  # Tertiary objective (regularization)


class ControlMode(Enum):
    """Control modes for robot actuation."""

    TORQUE = auto()  # Direct torque control
    POSITION = auto()  # Position control
    VELOCITY = auto()  # Velocity control
    IMPEDANCE = auto()  # Impedance control
    ADMITTANCE = auto()  # Admittance control
    HYBRID = auto()  # Hybrid force/position


class GaitPhase(Enum):
    """Phases of a walking gait cycle."""

    DOUBLE_SUPPORT = auto()  # Both feet on ground
    LEFT_SWING = auto()  # Left foot swinging
    RIGHT_SWING = auto()  # Right foot swinging
    FLIGHT = auto()  # Both feet in air (running)


class SupportState(Enum):
    """Support state for balance control."""

    DOUBLE = auto()  # Double support
    LEFT_SINGLE = auto()  # Left foot only
    RIGHT_SINGLE = auto()  # Right foot only
    FLIGHT = auto()  # No ground contact


@dataclass(frozen=True)
class ContactState:
    """Immutable representation of a contact state.

    Design by Contract:
        Invariants:
            - position.shape == (3,)
            - normal.shape == (3,)
            - ||normal|| == 1 (unit normal)
            - penetration >= 0
            - normal_force >= 0
            - friction_coefficient >= 0

    Attributes:
        contact_id: Unique identifier for this contact.
        body_a: Name of first body in contact.
        body_b: Name of second body in contact.
        position: Contact point in world frame [m].
        normal: Contact normal pointing from B to A.
        penetration: Penetration depth [m] (>= 0).
        normal_force: Normal force magnitude [N].
        friction_force: Friction force vector [N].
        friction_coefficient: Coulomb friction coefficient.
        contact_type: Type of contact geometry.
        is_active: Whether contact is currently active.
    """

    contact_id: int
    body_a: str
    body_b: str
    position: NDArray[np.float64]
    normal: NDArray[np.float64]
    penetration: float = 0.0
    normal_force: float = 0.0
    friction_force: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    friction_coefficient: float = 0.5
    contact_type: ContactType = ContactType.POINT
    is_active: bool = True

    def __post_init__(self) -> None:
        """Validate invariants after construction."""
        # Convert to numpy arrays if needed (for frozen dataclass)
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )
        object.__setattr__(self, "normal", np.asarray(self.normal, dtype=np.float64))
        object.__setattr__(
            self, "friction_force", np.asarray(self.friction_force, dtype=np.float64)
        )

        # Validate shapes
        if self.position.shape != (3,):
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        if self.normal.shape != (3,):
            raise ValueError(f"normal must be (3,), got {self.normal.shape}")
        if self.friction_force.shape != (3,):
            raise ValueError(
                f"friction_force must be (3,), got {self.friction_force.shape}"
            )

        # Validate values
        if self.penetration < 0:
            raise ValueError(f"penetration must be >= 0, got {self.penetration}")
        if self.normal_force < 0:
            raise ValueError(f"normal_force must be >= 0, got {self.normal_force}")
        if self.friction_coefficient < 0:
            raise ValueError(
                f"friction_coefficient must be >= 0, got {self.friction_coefficient}"
            )

        # Normalize normal vector
        norm = np.linalg.norm(self.normal)
        if norm > 1e-10:
            object.__setattr__(self, "normal", self.normal / norm)

    def get_wrench(self) -> NDArray[np.float64]:
        """Get contact wrench (force + torque) at contact point.

        Returns:
            6D wrench [fx, fy, fz, tx, ty, tz] in world frame.
        """
        force = self.normal * self.normal_force + self.friction_force
        # Torque is zero at contact point (point contact assumption)
        return np.concatenate([force, np.zeros(3)])

    def is_sliding(self, tolerance: float = 1e-6) -> bool:
        """Check if contact is at friction limit (sliding).

        Args:
            tolerance: Numerical tolerance for comparison.

        Returns:
            True if friction force magnitude equals friction limit.
        """
        friction_limit = self.friction_coefficient * self.normal_force
        friction_mag = float(np.linalg.norm(self.friction_force))
        return friction_mag >= friction_limit - tolerance

    def with_force(
        self,
        normal_force: float,
        friction_force: NDArray[np.float64] | None = None,
    ) -> ContactState:
        """Create new ContactState with updated forces.

        Args:
            normal_force: New normal force magnitude.
            friction_force: New friction force vector (optional).

        Returns:
            New ContactState with updated forces.
        """
        return ContactState(
            contact_id=self.contact_id,
            body_a=self.body_a,
            body_b=self.body_b,
            position=self.position.copy(),
            normal=self.normal.copy(),
            penetration=self.penetration,
            normal_force=normal_force,
            friction_force=(
                friction_force.copy()
                if friction_force is not None
                else self.friction_force.copy()
            ),
            friction_coefficient=self.friction_coefficient,
            contact_type=self.contact_type,
            is_active=self.is_active,
        )


@dataclass(frozen=True)
class TaskDescriptor:
    """Descriptor for a whole-body control task.

    Design by Contract:
        Invariants:
            - priority >= 0
            - weight > 0
            - jacobian and target dimensions are consistent

    Attributes:
        name: Human-readable task name.
        priority: Task priority (lower = higher priority).
        task_type: Type of task constraint.
        jacobian: Task Jacobian matrix (task_dim x n_v).
        target: Desired task-space acceleration.
        weight: Diagonal weight for soft tasks.
        lower_bound: Lower bound for inequality tasks.
        upper_bound: Upper bound for inequality tasks.
    """

    name: str
    priority: TaskPriority
    jacobian: NDArray[np.float64]
    target: NDArray[np.float64]
    weight: NDArray[np.float64] | None = None
    lower_bound: NDArray[np.float64] | None = None
    upper_bound: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate task descriptor."""
        object.__setattr__(
            self, "jacobian", np.asarray(self.jacobian, dtype=np.float64)
        )
        object.__setattr__(self, "target", np.asarray(self.target, dtype=np.float64))

        if self.weight is not None:
            object.__setattr__(
                self, "weight", np.asarray(self.weight, dtype=np.float64)
            )

        # Validate dimensions
        if self.jacobian.ndim != 2:
            raise ValueError(f"jacobian must be 2D, got {self.jacobian.ndim}D")

        task_dim = self.jacobian.shape[0]
        if self.target.shape != (task_dim,):
            raise ValueError(
                f"target shape {self.target.shape} doesn't match "
                f"jacobian rows {task_dim}"
            )

    @property
    def task_dim(self) -> int:
        """Get task space dimension."""
        return self.jacobian.shape[0]

    @property
    def config_dim(self) -> int:
        """Get configuration space dimension."""
        return self.jacobian.shape[1]


@dataclass(frozen=True)
class FootstepTarget:
    """Target footstep for locomotion planning.

    Attributes:
        position: Footstep position [x, y, z] in world frame [m].
        orientation: Footstep orientation as quaternion [w, x, y, z].
        foot: Which foot ('left' or 'right').
        timing: Desired landing time [s] from start.
        duration: Desired stance duration [s].
    """

    position: NDArray[np.float64]
    orientation: NDArray[np.float64]
    foot: str  # 'left' or 'right'
    timing: float
    duration: float

    def __post_init__(self) -> None:
        """Validate footstep target."""
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )
        object.__setattr__(
            self, "orientation", np.asarray(self.orientation, dtype=np.float64)
        )

        if self.position.shape != (3,):
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"orientation must be (4,), got {self.orientation.shape}")
        if self.foot not in ("left", "right"):
            raise ValueError(f"foot must be 'left' or 'right', got '{self.foot}'")
        if self.timing < 0:
            raise ValueError(f"timing must be >= 0, got {self.timing}")
        if self.duration <= 0:
            raise ValueError(f"duration must be > 0, got {self.duration}")


@dataclass
class RobotState:
    """Mutable robot state for real-time control.

    Attributes:
        timestamp: State timestamp [s].
        q: Joint positions [rad or m].
        v: Joint velocities [rad/s or m/s].
        tau: Joint torques [Nm or N].
        contacts: Active contact states.
    """

    timestamp: float
    q: NDArray[np.float64]
    v: NDArray[np.float64]
    tau: NDArray[np.float64] | None = None
    contacts: list[ContactState] = field(default_factory=list)

    @property
    def n_q(self) -> int:
        """Number of position coordinates."""
        return len(self.q)

    @property
    def n_v(self) -> int:
        """Number of velocity coordinates."""
        return len(self.v)


@dataclass(frozen=True)
class SolverResult:
    """Result from an optimization solver.

    Attributes:
        success: Whether solver found a solution.
        solution: Solution vector (if successful).
        cost: Optimal cost value.
        iterations: Number of iterations used.
        solve_time: Wall-clock solve time [s].
        status: Solver-specific status string.
        dual_solution: Dual variables (if available).
    """

    success: bool
    solution: NDArray[np.float64] | None
    cost: float = float("inf")
    iterations: int = 0
    solve_time: float = 0.0
    status: str = ""
    dual_solution: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class SensorReading:
    """Base class for sensor readings.

    Attributes:
        timestamp: Reading timestamp [s].
        sensor_id: Unique sensor identifier.
    """

    timestamp: float
    sensor_id: str


@dataclass(frozen=True)
class ForceTorqueReading(SensorReading):
    """Force/torque sensor reading.

    Attributes:
        wrench: 6D wrench [fx, fy, fz, tx, ty, tz].
    """

    wrench: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate reading."""
        object.__setattr__(self, "wrench", np.asarray(self.wrench, dtype=np.float64))
        if self.wrench.shape != (6,):
            raise ValueError(f"wrench must be (6,), got {self.wrench.shape}")

    @property
    def force(self) -> NDArray[np.float64]:
        """Get force component [fx, fy, fz]."""
        return self.wrench[:3]

    @property
    def torque(self) -> NDArray[np.float64]:
        """Get torque component [tx, ty, tz]."""
        return self.wrench[3:]


@dataclass(frozen=True)
class IMUReading(SensorReading):
    """IMU sensor reading.

    Attributes:
        linear_acceleration: Linear acceleration [ax, ay, az] [m/s^2].
        angular_velocity: Angular velocity [wx, wy, wz] [rad/s].
        orientation: Orientation quaternion [w, x, y, z] (if available).
    """

    linear_acceleration: NDArray[np.float64]
    angular_velocity: NDArray[np.float64]
    orientation: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate reading."""
        object.__setattr__(
            self,
            "linear_acceleration",
            np.asarray(self.linear_acceleration, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            np.asarray(self.angular_velocity, dtype=np.float64),
        )

        if self.linear_acceleration.shape != (3,):
            raise ValueError(
                f"linear_acceleration must be (3,), "
                f"got {self.linear_acceleration.shape}"
            )
        if self.angular_velocity.shape != (3,):
            raise ValueError(
                f"angular_velocity must be (3,), got {self.angular_velocity.shape}"
            )

        if self.orientation is not None:
            object.__setattr__(
                self,
                "orientation",
                np.asarray(self.orientation, dtype=np.float64),
            )
            if self.orientation.shape != (4,):
                raise ValueError(
                    f"orientation must be (4,), got {self.orientation.shape}"
                )
