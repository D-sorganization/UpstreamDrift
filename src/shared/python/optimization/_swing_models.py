from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class OptimizationObjective(Enum):
    """Available optimization objectives."""

    CLUBHEAD_VELOCITY = "clubhead_velocity"
    BALL_DISTANCE = "ball_distance"
    ACCURACY = "accuracy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    INJURY_RISK = "injury_risk"
    CONSISTENCY = "consistency"


class OptimizationConstraint(Enum):
    """Available optimization constraints."""

    JOINT_LIMITS = "joint_limits"
    TORQUE_LIMITS = "torque_limits"
    VELOCITY_LIMITS = "velocity_limits"
    CONTACT_CONSTRAINTS = "contact_constraints"
    KINEMATIC_CHAIN = "kinematic_chain"


@dataclass
class GolferModel:
    """Model of a golfer's physical characteristics."""

    height: float = 1.75
    mass: float = 75.0
    arm_length: float = 0.60
    trunk_length: float = 0.50

    arm_mass_ratio: float = 0.05
    trunk_mass_ratio: float = 0.43

    shoulder_rom: tuple[float, float] = (-2.5, 2.5)
    elbow_rom: tuple[float, float] = (0.0, 2.4)
    wrist_rom: tuple[float, float] = (-1.2, 1.2)
    hip_rom: tuple[float, float] = (-0.8, 0.8)
    trunk_rotation_rom: tuple[float, float] = (-1.5, 1.5)

    max_shoulder_torque: float = 100.0
    max_elbow_torque: float = 60.0
    max_wrist_torque: float = 20.0
    max_hip_torque: float = 150.0
    max_trunk_torque: float = 200.0

    flexibility_factor: float = 1.0


@dataclass
class ClubModel:
    """Model of a golf club."""

    total_length: float = 1.15
    shaft_length: float = 1.05
    head_mass: float = 0.20
    shaft_mass: float = 0.07
    grip_mass: float = 0.05

    shaft_flex: str = "regular"
    kick_point: str = "mid"

    loft_angle: float = 10.5
    face_angle: float = 0.0
    lie_angle: float = 56.0

    @property
    def total_mass(self) -> float:
        """Return the combined mass of head, shaft, and grip."""
        return self.head_mass + self.shaft_mass + self.grip_mass

    @property
    def club_moi(self) -> float:
        """Moment of inertia about grip end."""
        return (
            self.head_mass * self.total_length**2
            + self.shaft_mass * (self.shaft_length / 2) ** 2
        )


@dataclass
class OptimizationConfig:
    """Configuration for the optimization problem."""

    objectives: dict[OptimizationObjective, float] = field(
        default_factory=lambda: {
            OptimizationObjective.CLUBHEAD_VELOCITY: 1.0,
            OptimizationObjective.INJURY_RISK: 0.3,
        }
    )

    constraints: list[OptimizationConstraint] = field(
        default_factory=lambda: [
            OptimizationConstraint.JOINT_LIMITS,
            OptimizationConstraint.TORQUE_LIMITS,
        ]
    )

    n_nodes: int = 50
    swing_duration: float = 1.2
    backswing_fraction: float = 0.4

    max_iterations: int = 500
    tolerance: float = 1e-6
    solver: str = "SLSQP"

    @property
    def method(self) -> str:
        """Backward-compatible alias for the configured solver name."""
        return self.solver


@dataclass
class SwingTrajectory:
    """Represents a complete swing trajectory."""

    time: np.ndarray
    joint_angles: dict[str, np.ndarray]
    joint_velocities: dict[str, np.ndarray]
    joint_torques: dict[str, np.ndarray]

    clubhead_position: np.ndarray
    clubhead_velocity: np.ndarray

    impact_speed: float = 0.0
    impact_time: float = 0.0


@dataclass
class OptimizationResult:
    """Results from swing optimization."""

    success: bool
    message: str

    trajectory: SwingTrajectory | None = None

    predicted_clubhead_speed: float = 0.0
    predicted_ball_speed: float = 0.0
    predicted_carry_distance: float = 0.0
    predicted_launch_angle: float = 0.0
    predicted_spin_rate: float = 0.0

    peak_spinal_compression: float = 0.0
    peak_spinal_shear: float = 0.0
    injury_risk_score: float = 0.0

    objective_value: float = 0.0
    iterations: int = 0
    computation_time: float = 0.0

    speed_improvement: float = 0.0
    risk_reduction: float = 0.0

    @property
    def solver_status(self) -> str:
        """Return the canonical optimization status string."""
        return "success" if self.success else "failure"
