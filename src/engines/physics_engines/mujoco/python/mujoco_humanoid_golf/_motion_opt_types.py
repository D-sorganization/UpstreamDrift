from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizationObjectives:
    """Objectives for trajectory optimization."""

    maximize_club_speed: bool = True
    minimize_energy: bool = True
    minimize_jerk: bool = True
    minimize_torque: bool = True
    target_ball_position: np.ndarray | None = None

    weight_speed: float = 10.0
    weight_energy: float = 1.0
    weight_jerk: float = 0.5
    weight_torque: float = 0.1
    weight_accuracy: float = 5.0


@dataclass
class OptimizationConstraints:
    """Constraints for trajectory optimization."""

    joint_position_limits: bool = True
    joint_velocity_limits: bool = True
    joint_torque_limits: bool = True
    collision_avoidance: bool = False
    maintain_grip: bool = True
    balance_constraint: bool = False

    max_joint_velocity: np.ndarray | None = None
    max_joint_torque: np.ndarray | None = None


@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""

    success: bool
    optimal_trajectory: np.ndarray
    optimal_velocities: np.ndarray
    optimal_controls: np.ndarray
    objective_value: float
    num_iterations: int
    computation_time: float
    peak_club_speed: float
    final_club_position: np.ndarray
