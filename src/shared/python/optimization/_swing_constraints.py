import numpy as np

from src.shared.python.optimization._swing_kinematics import (
    JOINTS,
    vector_to_trajectory,
)
from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    OptimizationConstraint,
)


def build_constraints(
    config: OptimizationConfig,
    golfer: GolferModel,
    club: ClubModel,
    torque_limits: dict[str, float],
    system_moi: float,
) -> list[dict]:
    constraints = []

    if OptimizationConstraint.TORQUE_LIMITS in config.constraints:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: torque_constraint(
                    x, config, golfer, club, torque_limits, system_moi
                ),
            }
        )

    if OptimizationConstraint.KINEMATIC_CHAIN in config.constraints:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: kinematic_sequence_constraint(
                    x, config, golfer, club, system_moi
                ),
            }
        )

    return constraints


def torque_constraint(
    x: np.ndarray,
    config: OptimizationConfig,
    golfer: GolferModel,
    club: ClubModel,
    torque_limits: dict[str, float],
    system_moi: float,
) -> np.ndarray:
    if not (x is not None):
        raise ValueError("x must be provided")
    trajectory = vector_to_trajectory(x, config, golfer, club, system_moi)
    violations = []

    for joint in JOINTS:
        torque = trajectory.joint_torques[joint]
        limit = torque_limits[joint]
        violations.extend(limit - np.abs(torque))

    return np.array(violations)


def kinematic_sequence_constraint(
    x: np.ndarray,
    config: OptimizationConfig,
    golfer: GolferModel,
    club: ClubModel,
    system_moi: float,
) -> np.ndarray:
    if not (x is not None):
        raise ValueError("x must be provided")
    trajectory = vector_to_trajectory(x, config, golfer, club, system_moi)

    sequence = [
        "hip_rotation",
        "trunk_rotation",
        "shoulder_horizontal",
        "wrist_cock",
    ]

    peak_times = []
    for joint in sequence:
        if joint in trajectory.joint_velocities:
            vel = np.abs(trajectory.joint_velocities[joint])
            peak_idx = np.argmax(vel)
            peak_times.append(trajectory.time[peak_idx])

    violations = []
    for i in range(len(peak_times) - 1):
        violations.append(peak_times[i + 1] - peak_times[i])

    return np.array(violations)
