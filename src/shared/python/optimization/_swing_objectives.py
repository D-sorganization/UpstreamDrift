import numpy as np

from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.optimization._swing_kinematics import (
    JOINTS,
    vector_to_trajectory,
)
from src.shared.python.optimization._swing_models import (
    ClubModel,
    OptimizationConfig,
    OptimizationObjective,
    SwingTrajectory,
)


def compute_objective(
    x: np.ndarray,
    config: OptimizationConfig,
    torque_limits: dict[str, float],
    golfer_model,
    club: ClubModel,
    system_moi: float,
) -> float:
    if not (x is not None):
        raise ValueError("x must be provided")
    trajectory = vector_to_trajectory(x, config, golfer_model, club, system_moi)
    objective = 0.0

    if OptimizationObjective.CLUBHEAD_VELOCITY in config.objectives:
        weight = config.objectives[OptimizationObjective.CLUBHEAD_VELOCITY]
        speed = trajectory.impact_speed
        objective -= weight * speed / 50.0

    if OptimizationObjective.INJURY_RISK in config.objectives:
        weight = config.objectives[OptimizationObjective.INJURY_RISK]
        risk = compute_injury_risk(trajectory, torque_limits)
        objective += weight * risk / 100.0

    if OptimizationObjective.ENERGY_EFFICIENCY in config.objectives:
        weight = config.objectives[OptimizationObjective.ENERGY_EFFICIENCY]
        energy = compute_energy_cost(trajectory)
        objective += weight * energy / 1000.0

    return objective


def compute_injury_risk(
    trajectory: SwingTrajectory,
    torque_limits: dict[str, float],
) -> float:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    risk = 0.0

    for vel in trajectory.joint_velocities.values():
        max_vel = np.max(np.abs(vel))
        if max_vel > 20:
            risk += 10

    for joint, torque in trajectory.joint_torques.items():
        max_torque = np.max(np.abs(torque))
        limit = torque_limits.get(joint, 100)
        if max_torque > 0.8 * limit:
            risk += 15

    trunk_rot = trajectory.joint_angles.get("trunk_rotation", np.zeros(1))
    max_rotation = np.max(np.abs(trunk_rot))
    if max_rotation > 1.2:
        risk += 20

    return min(risk, 100)


def compute_energy_cost(trajectory: SwingTrajectory) -> float:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    total_work = 0.0
    dt = trajectory.time[1] - trajectory.time[0] if len(trajectory.time) > 1 else 0.001

    for joint in JOINTS:
        if joint in trajectory.joint_torques and joint in trajectory.joint_velocities:
            torque = trajectory.joint_torques[joint]
            velocity = trajectory.joint_velocities[joint]
            power = torque * velocity
            if hasattr(np, "trapezoid"):
                work = np.trapezoid(np.abs(power), dx=dt)
            else:
                trapz_func = getattr(np, "trapz")  # noqa: B009
                work = trapz_func(np.abs(power), dx=dt)
            total_work += work

    return total_work


def compute_metrics(
    trajectory: SwingTrajectory,
    club: ClubModel,
    torque_limits: dict[str, float],
) -> dict:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    clubhead_speed = trajectory.impact_speed

    smash_factor = 1.50 if club.loft_angle < 15 else 1.35
    ball_speed = clubhead_speed * smash_factor

    launch_angle = club.loft_angle
    carry = (ball_speed**2 * np.sin(2 * np.radians(launch_angle))) / GRAVITY_M_S2
    carry *= 0.9

    spin_rate = 2500 if club.loft_angle < 15 else 6000

    injury_risk = compute_injury_risk(trajectory, torque_limits)

    trunk_vel = trajectory.joint_velocities.get("trunk_rotation", np.zeros(1))
    max_trunk_vel = np.max(np.abs(trunk_vel))
    spinal_compression = 4.0 + max_trunk_vel * 0.2
    spinal_shear = 0.3 + max_trunk_vel * 0.05

    return {
        "clubhead_speed": clubhead_speed,
        "ball_speed": ball_speed,
        "carry_distance": carry,
        "launch_angle": launch_angle,
        "spin_rate": spin_rate,
        "spinal_compression": spinal_compression,
        "spinal_shear": spinal_shear,
        "injury_risk": injury_risk,
    }
