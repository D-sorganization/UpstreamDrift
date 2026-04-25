from typing import cast

import numpy as np

from src.shared.python.optimization._swing_models import (
    ClubModel,
    GolferModel,
    OptimizationConfig,
    SwingTrajectory,
)

JOINTS = [
    "hip_rotation",
    "trunk_rotation",
    "shoulder_horizontal",
    "shoulder_vertical",
    "elbow_flexion",
    "wrist_cock",
    "wrist_rotation",
]


def generate_initial_guess(
    golfer: GolferModel,
    config: OptimizationConfig,
    joint_limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    n_joints = len(JOINTS)
    n_nodes = config.n_nodes
    t = np.linspace(0, config.swing_duration, n_nodes)
    t_top = config.swing_duration * config.backswing_fraction

    angles = np.zeros((n_joints, n_nodes))
    velocities = np.zeros((n_joints, n_nodes))

    for i, joint in enumerate(JOINTS):
        lo, hi = joint_limits[joint]
        mid = (lo + hi) / 2
        amp = (hi - lo) / 4

        for j, tj in enumerate(t):
            if tj <= t_top:
                phase = np.pi * tj / t_top
                angles[i, j] = mid + amp * np.sin(phase)
            else:
                phase = np.pi * (tj - t_top) / (config.swing_duration - t_top)
                angles[i, j] = mid + amp * np.sin(np.pi - phase)

        dt = t[1] - t[0]
        velocities[i, :] = np.gradient(angles[i, :], dt)

    x = np.concatenate([angles.flatten(), velocities.flatten()])
    return cast(np.ndarray, x)


def trajectory_to_vector(trajectory: SwingTrajectory) -> np.ndarray:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    angles = np.array([trajectory.joint_angles[j] for j in JOINTS])
    velocities = np.array([trajectory.joint_velocities[j] for j in JOINTS])
    return np.concatenate([angles.flatten(), velocities.flatten()])


def vector_to_trajectory(
    x: np.ndarray,
    config: OptimizationConfig,
    golfer: GolferModel,
    club: ClubModel,
    system_moi: float,
) -> SwingTrajectory:
    if not (x is not None):
        raise ValueError("x must be provided")
    n_joints = len(JOINTS)
    n_nodes = config.n_nodes

    angles = x[: n_joints * n_nodes].reshape(n_joints, n_nodes)
    velocities = x[n_joints * n_nodes :].reshape(n_joints, n_nodes)

    t = np.linspace(0, config.swing_duration, n_nodes)

    joint_angles = {JOINTS[i]: angles[i] for i in range(n_joints)}
    joint_velocities = {JOINTS[i]: velocities[i] for i in range(n_joints)}

    joint_torques = {}
    dt = t[1] - t[0]
    for i, joint in enumerate(JOINTS):
        accel = np.gradient(velocities[i], dt)
        joint_torques[joint] = system_moi * accel * 0.1

    clubhead_pos, clubhead_vel = compute_clubhead_trajectory(
        joint_angles, t, golfer, club
    )

    speed = np.linalg.norm(clubhead_vel, axis=1)
    impact_idx = np.argmax(speed)

    return SwingTrajectory(
        time=t,
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        clubhead_position=clubhead_pos,
        clubhead_velocity=clubhead_vel,
        impact_speed=speed[impact_idx],
        impact_time=t[impact_idx],
    )


def compute_clubhead_trajectory(
    joint_angles: dict[str, np.ndarray],
    time: np.ndarray,
    golfer: GolferModel,
    club: ClubModel,
) -> tuple[np.ndarray, np.ndarray]:
    if not (joint_angles is not None):
        raise ValueError("joint_angles must be provided")
    n_frames = len(time)
    position = np.zeros((n_frames, 3))
    velocity = np.zeros((n_frames, 3))

    arm_length = golfer.arm_length
    club_length = club.total_length

    for i in range(n_frames):
        trunk_rot = joint_angles.get("trunk_rotation", np.zeros(n_frames))[i]
        shoulder_h = joint_angles.get("shoulder_horizontal", np.zeros(n_frames))[i]
        joint_angles.get("shoulder_vertical", np.zeros(n_frames))[i]
        joint_angles.get("elbow_flexion", np.zeros(n_frames))[i]
        wrist = joint_angles.get("wrist_cock", np.zeros(n_frames))[i]

        total_angle = trunk_rot + shoulder_h + wrist

        position[i, 0] = (arm_length + club_length) * np.sin(total_angle)
        position[i, 1] = 0
        position[i, 2] = (arm_length + club_length) * np.cos(total_angle) - club_length

    dt = time[1] - time[0] if len(time) > 1 else 0.001
    for dim in range(3):
        velocity[:, dim] = np.gradient(position[:, dim], dt)

    return position, velocity


def get_bounds(
    golfer: GolferModel,
    config: OptimizationConfig,
    joint_limits: dict[str, tuple[float, float]],
) -> list[tuple[float, float]]:
    bounds = []

    for joint in JOINTS:
        lo, hi = joint_limits[joint]
        flex = golfer.flexibility_factor
        for _ in range(config.n_nodes):
            bounds.append((lo * flex, hi * flex))

    max_vel = 30.0
    for _ in range(len(JOINTS) * config.n_nodes):
        bounds.append((-max_vel, max_vel))

    return bounds
