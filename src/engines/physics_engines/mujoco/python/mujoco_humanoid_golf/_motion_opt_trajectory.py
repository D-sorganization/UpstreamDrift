from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def generate_initial_guess(
    model,
    num_knot_points: int,
) -> np.ndarray:
    if not (model is not None):
        raise ValueError("model must be provided")
    address_pose = np.zeros(model.nv)
    backswing_pose = np.zeros(model.nv)
    downswing_pose = np.zeros(model.nv)
    impact_pose = np.zeros(model.nv)
    followthrough_pose = np.zeros(model.nv)

    if model.nv >= 10:
        backswing_pose[0] = -1.5
        backswing_pose[1] = 0.5
        backswing_pose[2] = 1.5

    downswing_pose = (backswing_pose + impact_pose) / 2

    if model.nv >= 10:
        impact_pose[0] = 1.0
        impact_pose[3] = -0.5

    if model.nv >= 10:
        followthrough_pose[0] = 1.8

    key_poses = np.array(
        [
            address_pose,
            address_pose,
            backswing_pose,
            downswing_pose,
            impact_pose,
            followthrough_pose,
            followthrough_pose,
            followthrough_pose,
            followthrough_pose,
            followthrough_pose,
        ],
    )

    return key_poses[:num_knot_points]


def compute_bounds(
    model,
    constraints,
    num_knot_points: int,
) -> list[tuple[float, float]]:
    if not (model is not None):
        raise ValueError("model must be provided")
    bounds = []

    for _knot in range(num_knot_points):
        for joint_idx in range(model.njnt):
            if constraints.joint_position_limits and model.jnt_limited[joint_idx]:
                q_min = model.jnt_range[joint_idx, 0]
                q_max = model.jnt_range[joint_idx, 1]
            else:
                q_min = -np.pi
                q_max = np.pi

            bounds.append((q_min, q_max))

        for _ in range(model.nv - model.njnt):
            bounds.append((-10.0, 10.0))

    return bounds


def interpolate_trajectory(
    model,
    trajectory: np.ndarray,
    swing_duration: float,
    num_knot_points: int,
) -> tuple[np.ndarray, float, int]:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    dt = model.opt.timestep
    num_steps = int(swing_duration / dt)

    knot_times = np.linspace(0, swing_duration, num_knot_points)
    sim_times = np.linspace(0, swing_duration, num_steps)

    trajectory_interp = np.zeros((num_steps, model.nv))
    for dof in range(model.nv):
        spline = CubicSpline(knot_times, trajectory[:, dof])
        trajectory_interp[:, dof] = spline(sim_times)

    return trajectory_interp, dt, num_steps


def compute_jerk(
    trajectory: np.ndarray,
    swing_duration: float,
    num_knot_points: int,
) -> float:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    dt = swing_duration / (num_knot_points - 1)

    accel = np.diff(trajectory, n=2, axis=0) / dt**2
    jerk = np.diff(accel, axis=0) / dt

    return float(np.sum(np.abs(jerk)))
