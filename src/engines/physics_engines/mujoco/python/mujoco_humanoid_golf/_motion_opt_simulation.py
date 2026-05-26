from __future__ import annotations

import mujoco
import numpy as np

from ._motion_opt_trajectory import interpolate_trajectory


def setup_velocity_constraint(model, constraints, num_knot_points, swing_duration):
    if not (model is not None):
        raise ValueError("model must be provided")

    def velocity_constraint(x: np.ndarray) -> np.ndarray:
        trajectory = x.reshape(num_knot_points, model.nv)
        dt = swing_duration / (num_knot_points - 1)

        velocities = np.diff(trajectory, axis=0) / dt

        max_vel = constraints.max_joint_velocity
        if max_vel is None:
            max_vel = np.ones(model.nv) * 10.0

        violations = max_vel - np.abs(velocities)
        return np.asarray(violations.flatten())

    return velocity_constraint


def setup_constraints(model, constraints, num_knot_points, swing_duration):
    if not (model is not None):
        raise ValueError("model must be provided")
    constraint_list = []

    if constraints.joint_velocity_limits:
        fn = setup_velocity_constraint(
            model, constraints, num_knot_points, swing_duration
        )
        constraint_list.append({"type": "ineq", "fun": fn})

    return constraint_list


def detect_jacobian_api(model, data, club_head_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    jacp_flat = np.zeros(3 * model.nv)
    jacr_flat = np.zeros(3 * model.nv)
    use_flat_jac = False

    if club_head_id is not None:
        try:
            mujoco.mj_jacBody(model, data, jacp, jacr, club_head_id)
        except TypeError:
            use_flat_jac = True

    return jacp, jacr, jacp_flat, jacr_flat, use_flat_jac


def compute_club_speed(
    model,
    data,
    club_head_id,
    jacp,
    jacr,
    jacp_flat,
    jacr_flat,
    use_flat_jac,
) -> float:
    if not (jacp is not None):
        raise ValueError("jacp must be provided")
    if use_flat_jac:
        mujoco.mj_jacBody(model, data, jacp_flat, jacr_flat, club_head_id)
        jacp[:] = jacp_flat.reshape(3, model.nv)
    else:
        mujoco.mj_jacBody(model, data, jacp, jacr, club_head_id)

    vel = jacp @ data.qvel
    return float(np.linalg.norm(vel))


def collect_simulation_metrics(
    model,
    club_speeds,
    club_positions,
    controls,
    velocities,
) -> dict:
    if not (club_speeds is not None):
        raise ValueError("club_speeds must be provided")
    peak_club_speed = float(max(float(s) for s in club_speeds)) if club_speeds else 0.0
    total_energy = np.vdot(
        np.abs(controls), np.abs(velocities[:, : model.nu])
    )  # ⚡ Bolt: np.vdot is ~1.5x faster than np.sum(a * b)
    final_club_position = club_positions[-1] if club_positions else np.zeros(3)

    return {
        "peak_club_speed": peak_club_speed,
        "total_energy": total_energy,
        "final_club_position": final_club_position,
    }


def simulate_trajectory(
    model,
    data,
    trajectory: np.ndarray,
    club_head_id,
    swing_duration: float,
    num_knot_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not (trajectory is not None):
        raise ValueError("trajectory must be provided")
    trajectory_interp, dt, num_steps = interpolate_trajectory(
        model, trajectory, swing_duration, num_knot_points
    )

    velocities = np.zeros((num_steps, model.nv))
    controls = np.zeros((num_steps, model.nu))
    club_speeds: list[float] = []
    club_positions: list[np.ndarray] = []

    mujoco.mj_resetData(model, data)
    jacp, jacr, jacp_flat, jacr_flat, use_flat_jac = detect_jacobian_api(
        model, data, club_head_id
    )

    for step in range(num_steps):
        data.qpos[:] = trajectory_interp[step]

        if step < num_steps - 1:
            desired_vel = (trajectory_interp[step + 1] - trajectory_interp[step]) / dt
        else:
            desired_vel = np.zeros(model.nv)

        kp = 100.0
        kd = 20.0
        pos_error = trajectory_interp[step] - data.qpos
        vel_error = desired_vel - data.qvel
        ctrl = np.clip(kp * pos_error + kd * vel_error, -100.0, 100.0)
        data.ctrl[:] = ctrl[: model.nu]

        mujoco.mj_step(model, data)

        velocities[step] = data.qvel.copy()
        controls[step] = data.ctrl.copy()

        if club_head_id is not None:
            club_speeds.append(
                compute_club_speed(
                    model,
                    data,
                    club_head_id,
                    jacp,
                    jacr,
                    jacp_flat,
                    jacr_flat,
                    use_flat_jac,
                )
            )
            club_positions.append(data.xpos[club_head_id].copy())

    metrics = collect_simulation_metrics(
        model, club_speeds, club_positions, controls, velocities
    )
    return velocities, controls, metrics


def evaluate_objective(
    x: np.ndarray,
    model,
    data,
    objectives,
    club_head_id,
    swing_duration: float,
    num_knot_points: int,
) -> float:
    if not (x is not None):
        raise ValueError("x must be provided")
    from ._motion_opt_trajectory import compute_jerk

    trajectory = x.reshape(num_knot_points, model.nv)

    _, controls, metrics = simulate_trajectory(
        model, data, trajectory, club_head_id, swing_duration, num_knot_points
    )

    objective = 0.0

    if objectives.maximize_club_speed:
        objective -= objectives.weight_speed * metrics["peak_club_speed"]

    if objectives.minimize_energy:
        objective += objectives.weight_energy * metrics["total_energy"]

    if objectives.minimize_jerk:
        jerk = compute_jerk(trajectory, swing_duration, num_knot_points)
        objective += objectives.weight_jerk * jerk

    if objectives.minimize_torque:
        total_torque = np.sum(np.abs(controls))
        objective += objectives.weight_torque * total_torque

    if objectives.target_ball_position is not None:
        distance_error = float(
            np.linalg.norm(
                metrics["final_club_position"] - objectives.target_ball_position,
            ),
        )
        objective += objectives.weight_accuracy * distance_error

    return objective
