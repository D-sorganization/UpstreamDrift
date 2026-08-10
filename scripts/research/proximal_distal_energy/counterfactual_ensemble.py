"""Matched-state torque-killswitch ensembles for the double pendulum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.research.proximal_distal_energy.interaction_forces import (
    force_power_decomposition,
    reaction_force_decomposition,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    clubhead_speed,
)
from scripts.research.proximal_distal_energy.torque_programs import TorqueProgram
from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.protocol import SimState, Trace


@dataclass(frozen=True)
class KillswitchCase:
    """Two futures from one state plus directly comparable mechanical traces."""

    commanded: Trace
    zero_torque: Trace
    commanded_force: np.ndarray
    zero_torque_force: np.ndarray
    commanded_force_power: np.ndarray
    zero_torque_force_power: np.ndarray
    metrics: dict[str, float]


def controls_at_times(program: TorqueProgram, times_s: np.ndarray) -> np.ndarray:
    """Sample an open-loop program at absolute source-trace times."""
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must be a finite one-dimensional array")
    if np.any(times < 0.0):
        raise ValueError("times_s must be nonnegative")
    controls = np.empty((times.size, 2), dtype=float)
    controls[:, 0] = program.shoulder_torque_nm
    before = times < program.onset_s
    controls[before, 1] = -abs(program.wrist_restrain_nm)
    controls[~before, 1] = program.wrist_drive_nm
    return controls


def _accelerations(
    params: GolfModelParams, trace: Trace, controls: np.ndarray
) -> np.ndarray:
    backend = make_backend("ode", params)
    return np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(trace.q, trace.v, controls, strict=True)
        ]
    )


def _force_and_power(
    params: GolfModelParams,
    inertials: PlanarInertials,
    trace: Trace,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qdd = _accelerations(params, trace, controls)
    forces = reaction_force_decomposition(inertials, trace.q, trace.v, qdd)
    powers = force_power_decomposition(inertials, trace.q, trace.v, forces)
    return forces.total, powers.total


def _horizon_steps(horizon_s: float, dt_s: float) -> int:
    if not np.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive and finite")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be positive and finite")
    steps = int(round(horizon_s / dt_s))
    if steps <= 0 or not np.isclose(steps * dt_s, horizon_s, atol=1e-12):
        raise ValueError("horizon_s must be an integer multiple of dt_s")
    return steps


def evaluate_killswitch_case(
    params: GolfModelParams,
    inertials: PlanarInertials,
    program: TorqueProgram,
    *,
    source_time_s: float,
    source_q: np.ndarray,
    source_v: np.ndarray,
    horizon_s: float,
    dt_s: float,
) -> KillswitchCase:
    """Integrate and compare commanded and zero-torque matched-state futures."""
    q0 = np.asarray(source_q, dtype=float).reshape(-1)
    v0 = np.asarray(source_v, dtype=float).reshape(-1)
    if q0.shape != (2,) or v0.shape != (2,):
        raise ValueError("source_q and source_v must each have shape (2,)")
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(v0)):
        raise ValueError("source state must be finite")
    if not np.isfinite(source_time_s) or source_time_s < 0.0:
        raise ValueError("source_time_s must be nonnegative and finite")
    steps = _horizon_steps(horizon_s, dt_s)
    control_times = source_time_s + np.arange(steps, dtype=float) * dt_s
    controls = controls_at_times(program, control_times)
    aligned_controls = np.vstack((controls, controls[-1]))
    zero_controls = np.zeros_like(aligned_controls)
    state = SimState(q=q0, v=v0, time=source_time_s)

    commanded_backend = make_backend("ode", params)
    commanded_backend.reset(state)
    commanded = commanded_backend.rollout(controls, steps, dt_s)
    zero_backend = make_backend("ode", params)
    zero_backend.reset(state)
    zero_torque = zero_backend.rollout(None, steps, dt_s)

    commanded_force, commanded_power = _force_and_power(
        params, inertials, commanded, aligned_controls
    )
    zero_force, zero_power = _force_and_power(
        params, inertials, zero_torque, zero_controls
    )
    commanded_speed = clubhead_speed(inertials, commanded.q, commanded.v)
    zero_speed = clubhead_speed(inertials, zero_torque.q, zero_torque.v)
    q_delta = commanded.q[-1] - zero_torque.q[-1]
    v_delta = commanded.v[-1] - zero_torque.v[-1]
    normalized_state = np.concatenate((q_delta / 1.0, v_delta / 1.0))
    metrics = {
        "matched_initial_state_error": float(
            np.linalg.norm(commanded.q[0] - zero_torque.q[0])
            + np.linalg.norm(commanded.v[0] - zero_torque.v[0])
        ),
        "terminal_q_distance_rad": float(np.linalg.norm(q_delta)),
        "terminal_v_distance_rad_s": float(np.linalg.norm(v_delta)),
        "terminal_state_distance": float(np.linalg.norm(normalized_state)),
        "terminal_force_distance_n": float(
            np.linalg.norm(commanded_force[-1] - zero_force[-1])
        ),
        "terminal_power_difference_w": float(commanded_power[-1] - zero_power[-1]),
        "force_work_difference_j": float(
            np.trapezoid(commanded_power - zero_power, commanded.t)
        ),
        "terminal_clubhead_speed_difference_m_s": float(
            commanded_speed[-1] - zero_speed[-1]
        ),
        "initial_commanded_force_n": float(np.linalg.norm(commanded_force[0])),
        "initial_pointwise_ztcf_force_n": float(np.linalg.norm(zero_force[0])),
        "initial_commanded_force_power_w": float(commanded_power[0]),
        "initial_pointwise_ztcf_force_power_w": float(zero_power[0]),
    }
    return KillswitchCase(
        commanded=commanded,
        zero_torque=zero_torque,
        commanded_force=commanded_force,
        zero_torque_force=zero_force,
        commanded_force_power=commanded_power,
        zero_torque_force_power=zero_power,
        metrics=metrics,
    )


def evaluate_killswitch_ensemble(
    params: GolfModelParams,
    inertials: PlanarInertials,
    program: TorqueProgram,
    t: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    *,
    cut_times_s: tuple[float, ...],
    horizons_s: tuple[float, ...],
    timesteps_s: tuple[float, ...],
) -> list[dict[str, float]]:
    """Evaluate the Cartesian product of cut time, horizon, and timestep."""
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    if t_arr.ndim != 1 or q_arr.shape != (t_arr.size, 2) or v_arr.shape != q_arr.shape:
        raise ValueError("source trace must have t=(T,), q=v=(T, 2)")
    if not cut_times_s or not horizons_s or not timesteps_s:
        raise ValueError("cut-time, horizon, and timestep grids must be nonempty")
    if min(cut_times_s) < t_arr[0] or max(cut_times_s) > t_arr[-1]:
        raise ValueError("cut times must lie within the source trace")

    rows: list[dict[str, float]] = []
    for cut_time in cut_times_s:
        source_q = np.array([np.interp(cut_time, t_arr, q_arr[:, j]) for j in range(2)])
        source_v = np.array([np.interp(cut_time, t_arr, v_arr[:, j]) for j in range(2)])
        for horizon in horizons_s:
            for timestep in timesteps_s:
                case = evaluate_killswitch_case(
                    params,
                    inertials,
                    program,
                    source_time_s=cut_time,
                    source_q=source_q,
                    source_v=source_v,
                    horizon_s=horizon,
                    dt_s=timestep,
                )
                rows.append(
                    {
                        "cut_time_s": float(cut_time),
                        "horizon_s": float(horizon),
                        "dt_s": float(timestep),
                        **case.metrics,
                    }
                )
    return rows
