from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.contracts import require

from .plant import (
    OPTION_A_DOUBLE_PENDULUM_SPEC,
    TORQUE_LIMIT_NM,
    build_mujoco_model,
    linearize_step,
)
from .trace_contract import (
    FAILURE_DEFINITION,
    FAILURE_DEFINITION_ID,
    RECOVERY_THRESHOLD_RAD,
    TRACE_SCHEMA_VERSION,
    TRACKING_ERROR_BASELINE_RAD,
    payload_sha256,
    validate_trace_bundle,
)

HORIZON_S = 0.35
_PD_KP = np.array([55.0, 40.0])
_PD_KD = np.array([9.0, 7.0])


@dataclass(frozen=True)
class PerturbationTrial:
    impulse_nm_s: float
    direction: tuple[float, float]
    onset_fraction: float


def _perturbation_grid(seed: int) -> tuple[PerturbationTrial, ...]:
    require(seed >= 0, "seed must be non-negative")
    trials = [
        PerturbationTrial(float(m), (float(d[0]), float(d[1])), float(o))
        for m in (0.2, 9.0)
        for d in ((1.0, 0.0), (0.0, 1.0))
        for o in (0.18, 0.42)
    ]
    order = np.random.default_rng(seed).permutation(len(trials))
    return tuple(trials[int(i)] for i in order)


def _nominal_torque(t: float) -> np.ndarray:
    w = 2.0 * np.pi * 1.1 * t
    return np.array([7.0 * np.sin(w), 3.5 * np.sin(w + 0.4)], dtype=np.float64)


def _reference(model: Any, steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    import mujoco as mj

    data = mj.MjData(model)
    q = np.zeros((steps, 2))
    v = np.zeros((steps, 2))
    for k in range(steps):
        q[k], v[k] = data.qpos.copy(), data.qvel.copy()
        data.ctrl[:] = _nominal_torque(k * dt)
        mj.mj_step(model, data)
    return q, v


def run_swing_recovery_fixture(seed: int = 13837) -> dict[str, Any]:
    import mujoco as mj

    model = build_mujoco_model(OPTION_A_DOUBLE_PENDULUM_SPEC)
    dt = float(model.opt.timestep)
    steps = int(round(HORIZON_S / dt))
    q_ref, v_ref = _reference(model, steps, dt)
    rows_x, rows_u, rows_a, rows_b, trial_index = [], [], [], [], []
    outcomes, baseline_times, efforts, metadata = [], [], [], []

    for trial_id, trial in enumerate(_perturbation_grid(seed)):
        data = mj.MjData(model)
        onset = int(round(trial.onset_fraction * (steps - 1)))
        baseline_time, effort = float("nan"), 0.0
        for k in range(steps):
            q, vel = data.qpos.copy(), data.qvel.copy()
            if k == onset:
                data.qvel[:] = vel + trial.impulse_nm_s * np.array(trial.direction)
                vel = data.qvel.copy()
            u = np.clip(
                _PD_KP * (q_ref[k] - q) + _PD_KD * (v_ref[k] - vel),
                -TORQUE_LIMIT_NM,
                TORQUE_LIMIT_NM,
            )
            a, b = linearize_step(model, data, u)
            rows_x.append(np.concatenate([q, vel]))
            rows_u.append(u)
            rows_a.append(a)
            rows_b.append(b)
            trial_index.append(trial_id)
            err = float(np.linalg.norm(q - q_ref[k]))
            if np.isnan(baseline_time) and err > TRACKING_ERROR_BASELINE_RAD:
                baseline_time = k * dt
            effort += float(np.dot(u, u) * dt)
            data.ctrl[:] = u
            mj.mj_step(model, data)
        terminal = float(np.linalg.norm(data.qpos - q_ref[-1]))
        outcomes.append(
            "recovered"
            if np.isfinite(terminal) and terminal <= RECOVERY_THRESHOLD_RAD
            else "unrecoverable"
        )
        baseline_times.append(baseline_time)
        efforts.append(effort)
        metadata.append(
            {
                "impulse_nm_s": trial.impulse_nm_s,
                "direction_0": trial.direction[0],
                "direction_1": trial.direction[1],
                "onset_fraction": trial.onset_fraction,
            }
        )

    bundle: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "failure_definition": FAILURE_DEFINITION,
        "failure_definition_id": FAILURE_DEFINITION_ID,
        "seed": seed,
        "mujoco_version": mj.__version__,
        "timestep_s": dt,
        "torque_limit_nm": TORQUE_LIMIT_NM,
        "perturbation_metadata": metadata,
        "q_ref_k": q_ref,
        "v_ref_k": v_ref,
        "x_k": np.asarray(rows_x, dtype=np.float64),
        "u_k": np.asarray(rows_u, dtype=np.float64),
        "A_k": np.asarray(rows_a, dtype=np.float64),
        "B_k": np.asarray(rows_b, dtype=np.float64),
        "trial_index": np.asarray(trial_index, dtype=np.int32),
        "outcome_label": np.asarray(outcomes),
        "baseline_flag_time_s": np.asarray(baseline_times, dtype=np.float64),
        "control_effort": np.asarray(efforts, dtype=np.float64),
    }
    bundle["payload_sha256"] = payload_sha256(bundle)
    validate_trace_bundle(bundle)
    return bundle
