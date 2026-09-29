"""Generate deterministic evidence for the forward constrained two-hand model.

The study separates the force-generated club couple from directly applied
wrist torque. It also branches zero-command counterfactuals from achieved
states, records solver closure, and exposes timestep and projection
sensitivity. The output is model evidence, not physiological validation.
"""

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.forward_two_arm import (
    ForwardTwoArmConfig,
    ForwardTwoArmTrace,
    branch_zero_command,
    rollout_forward_two_arm,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
    contact_wrench,
    constraint_acceleration_bias_audit,
    constraint_jacobian,
    control_generalized_force,
    decompose_contact_forces,
    kinematics,
)

FloatArray = npt.NDArray[np.float64]
SCHEMA_VERSION = "forward-two-arm-evidence-v1"
BASELINE_DURATION_S = 0.4
BASELINE_STEP_S = 0.00025
REPRESENTATIVE_CUT_TIME_S = 0.2
KILLSWITCH_HORIZON_S = 0.05


def _command(time_s: float, _q: FloatArray, _qdot: FloatArray) -> TwoArmControl:
    """Return the declared smooth, open-loop joint-torque intervention."""
    phase = float(np.clip(time_s / BASELINE_DURATION_S, 0.0, 1.0))
    return TwoArmControl(
        right_shoulder_nm=18.0 + 4.0 * phase,
        right_elbow_nm=7.0 - 1.5 * phase,
        right_wrist_nm=-3.0 + 2.0 * phase,
        left_shoulder_nm=16.0 + 3.0 * phase,
        left_elbow_nm=6.0 - phase,
        left_wrist_nm=2.0 - phase,
    )


def _initial_state(params: TwoArmParams) -> tuple[FloatArray, FloatArray]:
    configuration = params.consistent_configuration(np.array([0.0, -0.5]), 0.16)
    return configuration, np.zeros(7, dtype=float)


def _grip_velocities(
    state: FloatArray, velocity: FloatArray, params: TwoArmParams
) -> tuple[FloatArray, FloatArray]:
    club_derivative = np.array([np.cos(state[6]), np.sin(state[6])])
    center_velocity = velocity[4:6]
    angular_velocity = velocity[6]
    return (
        center_velocity
        + params.right_grip_offset_m * angular_velocity * club_derivative,
        center_velocity
        + params.left_grip_offset_m * angular_velocity * club_derivative,
    )


def _trace_observables(
    trace: ForwardTwoArmTrace, params: TwoArmParams
) -> dict[str, FloatArray]:
    samples = trace.time.size
    resultant = np.empty((samples, 2))
    differential = np.empty((samples, 2))
    force_couple = np.empty(samples)
    contact_power = np.empty(samples)
    control_power = np.empty(samples)
    wrist_torque = np.empty(samples)
    wrench_power = np.empty(samples)
    constraint_power = np.empty(samples)
    for index, (state, velocity, forces, control) in enumerate(
        zip(
            trace.q,
            trace.qdot,
            trace.contact_force_on_club_n,
            trace.controls,
            strict=True,
        )
    ):
        points = kinematics(state, params)
        right_velocity, left_velocity = _grip_velocities(state, velocity, params)
        wrench = contact_wrench(
            forces[0],
            forces[1],
            points["right_grip"],
            points["left_grip"],
            points["club_center"],
            right_velocity,
            left_velocity,
        )
        modes = decompose_contact_forces(forces[0], forces[1])
        resultant[index] = modes.resultant_n
        differential[index] = modes.differential_n
        force_couple[index] = wrench.moment_about_center_nm
        contact_power[index] = wrench.contact_power_w
        control_power[index] = control_generalized_force(control) @ velocity
        wrist_torque[index] = control.right_wrist_nm + control.left_wrist_nm
        wrench_power[index] = (
            wrench.resultant_force_n @ velocity[4:6]
            + wrench.moment_about_center_nm * velocity[6]
        )
        constraint_power[index] = trace.multipliers_n[index] @ (
            constraint_jacobian(state, params) @ velocity
        )
    return {
        "resultant_contact_force_n": resultant,
        "differential_contact_force_n": differential,
        "force_generated_couple_nm": force_couple,
        "contact_power_w": contact_power,
        "wrench_power_w": wrench_power,
        "constraint_two_sided_power_residual_w": constraint_power,
        "control_power_w": control_power,
        "direct_wrist_torque_nm": wrist_torque,
    }


def _first_negative_time(time: FloatArray, values: FloatArray) -> float | None:
    if values[0] < 0.0:
        return float(time[0])
    crossings = np.flatnonzero((values[:-1] >= 0.0) & (values[1:] < 0.0))
    if crossings.size == 0:
        return None
    return float(time[int(crossings[0]) + 1])


def _negative_persistence_s(time: FloatArray, values: FloatArray) -> float:
    if values[0] >= 0.0:
        return 0.0
    nonnegative = np.flatnonzero(values >= 0.0)
    end = int(nonnegative[0]) if nonnegative.size else values.size - 1
    return float(time[end] - time[0])


def _closure_record(
    trace: ForwardTwoArmTrace, observables: dict[str, FloatArray]
) -> dict[str, float | int]:
    control_work = float(np.trapezoid(observables["control_power_w"], x=trace.time))
    energy_change = float(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0])
    return {
        "constraint_rank_min": int(np.min(trace.constraint_rank)),
        "position_constraint_max_m": float(np.max(trace.position_constraint_norm_m)),
        "velocity_constraint_max_m_s": float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
        "kkt_residual_max": float(np.max(trace.kkt_residual_norm)),
        "acceleration_constraint_residual_max": float(
            np.max(trace.acceleration_constraint_residual_norm)
        ),
        "mechanical_energy_change_j": energy_change,
        "applied_control_work_j": control_work,
        "work_energy_residual_abs_j": abs(energy_change - control_work),
        "projection_energy_change_sum_j": float(
            np.sum(trace.projection_energy_change_j)
        ),
        "projection_energy_change_absolute_sum_j": float(
            np.sum(np.abs(trace.projection_energy_change_j))
        ),
        "projection_energy_change_max_abs_j": float(
            np.max(np.abs(trace.projection_energy_change_j))
        ),
        "projection_correction_max_m": trace.maximum_projection_correction_m,
        "contact_wrench_power_equivalence_max_w": float(
            np.max(
                np.abs(observables["contact_power_w"] - observables["wrench_power_w"])
            )
        ),
        "constraint_two_sided_power_residual_max_w": float(
            np.max(np.abs(observables["constraint_two_sided_power_residual_w"]))
        ),
    }


def _trace_arrays(
    prefix: str,
    trace: ForwardTwoArmTrace,
    observables: dict[str, FloatArray],
) -> dict[str, FloatArray]:
    arrays: dict[str, FloatArray] = {
        f"{prefix}_time_s": trace.time,
        f"{prefix}_q": trace.q,
        f"{prefix}_qdot": trace.qdot,
        f"{prefix}_qddot": trace.qddot,
        f"{prefix}_multipliers_n": trace.multipliers_n,
        f"{prefix}_contact_force_on_club_n": trace.contact_force_on_club_n,
        f"{prefix}_mechanical_energy_j": trace.mechanical_energy_j,
        f"{prefix}_position_constraint_norm_m": (trace.position_constraint_norm_m),
        f"{prefix}_velocity_constraint_norm_m_s": (trace.velocity_constraint_norm_m_s),
        f"{prefix}_kkt_residual_norm": trace.kkt_residual_norm,
        f"{prefix}_acceleration_constraint_residual_norm": (
            trace.acceleration_constraint_residual_norm
        ),
        f"{prefix}_projection_correction_norm_m": (trace.projection_correction_norm_m),
        f"{prefix}_projection_energy_change_j": trace.projection_energy_change_j,
    }
    arrays.update({f"{prefix}_{name}": value for name, value in observables.items()})
    return arrays


def _rollout(
    params: TwoArmParams,
    *,
    step_s: float,
    projection_tolerance_m: float = 1e-10,
    duration_s: float = BASELINE_DURATION_S,
) -> ForwardTwoArmTrace:
    q0, qdot0 = _initial_state(params)
    return rollout_forward_two_arm(
        q0,
        qdot0,
        _command,
        params,
        ForwardTwoArmConfig(
            duration_s=duration_s,
            step_s=step_s,
            projection_tolerance_m=projection_tolerance_m,
        ),
    )


def _sensitivity_row(
    params: TwoArmParams,
    *,
    step_s: float,
    projection_tolerance_m: float,
) -> dict[str, float | None]:
    trace = _rollout(
        params,
        step_s=step_s,
        projection_tolerance_m=projection_tolerance_m,
    )
    observables = _trace_observables(trace, params)
    couple = observables["force_generated_couple_nm"]
    closure = _closure_record(trace, observables)
    return {
        "step_s": step_s,
        "projection_tolerance_m": projection_tolerance_m,
        "first_negative_time_s": _first_negative_time(trace.time, couple),
        "minimum_force_generated_couple_nm": float(np.min(couple)),
        "position_constraint_max_m": float(np.max(trace.position_constraint_norm_m)),
        "projection_correction_max_m": trace.maximum_projection_correction_m,
        "projection_energy_change_sum_j": closure["projection_energy_change_sum_j"],
        "work_energy_residual_abs_j": closure["work_energy_residual_abs_j"],
    }


def _parameters_record(params: TwoArmParams) -> dict[str, Any]:
    values = asdict(params)
    values["right_shoulder_m"] = list(params.right_shoulder_m)
    values["left_shoulder_m"] = list(params.left_shoulder_m)
    return values


def _source_hashes() -> dict[str, str]:
    source_directory = Path(__file__).resolve().parent
    repository_root = source_directory.parents[2]
    paths = (
        source_directory / "forward_two_arm.py",
        source_directory / "run_forward_two_arm_study.py",
        source_directory / "two_arm_closed_loop.py",
    )
    return {
        path.relative_to(repository_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }


def _run_study_uncached() -> tuple[dict[str, Any], dict[str, FloatArray]]:
    params = TwoArmParams.publication_default()
    baseline = _rollout(params, step_s=BASELINE_STEP_S)
    baseline_observables = _trace_observables(baseline, params)
    baseline_couple = baseline_observables["force_generated_couple_nm"]

    cut_index = int(round(REPRESENTATIVE_CUT_TIME_S / BASELINE_STEP_S))
    branch = branch_zero_command(
        baseline,
        cut_index=cut_index,
        horizon_s=KILLSWITCH_HORIZON_S,
        params=params,
    )
    branch_observables = _trace_observables(branch, params)
    branch_couple = branch_observables["force_generated_couple_nm"]

    killswitch_rows: list[dict[str, float | int | None]] = []
    for cut_time_s in (0.18, 0.19, 0.2, 0.21, 0.22, 0.24):
        index = int(round(cut_time_s / BASELINE_STEP_S))
        candidate = branch_zero_command(
            baseline,
            cut_index=index,
            horizon_s=KILLSWITCH_HORIZON_S,
            params=params,
        )
        candidate_observables = _trace_observables(candidate, params)
        candidate_couple = candidate_observables["force_generated_couple_nm"]
        killswitch_rows.append(
            {
                "cut_index": index,
                "cut_time_s": float(baseline.time[index]),
                "first_negative_time_s": _first_negative_time(
                    candidate.time, candidate_couple
                ),
                "negative_persistence_s": _negative_persistence_s(
                    candidate.time, candidate_couple
                ),
                "minimum_force_generated_couple_nm": float(np.min(candidate_couple)),
            }
        )

    zero_arm_params = replace(
        params,
        right_grip_offset_m=0.0,
        left_grip_offset_m=0.0,
    )
    zero_arm = _rollout(zero_arm_params, step_s=0.001, duration_s=0.05)
    zero_arm_observables = _trace_observables(zero_arm, zero_arm_params)

    timestep_convergence = [
        _sensitivity_row(
            params,
            step_s=step_s,
            projection_tolerance_m=1e-10,
        )
        for step_s in (0.002, 0.001, 0.0005)
    ]
    projection_sensitivity = [
        _sensitivity_row(
            params,
            step_s=0.001,
            projection_tolerance_m=tolerance,
        )
        for tolerance in (1e-8, 1e-10)
    ]

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "forward-constrained-two-hand-v1",
        "trajectory_kind": "forward_constrained_dynamics",
        "integrator": "velocity-verlet-with-mass-metric-rattle-projection",
        "prescribed_kinematics": False,
        "physiological_evidence": False,
        "human_validation": False,
        "source_files": _source_hashes(),
        "claim_boundary": (
            "The result demonstrates mechanically feasible zero-command persistence of "
            "a force-generated negative club couple in this planar constrained model. "
            "Zero command is not biological passivity; the result does not identify "
            "muscle strategy, prove human use, or establish optimal coaching advice."
        ),
        "parameters": _parameters_record(params),
        "command_intervention": {
            "kind": "smooth_open_loop_joint_torque",
            "duration_s": BASELINE_DURATION_S,
            "step_s": BASELINE_STEP_S,
            "definition": {
                "right_shoulder_nm": "18 + 4 t/0.4",
                "right_elbow_nm": "7 - 1.5 t/0.4",
                "right_wrist_nm": "-3 + 2 t/0.4",
                "left_shoulder_nm": "16 + 3 t/0.4",
                "left_elbow_nm": "6 - t/0.4",
                "left_wrist_nm": "2 - t/0.4",
            },
        },
        "baseline": {
            "sample_count": int(baseline.time.size),
            "closure": _closure_record(baseline, baseline_observables),
            "force_generated_couple": {
                "definition": (
                    "sum of grip-force moments about the declared club center; "
                    "direct wrist torque excluded"
                ),
                "minimum_nm": float(np.min(baseline_couple)),
                "maximum_nm": float(np.max(baseline_couple)),
                "first_negative_time_s": _first_negative_time(
                    baseline.time, baseline_couple
                ),
            },
            "direct_wrist_torque": {
                "minimum_nm": float(
                    np.min(baseline_observables["direct_wrist_torque_nm"])
                ),
                "maximum_nm": float(
                    np.max(baseline_observables["direct_wrist_torque_nm"])
                ),
            },
        },
        "constraint_acceleration_bias_audit": {
            "method": "exact centripetal expression versus five-point centered directional derivative of J",
            "tolerance_m_s2": 1.0e-7,
            "maximum_residual_m_s2": float(
                max(
                    constraint_acceleration_bias_audit(q, qdot, params)
                    for q, qdot in zip(baseline.q, baseline.qdot, strict=True)
                )
            ),
        },
        "representative_killswitch": {
            "cut_index": cut_index,
            "cut_time_s": float(baseline.time[cut_index]),
            "horizon_s": KILLSWITCH_HORIZON_S,
            "initial_state_exactly_inherited": bool(
                np.array_equal(branch.q[0], baseline.q[cut_index])
                and np.array_equal(branch.qdot[0], baseline.qdot[cut_index])
            ),
            "applied_command_after_cut": "zero",
            "initial_force_generated_couple_nm": float(branch_couple[0]),
            "negative_persistence_s": _negative_persistence_s(
                branch.time, branch_couple
            ),
            "minimum_force_generated_couple_nm": float(np.min(branch_couple)),
        },
        "killswitch_ensemble": killswitch_rows,
        "zero_contact_moment_arm_control": {
            "intervention": (
                "set both grip offsets to zero and recompute a consistent state"
            ),
            "maximum_abs_force_generated_couple_nm": float(
                np.max(np.abs(zero_arm_observables["force_generated_couple_nm"]))
            ),
        },
        "timestep_convergence": timestep_convergence,
        "projection_sensitivity": projection_sensitivity,
        "falsifiers": [
            "constraint rank below four or closure residual above tolerance",
            "work-energy residual that does not converge with timestep",
            "negative-couple onset that is not stable to timestep refinement",
            "nonzero force-generated couple after both grip moment arms are removed",
            "loss of negative-couple persistence immediately after zero-command branching",
        ],
    }
    arrays = _trace_arrays("baseline", baseline, baseline_observables)
    arrays.update(_trace_arrays("branch", branch, branch_observables))
    return record, arrays


@lru_cache(maxsize=1)
def _cached_study() -> tuple[dict[str, Any], dict[str, FloatArray]]:
    return _run_study_uncached()


def run_study() -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Execute the deterministic Phase 1 study and return detached results."""
    record, arrays = _cached_study()
    return deepcopy(record), {name: value.copy() for name, value in arrays.items()}


def write_study(
    output_directory: str | Path,
    *,
    record: dict[str, Any],
    arrays: dict[str, FloatArray],
) -> tuple[Path, Path]:
    """Write deterministic JSON metadata and compressed numerical arrays."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "forward_two_arm_study.json"
    npz_path = destination / "forward_two_arm_study.npz"
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload: dict[str, Any] = dict(sorted(arrays.items()))
    np.savez_compressed(npz_path, **payload)
    return json_path, npz_path


if __name__ == "__main__":
    evidence_record, evidence_arrays = run_study()
    write_study(
        Path("docs/research/proximal_distal_energy_transfer/data"),
        record=evidence_record,
        arrays=evidence_arrays,
    )


__all__ = ["run_study", "write_study"]
