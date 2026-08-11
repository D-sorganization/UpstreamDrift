"""Generate coupled moving-base and flexible-club evidence.

The output keeps base motion, two-hand constraint forces, direct wrist torque,
shaft storage, shaft damping, and numerical projection terms separate.  Every
counterfactual is a new forward rollout; no measured or prescribed kinematics
enter the study.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.research.proximal_distal_energy.mechanism_ladder import (
    embed_planar_sample,
)
from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleConfig,
    MovingBaseFlexibleParams,
    MovingBaseFlexibleTrace,
    initial_state,
    rollout,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
SCHEMA_VERSION = "moving-base-flexible-club-evidence-v1"
STUDY_ID = "coupled-moving-base-flexible-club-v1"
BASELINE_DURATION_S = 0.4
BASELINE_STEP_S = 0.0005
BRANCH_CUT_TIME_S = 0.2
BRANCH_DURATION_S = 0.05


def _source_hashes() -> dict[str, str]:
    relative = (
        "scripts/research/proximal_distal_energy/moving_base_flexible_club.py",
        "scripts/research/proximal_distal_energy/run_moving_base_flexible_study.py",
        "scripts/research/proximal_distal_energy/two_arm_closed_loop.py",
    )
    return {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in relative
    }


def command(time_s: float, _q: np.ndarray, _qdot: np.ndarray) -> TwoArmControl:
    """Return the declared smooth open-loop torque intervention."""
    phase = float(np.clip(time_s / BASELINE_DURATION_S, 0.0, 1.0))
    return TwoArmControl(
        right_shoulder_nm=18.0 + 4.0 * phase,
        right_elbow_nm=7.0 - 1.5 * phase,
        right_wrist_nm=-3.0 + 2.0 * phase,
        left_shoulder_nm=16.0 + 3.0 * phase,
        left_elbow_nm=6.0 - phase,
        left_wrist_nm=2.0 - phase,
    )


def _zero(_time_s: float, _q: np.ndarray, _qdot: np.ndarray) -> TwoArmControl:
    return TwoArmControl.zero()


@cache
def _run(
    params: MovingBaseFlexibleParams,
    duration_s: float,
    step_s: float,
) -> MovingBaseFlexibleTrace:
    q0, qdot0 = initial_state(params)
    return rollout(
        q0,
        qdot0,
        command,
        params,
        MovingBaseFlexibleConfig(duration_s=duration_s, step_s=step_s),
    )


def _first_negative_time(time: np.ndarray, values: np.ndarray) -> float | None:
    crossings = np.flatnonzero((values[:-1] >= 0.0) & (values[1:] < 0.0))
    return None if crossings.size == 0 else float(time[int(crossings[0]) + 1])


def _negative_persistence(time: np.ndarray, values: np.ndarray) -> float:
    if values[0] >= 0.0:
        return 0.0
    nonnegative = np.flatnonzero(values >= 0.0)
    end = int(nonnegative[0]) if nonnegative.size else values.size - 1
    return float(time[end] - time[0])


def _closure(trace: MovingBaseFlexibleTrace) -> dict[str, float | int]:
    supplied_power = trace.applied_control_power_w + trace.dissipation_power_w
    work = float(np.trapezoid(supplied_power, x=trace.time))
    energy_change = float(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0])
    return {
        "constraint_rank_min": 4,
        "position_constraint_max_m": float(np.max(trace.position_constraint_norm_m)),
        "velocity_constraint_max_m_s": float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
        "kkt_residual_max": float(np.max(trace.kkt_residual_norm)),
        "acceleration_constraint_residual_max": float(
            np.max(trace.acceleration_constraint_residual_norm)
        ),
        "contact_power_identity_max_w": float(
            np.max(np.abs(trace.contact_power_identity_residual_w))
        ),
        "mechanical_energy_change_j": energy_change,
        "applied_control_work_j": float(
            np.trapezoid(trace.applied_control_power_w, x=trace.time)
        ),
        "dissipated_work_j": float(
            np.trapezoid(trace.dissipation_power_w, x=trace.time)
        ),
        "net_nonconstraint_work_j": work,
        "work_energy_residual_abs_j": abs(energy_change - work),
        "projection_energy_change_sum_j": float(
            np.sum(trace.projection_energy_change_j)
        ),
        "projection_correction_max_m": float(
            np.max(trace.projection_correction_norm_m)
        ),
    }


def _summary(trace: MovingBaseFlexibleTrace) -> dict[str, Any]:
    speed = np.linalg.norm(trace.clubhead_velocity_m_s, axis=1)
    couple = trace.force_generated_couple_nm
    return {
        "duration_s": float(trace.time[-1] - trace.time[0]),
        "step_s": float(trace.time[1] - trace.time[0]),
        "maximum_base_displacement_m": float(
            np.max(np.linalg.norm(trace.q[:, 4:6], axis=1))
        ),
        "maximum_abs_shaft_flex_deg": float(np.max(np.abs(np.rad2deg(trace.q[:, 9])))),
        "peak_shaft_strain_energy_j": float(np.max(trace.shaft_strain_energy_j)),
        "peak_clubhead_speed_m_s": float(np.max(speed)),
        "force_generated_couple": {
            "minimum_nm": float(np.min(couple)),
            "maximum_nm": float(np.max(couple)),
            "first_negative_time_s": _first_negative_time(trace.time, couple),
        },
        "direct_wrist_torque": {
            "minimum_nm": float(np.min(trace.direct_wrist_torque_nm)),
            "maximum_nm": float(np.max(trace.direct_wrist_torque_nm)),
        },
        "maximum_abs_shaft_elastic_moment_nm": float(
            np.max(np.abs(trace.shaft_elastic_moment_nm))
        ),
        "maximum_abs_shaft_damping_moment_nm": float(
            np.max(np.abs(trace.shaft_damping_moment_nm))
        ),
        "closure": _closure(trace),
    }


def _trace_arrays(prefix: str, trace: MovingBaseFlexibleTrace) -> dict[str, np.ndarray]:
    fields = (
        "q",
        "qdot",
        "qddot",
        "multipliers_n",
        "contact_force_on_club_n",
        "force_generated_couple_nm",
        "direct_wrist_torque_nm",
        "contact_power_w",
        "contact_wrench_power_w",
        "contact_power_identity_residual_w",
        "shaft_elastic_moment_nm",
        "shaft_damping_moment_nm",
        "shaft_strain_energy_j",
        "clubhead_position_m",
        "clubhead_velocity_m_s",
        "mechanical_energy_j",
        "applied_control_power_w",
        "dissipation_power_w",
        "position_constraint_norm_m",
        "velocity_constraint_norm_m_s",
        "kkt_residual_norm",
        "acceleration_constraint_residual_norm",
        "projection_correction_norm_m",
        "projection_energy_change_j",
    )
    result = {f"{prefix}_time_s": trace.time}
    result.update(
        {f"{prefix}_{name}": np.asarray(getattr(trace, name)) for name in fields}
    )
    return result


def _wrench_records(trace: MovingBaseFlexibleTrace) -> list[dict[str, object]]:
    records = []
    for target_time in (0.0, 0.1, 0.2, 0.25, 0.3, 0.4):
        index = int(np.argmin(np.abs(trace.time - target_time)))
        force = np.sum(trace.contact_force_on_club_n[index], axis=0)
        sample = embed_planar_sample(
            model_tier=trace.model_tier,
            time_s=float(trace.time[index]),
            reference_point_xy_m=trace.q[index, 6:8],
            force_xy_n=force,
            couple_z_nm=float(
                trace.force_generated_couple_nm[index]
                + trace.direct_wrist_torque_nm[index]
            ),
            linear_velocity_xy_m_s=trace.qdot[index, 6:8],
            angular_velocity_z_rad_s=float(trace.qdot[index, 8]),
        )
        record = sample.as_record()
        record["force_generated_couple_nm"] = float(
            trace.force_generated_couple_nm[index]
        )
        record["direct_wrist_torque_nm"] = float(trace.direct_wrist_torque_nm[index])
        record["shaft_flex_rad"] = float(trace.q[index, 9])
        records.append(record)
    return records


def _replace_sensitivity_parameter(
    params: MovingBaseFlexibleParams, parameter: str, value: float
) -> MovingBaseFlexibleParams:
    """Apply one registered scalar sensitivity without an untyped field map."""
    if parameter == "base_stiffness_n_m":
        return replace(params, base_stiffness_n_m=value)
    if parameter == "shaft_stiffness_nm_rad":
        return replace(params, shaft_stiffness_nm_rad=value)
    if parameter == "shaft_damping_nms_rad":
        return replace(params, shaft_damping_nms_rad=value)
    raise ValueError(f"unsupported sensitivity parameter: {parameter}")


def run_study() -> tuple[dict, dict[str, np.ndarray]]:
    """Execute all declared trajectories and return portable evidence."""
    reference = MovingBaseFlexibleParams.publication_default()
    baseline = _run(reference, BASELINE_DURATION_S, BASELINE_STEP_S)
    cut_index = int(np.argmin(np.abs(baseline.time - BRANCH_CUT_TIME_S)))
    branch = rollout(
        baseline.q[cut_index],
        baseline.qdot[cut_index],
        _zero,
        reference,
        MovingBaseFlexibleConfig(
            duration_s=BRANCH_DURATION_S,
            step_s=BASELINE_STEP_S,
            start_time_s=float(baseline.time[cut_index]),
        ),
    )
    coincident = replace(reference, right_grip_offset_m=0.0, left_grip_offset_m=0.0)
    coincident_trace = _run(coincident, 0.05, 0.001)

    sensitivity_declarations = (
        ("base_stiffness_n_m", 12000.0),
        ("base_stiffness_n_m", 48000.0),
        ("shaft_stiffness_nm_rad", 40.0),
        ("shaft_stiffness_nm_rad", 160.0),
        ("shaft_damping_nms_rad", 0.0),
    )
    sensitivity = []
    for parameter, value in sensitivity_declarations:
        params = _replace_sensitivity_parameter(reference, parameter, value)
        trace = _run(params, BASELINE_DURATION_S, 0.001)
        summary = _summary(trace)
        sensitivity.append(
            {
                "parameter": parameter,
                "value": value,
                "minimum_force_generated_couple_nm": summary["force_generated_couple"][
                    "minimum_nm"
                ],
                "first_negative_time_s": summary["force_generated_couple"][
                    "first_negative_time_s"
                ],
                "maximum_base_displacement_m": summary["maximum_base_displacement_m"],
                "maximum_abs_shaft_flex_deg": summary["maximum_abs_shaft_flex_deg"],
                "peak_shaft_strain_energy_j": summary["peak_shaft_strain_energy_j"],
                "peak_clubhead_speed_m_s": summary["peak_clubhead_speed_m_s"],
            }
        )

    convergence = []
    for step_s in (0.002, 0.001, 0.0005):
        trace = _run(reference, BASELINE_DURATION_S, step_s)
        summary = _summary(trace)
        convergence.append(
            {
                "step_s": step_s,
                "minimum_force_generated_couple_nm": summary["force_generated_couple"][
                    "minimum_nm"
                ],
                "first_negative_time_s": summary["force_generated_couple"][
                    "first_negative_time_s"
                ],
                "work_energy_residual_abs_j": summary["closure"][
                    "work_energy_residual_abs_j"
                ],
                "projection_correction_max_m": summary["closure"][
                    "projection_correction_max_m"
                ],
            }
        )

    arrays = {
        **_trace_arrays("baseline", baseline),
        **_trace_arrays("branch", branch),
        **_trace_arrays("coincident_grip", coincident_trace),
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "trajectory_kind": "forward_constrained_dynamics",
        "base_motion_prescribed": False,
        "shaft_flex_prescribed": False,
        "human_validation": False,
        "physiological_evidence": False,
        "source_files": _source_hashes(),
        "model_boundary": (
            "Planar finite-mass translating base, two two-link arms, two ideal "
            "point contacts, and one lumped torsional club-flex mode. Parameters "
            "are declared mechanism-study values, not subject calibration."
        ),
        "reference_parameters": json.loads(json.dumps(asdict(reference))),
        "baseline": _summary(baseline),
        "zero_command_branch": {
            "cut_index": cut_index,
            "cut_time_s": float(baseline.time[cut_index]),
            "initial_state_exactly_inherited": bool(
                np.array_equal(branch.q[0], baseline.q[cut_index])
                and np.array_equal(branch.qdot[0], baseline.qdot[cut_index])
            ),
            "initial_force_generated_couple_nm": float(
                branch.force_generated_couple_nm[0]
            ),
            "minimum_force_generated_couple_nm": float(
                np.min(branch.force_generated_couple_nm)
            ),
            "negative_persistence_s": _negative_persistence(
                branch.time, branch.force_generated_couple_nm
            ),
            "closure": _closure(branch),
        },
        "coincident_grip_negative_control": {
            "right_grip_offset_m": 0.0,
            "left_grip_offset_m": 0.0,
            "maximum_abs_force_generated_couple_nm": float(
                np.max(np.abs(coincident_trace.force_generated_couple_nm))
            ),
        },
        "mechanism_sensitivity": sensitivity,
        "timestep_convergence": convergence,
        "common_wrench_samples": _wrench_records(baseline),
        "falsification_tests": [
            {
                "claim": "late negative force-generated couple survives coupled base and flex dynamics",
                "failure_condition": "no sign reversal under the declared reference trajectory",
            },
            {
                "claim": "the negative couple can persist without continued command",
                "failure_condition": "same-state zero-command branch is nonnegative immediately or within 45 ms",
            },
            {
                "claim": "hand separation supplies the geometric moment arm",
                "failure_condition": "coincident grip contacts retain nonzero force-generated couple",
            },
            {
                "claim": "base motion and shaft flex are endogenous",
                "failure_condition": "either coordinate remains identically zero under nonzero coupled loading",
            },
            {
                "claim": "reported transfer respects ideal-contact power equivalence",
                "failure_condition": "two-point and transported-wrench powers differ beyond numerical tolerance",
            },
            {
                "claim": "numerical conclusions are resolution-stable",
                "failure_condition": "timing, sign, or energy closure fails to converge under timestep refinement",
            },
        ],
    }
    return record, arrays


def write_study(
    output_dir: Path,
    *,
    record: dict[str, Any] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path]:
    """Write deterministic JSON and compressed numerical traces."""
    if record is None or arrays is None:
        record, arrays = run_study()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "moving_base_flexible_study.json"
    npz_path = output_dir / "moving_base_flexible_study.npz"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    record, arrays = run_study()
    write_study(DATA_DIR, record=record, arrays=arrays)


if __name__ == "__main__":
    main()
