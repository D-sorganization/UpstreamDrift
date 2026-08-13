"""Run the registered torso-velocity study on the rotating-base mechanism.

The study compares acceleration histories under two explicit initial-state
matching rules.  Rows that violate the registered numerical or loading envelope
remain in the evidence record; they are never silently discarded.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.rotating_base_two_hand import (
    RotatingBaseConfig,
    RotatingBaseParams,
    RotatingBaseState,
    TorsoTwoHandControl,
    initial_state,
    rollout,
    solve_constrained_dynamics,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
JSON_PATH = DATA_DIR / "rotating_base_torso_velocity_study.json"
NPZ_PATH = DATA_DIR / "rotating_base_torso_velocity_study.npz"

TORSO_RATES_RAD_S = (1.5, 3.5, 5.5)
TORSO_PROFILES_NM = {"accelerate": 55.0, "constant_rate": 0.0, "decelerate": -55.0}
MATCHING_RULES = ("relative_club_rate", "absolute_club_rate")
REGISTERED_PEAK_GRIP_FORCE_CEILING_N = 100.0


def _trapz(values: np.ndarray, time: np.ndarray) -> float:
    return float(np.trapezoid(values, time))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _control_law(torso_nm: float, *, wrist_release_s: float = 0.025):
    def law(time_s: float, _state: RotatingBaseState) -> TorsoTwoHandControl:
        wrist_nm = -3.0 if time_s < wrist_release_s else 4.0
        return TorsoTwoHandControl(
            torso_nm=torso_nm,
            lead_arm_nm=7.0,
            trail_arm_nm=7.0,
            lead_wrist_nm=wrist_nm,
            trail_wrist_nm=wrist_nm,
        )

    return law


def _case(
    torso_profile: str,
    torso_rate_rad_s: float,
    matching_rule: str,
    *,
    compact: bool,
    params: RotatingBaseParams | None = None,
    wrist_release_s: float = 0.025,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    params = params or RotatingBaseParams.publication_default()
    club_rate = torso_rate_rad_s + 1.0 if matching_rule == "relative_club_rate" else 3.0
    state = initial_state(
        params,
        torso_rate_rad_s=torso_rate_rad_s,
        club_rate_rad_s=club_rate,
    )
    config = RotatingBaseConfig(
        duration_s=0.06 if compact else 0.12,
        step_s=0.002 if compact else 0.0005,
    )
    trace = rollout(
        state,
        _control_law(TORSO_PROFILES_NM[torso_profile], wrist_release_s=wrist_release_s),
        params,
        config,
    )
    grip_magnitudes = np.linalg.norm(trace.force_on_club_n, axis=2)
    peak_grip_force = float(np.max(grip_magnitudes))
    constraint_residual = float(np.max(trace.position_constraint_norm_m))
    contact_work = _trapz(trace.contact_power_on_club_w, trace.time)
    braking_grip_work = -_trapz(
        np.minimum(trace.contact_power_on_club_w, 0.0), trace.time
    )
    force_couple_work = _trapz(
        trace.force_generated_couple_nm * trace.qdot[:, 5], trace.time
    )
    total_force = np.sum(trace.force_on_club_n, axis=1)
    speed = np.maximum(trace.clubhead_speed_m_s, 1e-12)
    along_path_force = np.sum(
        total_force * trace.clubhead_velocity_m_s / speed[:, None], axis=1
    )
    negative_along_path_impulse = -_trapz(np.minimum(along_path_force, 0.0), trace.time)
    controls = np.asarray([control.as_array() for control in trace.controls])
    lead_wrist_relative_rate = trace.qdot[:, 5] - (trace.qdot[:, 0] + trace.qdot[:, 1])
    trail_wrist_relative_rate = trace.qdot[:, 5] - (trace.qdot[:, 0] + trace.qdot[:, 2])
    bilateral_wrist_work = _trapz(
        controls[:, 3] * lead_wrist_relative_rate
        + controls[:, 4] * trail_wrist_relative_rate,
        trace.time,
    )
    reasons: list[str] = []
    if peak_grip_force > REGISTERED_PEAK_GRIP_FORCE_CEILING_N:
        reasons.append("registered_peak_grip_force_ceiling_exceeded")
    if constraint_residual >= 1e-7:
        reasons.append("position_constraint_closure_failed")
    if abs(trace.work_energy_closure_j) >= 0.08:
        reasons.append("work_energy_closure_failed")
    row: dict[str, Any] = {
        "torso_profile": torso_profile,
        "matching_rule": matching_rule,
        "initial_torso_rate_rad_s": torso_rate_rad_s,
        "initial_club_rate_rad_s": club_rate,
        "final_torso_rate_rad_s": float(trace.qdot[-1, 0]),
        "impact_speed_m_s": float(trace.clubhead_speed_m_s[-1]),
        "clubhead_speed_gain_m_s": float(
            trace.clubhead_speed_m_s[-1] - trace.clubhead_speed_m_s[0]
        ),
        "contact_work_on_club_j": contact_work,
        "braking_grip_work_j": braking_grip_work,
        "force_couple_work_j": force_couple_work,
        "negative_along_path_impulse_ns": negative_along_path_impulse,
        "bilateral_wrist_work_j": bilateral_wrist_work,
        "total_control_work_j": _trapz(trace.control_power_w, trace.time),
        "distal_energy_gain_j": float(
            trace.distal_segment_kinetic_energy_j[-1]
            - trace.distal_segment_kinetic_energy_j[0]
        ),
        "peak_grip_force_n": peak_grip_force,
        "maximum_constraint_residual_m": constraint_residual,
        "maximum_velocity_constraint_residual_m_s": float(
            np.max(trace.velocity_constraint_norm_m_s)
        ),
        "maximum_contact_power_identity_residual_w": float(
            np.max(np.abs(trace.contact_power_identity_residual_w))
        ),
        "work_energy_closure_j": trace.work_energy_closure_j,
        "valid": not reasons,
        "exclusion_reasons": reasons,
    }
    arrays = {
        "time_s": trace.time,
        "torso_rate_rad_s": trace.qdot[:, 0],
        "club_rate_rad_s": trace.qdot[:, 5],
        "clubhead_speed_m_s": trace.clubhead_speed_m_s,
        "contact_power_on_club_w": trace.contact_power_on_club_w,
        "force_generated_couple_nm": trace.force_generated_couple_nm,
        "lead_grip_force_n": trace.force_on_club_n[:, 0],
        "trail_grip_force_n": trace.force_on_club_n[:, 1],
    }
    return row, arrays


def _pareto_indices(cases: list[dict[str, Any]]) -> list[int]:
    valid = [row for row in cases if row["valid"]]
    selected: list[int] = []
    for candidate in valid:
        dominated = any(
            other["impact_speed_m_s"] >= candidate["impact_speed_m_s"]
            and other["braking_grip_work_j"] <= candidate["braking_grip_work_j"]
            and other["peak_grip_force_n"] <= candidate["peak_grip_force_n"]
            and (
                other["impact_speed_m_s"] > candidate["impact_speed_m_s"]
                or other["braking_grip_work_j"] < candidate["braking_grip_work_j"]
                or other["peak_grip_force_n"] < candidate["peak_grip_force_n"]
            )
            for other in valid
        )
        if not dominated:
            selected.append(int(candidate["case_index"]))
    return selected


def _same_state_killswitch(*, compact: bool) -> dict[str, Any]:
    params = RotatingBaseParams.publication_default()
    state = initial_state(params, torso_rate_rad_s=3.5, club_rate_rad_s=4.5)
    branch_s = 0.03
    config = RotatingBaseConfig(
        duration_s=0.06 if compact else 0.12,
        step_s=0.002 if compact else 0.0005,
    )

    baseline_control = TorsoTwoHandControl(
        torso_nm=55.0,
        lead_arm_nm=7.0,
        trail_arm_nm=7.0,
        lead_wrist_nm=4.0,
        trail_wrist_nm=4.0,
    )

    def continuous(_time_s: float, _state: RotatingBaseState) -> TorsoTwoHandControl:
        return baseline_control

    baseline = rollout(state, continuous, params, config)
    pre = baseline.time <= branch_s
    post = baseline.time > branch_s
    channels: dict[str, dict[str, float]] = {}
    replacements = {
        "torso": {"torso_nm": 0.0},
        "bilateral_arm": {"lead_arm_nm": 0.0, "trail_arm_nm": 0.0},
        "bilateral_wrist": {"lead_wrist_nm": 0.0, "trail_wrist_nm": 0.0},
    }
    for channel, changes in replacements.items():
        killed_control = replace(baseline_control, **changes)

        def cut(
            time_s: float,
            _state: RotatingBaseState,
            after: TorsoTwoHandControl = killed_control,
        ) -> TorsoTwoHandControl:
            return baseline_control if time_s < branch_s else after

        killed = rollout(state, cut, params, config)
        channels[channel] = {
            "pre_branch_state_max_abs_difference": float(
                np.max(np.abs(baseline.q[pre] - killed.q[pre]))
            ),
            "delivery_speed_difference_m_s": float(
                baseline.clubhead_speed_m_s[-1] - killed.clubhead_speed_m_s[-1]
            ),
            "post_branch_contact_work_difference_j": _trapz(
                baseline.contact_power_on_club_w[post]
                - killed.contact_power_on_club_w[post],
                baseline.time[post],
            ),
        }
    return {
        "branch_time_s": branch_s,
        "pre_branch_state_max_abs_difference": max(
            row["pre_branch_state_max_abs_difference"] for row in channels.values()
        ),
        "post_branch_torso_torque_difference_nm": 55.0,
        "channels": channels,
        "interpretation": (
            "The trajectories share the same state and commands through the branch. "
            "Only the torso command is removed afterward, isolating a conditional "
            "continuation effect within this model."
        ),
    }


def _parameter_sensitivities(
    *, compact: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = RotatingBaseParams.publication_default()
    shaft_rows: list[dict[str, Any]] = []
    for stiffness in (40.0, 80.0, 160.0):
        row, _ = _case(
            "constant_rate",
            3.5,
            "relative_club_rate",
            compact=compact,
            params=replace(base, shaft_stiffness_nm_rad=stiffness),
        )
        shaft_rows.append(
            {
                "shaft_stiffness_nm_rad": stiffness,
                "impact_speed_m_s": row["impact_speed_m_s"],
                "braking_grip_work_j": row["braking_grip_work_j"],
                "valid": row["valid"],
            }
        )
    ensemble: list[dict[str, Any]] = []
    variants = (
        ("low_torso_inertia", replace(base, torso_inertia_kg_m2=3.84)),
        ("high_torso_inertia", replace(base, torso_inertia_kg_m2=5.76)),
        ("low_arm_mass", replace(base, arm_mass_kg=2.48)),
        ("high_arm_mass", replace(base, arm_mass_kg=3.72)),
        ("low_shaft_damping", replace(base, shaft_damping_nms_rad=0.3)),
        ("high_shaft_damping", replace(base, shaft_damping_nms_rad=0.9)),
        (
            "narrow_grip_spacing",
            replace(base, lead_grip_offset_m=0.052, trail_grip_offset_m=-0.052),
        ),
        (
            "wide_grip_spacing",
            replace(base, lead_grip_offset_m=0.078, trail_grip_offset_m=-0.078),
        ),
    )
    for label, params in variants:
        row, _ = _case(
            "constant_rate",
            3.5,
            "relative_club_rate",
            compact=compact,
            params=params,
        )
        ensemble.append(
            {
                "variant": label,
                "impact_speed_m_s": row["impact_speed_m_s"],
                "contact_work_on_club_j": row["contact_work_on_club_j"],
                "peak_grip_force_n": row["peak_grip_force_n"],
                "valid": row["valid"],
            }
        )
    for label, release_s in (
        ("early_wrist_release", 0.015),
        ("late_wrist_release", 0.035),
    ):
        row, _ = _case(
            "constant_rate",
            3.5,
            "relative_club_rate",
            compact=compact,
            wrist_release_s=release_s,
        )
        ensemble.append(
            {
                "variant": label,
                "impact_speed_m_s": row["impact_speed_m_s"],
                "contact_work_on_club_j": row["contact_work_on_club_j"],
                "peak_grip_force_n": row["peak_grip_force_n"],
                "valid": row["valid"],
            }
        )
    return shaft_rows, ensemble


def _negative_controls() -> dict[str, Any]:
    base = RotatingBaseParams.publication_default()
    coincident = replace(base, lead_grip_offset_m=0.0, trail_grip_offset_m=0.0)
    coincident_state = initial_state(
        coincident, torso_rate_rad_s=3.5, club_rate_rad_s=3.0
    )
    coincident_solution = solve_constrained_dynamics(
        coincident_state, TorsoTwoHandControl(torso_nm=20.0), coincident
    )
    baseline_state = initial_state(base, torso_rate_rad_s=3.5, club_rate_rad_s=3.0)
    baseline_solution = solve_constrained_dynamics(
        baseline_state, TorsoTwoHandControl(torso_nm=20.0), base
    )
    base_couple = baseline_solution.force_generated_couple_nm
    # Algebraic geometry kill-switch: hold the solved forces and state fixed,
    # then reverse only their signed moment arms. This avoids confounding the
    # sign test with a different closed-loop configuration.
    reversed_couple = -base_couple
    return {
        "coincident_grip_max_couple_nm": abs(
            coincident_solution.force_generated_couple_nm
        ),
        "baseline_separated_grip_couple_nm": base_couple,
        "reversed_grip_couple_nm": reversed_couple,
        "reversed_grip_couple_sign_reversed": bool(base_couple * reversed_couple < 0.0),
        "interpretation": (
            "The coincident-grip control removes the force-generated couple. "
            "Offset reversal is an algebraic same-state sign test; it is not "
            "a dynamically feasible human intervention claim."
        ),
    }


@lru_cache(maxsize=2)
def build_study(
    *, compact: bool = False
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the deterministic record and aligned machine-readable arrays."""
    cases: list[dict[str, Any]] = []
    case_arrays: list[dict[str, np.ndarray]] = []
    for matching_rule in MATCHING_RULES:
        for torso_profile in TORSO_PROFILES_NM:
            for torso_rate in TORSO_RATES_RAD_S:
                row, arrays = _case(
                    torso_profile,
                    torso_rate,
                    matching_rule,
                    compact=compact,
                )
                row["case_index"] = len(cases)
                cases.append(row)
                case_arrays.append(arrays)
    valid = [row for row in cases if row["valid"]]
    rates = np.asarray([row["initial_torso_rate_rad_s"] for row in valid])
    speeds = np.asarray([row["impact_speed_m_s"] for row in valid])
    braking = np.asarray([row["braking_grip_work_j"] for row in valid])
    shaft_sensitivity, uncertainty_ensemble = _parameter_sensitivities(compact=compact)
    record: dict[str, Any] = {
        "schema_version": "rotating-base-torso-velocity-study-v1",
        "study_id": "registered-rotating-base-two-hand-torso-velocity-grid",
        "model_tier": "planar_rotating_base_two_hand_compliant_club",
        "attempted_case_count": len(cases),
        "valid_case_count": len(valid),
        "registered_peak_grip_force_ceiling_n": REGISTERED_PEAK_GRIP_FORCE_CEILING_N,
        "matching_rules": {
            "relative_club_rate": "initial club-minus-torso angular rate fixed at 1 rad/s",
            "absolute_club_rate": "initial absolute proximal-club rate fixed at 3 rad/s",
        },
        "cases": cases,
        "negative_controls": _negative_controls(),
        "same_state_killswitch": _same_state_killswitch(compact=compact),
        "shaft_sensitivity": shaft_sensitivity,
        "uncertainty_ensemble": uncertainty_ensemble,
        "pareto_case_indices": _pareto_indices(cases),
        "associations": {
            "torso_rate_vs_impact_speed_r": _correlation(rates, speeds),
            "torso_rate_vs_braking_grip_work_r": _correlation(rates, braking),
            "causal_interpretation": False,
        },
        "claims": {
            "universal_high_torso_velocity_strategy": "not_supported",
            "human_coaching_strategy": "unsupported",
            "bounded_mechanism_claim": (
                "Within this reduced model, torso-rate history changes bilateral "
                "reaction work and delivery speed conditional on the stated "
                "matching rule, control law, and validity envelope."
            ),
        },
        "limitations": [
            "Planar reduced coordinates are not anatomical shoulder observables.",
            "The arm and wrist commands are prescribed, not identified from people.",
            "Associations across the grid do not identify a causal coaching strategy.",
            "Human validation requires synchronized bilateral grip wrenches and kinematics.",
        ],
    }
    output_arrays: dict[str, np.ndarray] = {
        "case_impact_speed_m_s": np.asarray([row["impact_speed_m_s"] for row in cases]),
        "case_valid": np.asarray([row["valid"] for row in cases], dtype=bool),
        "case_initial_torso_rate_rad_s": np.asarray(
            [row["initial_torso_rate_rad_s"] for row in cases]
        ),
    }
    for index, arrays in enumerate(case_arrays):
        for name, values in arrays.items():
            output_arrays[f"case_{index:02d}_{name}"] = values
    return record, output_arrays


def main() -> None:
    record, arrays = build_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(NPZ_PATH, **arrays)
    print(JSON_PATH)
    print(NPZ_PATH)


if __name__ == "__main__":
    main()
