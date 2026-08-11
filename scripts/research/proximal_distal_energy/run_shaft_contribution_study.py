"""Run matched rigid/flexible and contribution-separation experiments."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.flexible_shaft_study import (
    FlexibleShaftParams,
    FlexibleTrace,
    energy_accounting,
    generalized_terms,
    rollout_flexible,
    rollout_rigid,
    shaft_interface_force,
    trace_kinematics,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
HORIZON_S = 0.50
ROBUSTNESS_HORIZON_S = 0.75
REFERENCE_DT_S = 0.0005
STIFFNESS_VALUES = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
DAMPING_VALUES = (0.0, 0.3, 0.6, 1.2)
CUT_TIMES = (None, 0.12, 0.18, 0.24, 0.30)
IMPACT_WINDOWS_S = (0.0, 0.005, 0.010, 0.020)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _impact_index(trace: FlexibleTrace) -> int | None:
    absolute_tip_angle = np.sum(trace.state[:, :3], axis=1)
    crossing = np.flatnonzero(
        (absolute_tip_angle[:-1] < 0.0) & (absolute_tip_angle[1:] >= 0.0)
    )
    return int(crossing[0]) if crossing.size else None


def _impact_summary(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> dict[str, Any]:
    kinematics = trace_kinematics(trace, params)
    index = _impact_index(trace)
    if index is None:
        return {
            "impact_found": False,
            "peak_speed_m_s": float(np.max(kinematics["tip_speed"])),
        }
    angle = np.sum(trace.state[:, :3], axis=1)
    fraction = float(-angle[index] / (angle[index + 1] - angle[index]))

    def interpolate(values: np.ndarray) -> float:
        return float(values[index] + fraction * (values[index + 1] - values[index]))

    impact_time = interpolate(trace.t)
    impact_speed = interpolate(kinematics["tip_speed"])
    flex = interpolate(trace.state[:, 2])
    flex_rate = interpolate(trace.state[:, 5])
    shaft_moment = -params.shaft_stiffness_nm_rad * flex
    damping_moment = -params.shaft_damping_nms_rad * flex_rate
    force = shaft_interface_force(trace, params)
    force_magnitude = np.linalg.norm(force, axis=1)
    window_metrics: dict[str, float] = {}
    for window in IMPACT_WINDOWS_S:
        if window == 0.0:
            value = impact_speed
        else:
            mask = np.abs(trace.t - impact_time) <= window
            value = float(np.max(kinematics["tip_speed"][mask]))
        window_metrics[f"peak_speed_within_{window * 1000:.0f}_ms_m_s"] = value
    return {
        "impact_found": True,
        "impact_time_s": impact_time,
        "impact_speed_m_s": impact_speed,
        "peak_speed_m_s": float(np.max(kinematics["tip_speed"])),
        "peak_speed_time_s": float(trace.t[np.argmax(kinematics["tip_speed"])]),
        "shaft_flex_at_impact_rad": flex,
        "shaft_flex_at_impact_deg": float(np.rad2deg(flex)),
        "shaft_elastic_moment_at_impact_nm": shaft_moment,
        "shaft_damping_moment_at_impact_nm": damping_moment,
        "shaft_interface_force_at_impact_n": interpolate(force_magnitude),
        **window_metrics,
    }


def _term_accounting(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> dict[str, dict[str, float]]:
    names = (
        "control",
        "momentum",
        "gravity",
        "joint_damping",
        "shaft_elastic",
        "shaft_damping",
    )
    velocity = trace.state[:, 3:]
    torque_arrays = {name: np.empty((trace.t.size, 3)) for name in names}
    for index, (time_s, state) in enumerate(zip(trace.t, trace.state, strict=True)):
        terms = generalized_terms(state, time_s, params)
        for name in names:
            torque_arrays[name][index] = terms[name]
    result: dict[str, dict[str, float]] = {}
    for name in names:
        power = np.sum(torque_arrays[name] * velocity, axis=1)
        result[name] = {
            "peak_abs_generalized_torque_nm": float(
                np.max(np.abs(torque_arrays[name]))
            ),
            "peak_abs_acceleration_contribution_rad_s2": float(
                np.max(np.abs(trace.contributions[name]))
            ),
            "peak_abs_generalized_power_w": float(np.max(np.abs(power))),
            "signed_power_integral_j": float(np.trapezoid(power, trace.t)),
        }
    return result


def _energy_summary(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    accounting = energy_accounting(trace, params)
    closure = (
        accounting["total_mechanical_energy"]
        - accounting["total_mechanical_energy"][0]
        - accounting["cumulative_nonconservative_work"]
    )
    switch_mask = np.abs(trace.t - params.wrist_onset_s) > 2.0 * np.median(
        np.diff(trace.t)
    )
    summary = {
        "peak_kinetic_energy_j": float(np.max(accounting["kinetic_energy"])),
        "peak_shaft_strain_energy_j": float(np.max(accounting["shaft_strain_energy"])),
        "final_nonconservative_work_j": float(
            accounting["cumulative_nonconservative_work"][-1]
        ),
        "final_energy_closure_error_j": float(closure[-1]),
        "maximum_abs_energy_closure_error_j": float(np.max(np.abs(closure))),
        "rms_rate_residual_away_from_switch_w": float(
            np.sqrt(np.mean(accounting["energy_rate_residual"][switch_mask] ** 2))
        ),
        "shaft_damping_dissipation_j": float(
            np.trapezoid(accounting["shaft_damping_power"], trace.t)
        ),
        "joint_damping_dissipation_j": float(
            np.trapezoid(accounting["joint_damping_power"], trace.t)
        ),
        "control_work_j": float(np.trapezoid(accounting["control_power"], trace.t)),
    }
    return summary, accounting


def _evaluate(
    name: str,
    params: FlexibleShaftParams,
    *,
    rigid: bool,
    dt_s: float = REFERENCE_DT_S,
) -> tuple[dict[str, Any], FlexibleTrace, dict[str, np.ndarray]]:
    trace = (rollout_rigid if rigid else rollout_flexible)(
        params, horizon_s=HORIZON_S, dt_s=dt_s
    )
    energy_summary, accounting = _energy_summary(trace, params)
    summary = {
        "name": name,
        "rigid": rigid,
        "dt_s": dt_s,
        "parameters": {
            "gravity_enabled": params.gravity_enabled,
            "joint_damping_enabled": params.joint_damping_enabled,
            "shaft_stiffness_nm_rad": params.shaft_stiffness_nm_rad,
            "shaft_damping_nms_rad": params.shaft_damping_nms_rad,
            "torque_cut_time_s": params.torque_cut_time_s,
        },
        "impact": _impact_summary(trace, params),
        "energy": energy_summary,
        "terms": _term_accounting(trace, params),
    }
    return summary, trace, accounting


def build_outputs() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the complete evidence record and selected plot arrays."""
    reference = FlexibleShaftParams.reference()
    variants = {
        "flexible_reference": (reference, False),
        "rigid_matched": (reference, True),
        "gravity_disabled": (reference.with_updates(gravity_enabled=False), False),
        "joint_damping_disabled": (
            reference.with_updates(joint_damping_enabled=False),
            False,
        ),
        "shaft_damping_disabled": (
            reference.with_updates(shaft_damping_nms_rad=0.0),
            False,
        ),
    }
    variant_summaries: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for name, (params, rigid) in variants.items():
        summary, trace, accounting = _evaluate(name, params, rigid=rigid)
        variant_summaries.append(summary)
        kinematics = trace_kinematics(trace, params)
        arrays.update(
            {
                f"{name}__time": trace.t,
                f"{name}__state": trace.state,
                f"{name}__qddot": trace.qddot,
                f"{name}__tip_speed": kinematics["tip_speed"],
                f"{name}__wrist1": kinematics["wrist1"],
                f"{name}__wrist2": kinematics["wrist2"],
                f"{name}__tip": kinematics["tip"],
                f"{name}__shaft_force": shaft_interface_force(trace, params),
                **{
                    f"{name}__accel_{term}": values
                    for term, values in trace.contributions.items()
                },
                **{
                    f"{name}__energy_{term}": values
                    for term, values in accounting.items()
                },
            }
        )

    timestep_rows: list[dict[str, Any]] = []
    for dt_s in (0.00025, 0.0005, 0.001):
        summary, _, _ = _evaluate(
            f"flexible_dt_{dt_s}", reference, rigid=False, dt_s=dt_s
        )
        timestep_rows.append(summary)

    robustness_rows: list[dict[str, Any]] = []
    for stiffness in STIFFNESS_VALUES:
        for damping in DAMPING_VALUES:
            for cut_time in CUT_TIMES:
                params = reference.with_updates(
                    shaft_stiffness_nm_rad=stiffness,
                    shaft_damping_nms_rad=damping,
                    torque_cut_time_s=cut_time,
                )
                trace = rollout_flexible(
                    params, horizon_s=ROBUSTNESS_HORIZON_S, dt_s=REFERENCE_DT_S
                )
                robustness_rows.append(
                    {
                        "shaft_stiffness_nm_rad": stiffness,
                        "shaft_damping_nms_rad": damping,
                        "torque_cut_time_s": cut_time,
                        **_impact_summary(trace, params),
                        "peak_shaft_strain_energy_j": float(
                            np.max(0.5 * stiffness * trace.state[:, 2] ** 2)
                        ),
                    }
                )

    record = {
        "provenance": {
            "git_sha": _git_sha(),
            "model": "Three-link planar point-mass chain with one lumped linear shaft-flex coordinate",
            "rigid_contract": "Exact coordinate reduction phi2 = dphi2 = 0 with the same mass distribution",
            "integrator": "fixed-step RK4",
            "horizon_s": HORIZON_S,
            "robustness_horizon_s": ROBUSTNESS_HORIZON_S,
            "reference_dt_s": REFERENCE_DT_S,
            "interpretation_boundary": (
                "This is a mechanism and sensitivity study, not a calibrated shaft, "
                "human-subject result, or validation of the archived Simscape model."
            ),
        },
        "reference_parameters": reference.__dict__,
        "variant_summaries": variant_summaries,
        "timestep_rows": timestep_rows,
        "robustness_grid": {
            "stiffness_values_nm_rad": list(STIFFNESS_VALUES),
            "damping_values_nms_rad": list(DAMPING_VALUES),
            "cut_times_s": list(CUT_TIMES),
            "impact_windows_s": list(IMPACT_WINDOWS_S),
            "rows": robustness_rows,
        },
    }
    return record, arrays


def main() -> None:
    """Write JSON metrics and compressed selected traces."""
    record, arrays = build_outputs()
    with (DATA_DIR / "shaft_contribution_study.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    np.savez_compressed(DATA_DIR / "shaft_contribution_traces.npz", **arrays)


if __name__ == "__main__":
    main()
