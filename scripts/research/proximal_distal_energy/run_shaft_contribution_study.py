"""Run matched rigid/flexible and contribution-separation experiments."""

from __future__ import annotations

import hashlib
import json
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
    velocity_bias_power_identity_residual,
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
SCHEMA_VERSION = "shaft-contribution-study-v2"
JSON_PATH = DATA_DIR / "shaft_contribution_study.json"
NPZ_PATH = DATA_DIR / "shaft_contribution_traces.npz"


def _canonicalize_json(value: Any) -> Any:
    """Normalize floating scalars to a stable ten-significant-digit record.

    Parallel BLAS reductions can differ by one final binary ulp across otherwise
    identical processes. The authority JSON reports scientific observables, not
    raw state arrays, so a declared decimal canonicalization prevents those
    meaningless last-bit differences from changing the evidence identity.
    """
    if isinstance(value, dict):
        return {key: _canonicalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(format(float(value), ".10g"))
    return value


def _canonicalize_array(values: np.ndarray) -> np.ndarray:
    """Return the declared 1e-7-resolution evidence representation."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        return array
    return np.round(array.astype(np.float64, copy=False), decimals=7)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        REPO_ROOT / "scripts/research/proximal_distal_energy/flexible_shaft_study.py",
        REPO_ROOT
        / "scripts/research/proximal_distal_energy/run_shaft_contribution_study.py",
        REPO_ROOT / "src/shared/python/pendulum_simulator/physics_triple.py",
    )
    return {path.relative_to(REPO_ROOT).as_posix(): _sha256(path) for path in paths}


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
    kinematic_gradient = np.gradient(kinematics["tip"], trace.t, axis=0, edge_order=2)
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
        "tip_velocity_method": "analytic relative-coordinate kinematics",
        "maximum_tip_position_gradient_discrepancy_m_s": float(
            np.max(
                np.linalg.norm(kinematics["tip_velocity"] - kinematic_gradient, axis=1)
            )
        ),
        **window_metrics,
    }


def _shaft_port_accounting(
    trace: FlexibleTrace, params: FlexibleShaftParams
) -> dict[str, np.ndarray]:
    """Return the two-sided compliant-joint wrench-power identity.

    The point-force powers cancel because the two adjacent links share the
    same joint-point velocity. Opposed couples act at different angular
    velocities, so their sum equals the relative-coordinate spring/damper
    power rather than zero.
    """
    kinematics = trace_kinematics(trace, params)
    force = shaft_interface_force(trace, params)
    joint_velocity = kinematics["wrist2_velocity"]
    omega_proximal = trace.state[:, 3] + trace.state[:, 4]
    omega_distal = omega_proximal + trace.state[:, 5]
    couple = -params.shaft_stiffness_nm_rad * trace.state[:, 2]
    if not trace.rigid:
        couple -= params.shaft_damping_nms_rad * trace.state[:, 5]
    distal_force_power = np.einsum("ij,ij->i", force, joint_velocity)
    proximal_force_power = np.einsum("ij,ij->i", -force, joint_velocity)
    distal_couple_power = couple * omega_distal
    proximal_couple_power = -couple * omega_proximal
    adjacent_body_power = (
        distal_force_power
        + proximal_force_power
        + distal_couple_power
        + proximal_couple_power
    )
    relative_coordinate_power = couple * trace.state[:, 5]
    return {
        "force": force,
        "joint_velocity": joint_velocity,
        "couple": couple,
        "omega_proximal": omega_proximal,
        "omega_distal": omega_distal,
        "distal_force_power": distal_force_power,
        "proximal_force_power": proximal_force_power,
        "distal_couple_power": distal_couple_power,
        "proximal_couple_power": proximal_couple_power,
        "adjacent_body_power": adjacent_body_power,
        "relative_coordinate_power": relative_coordinate_power,
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
    velocity_bias_residual = np.max(
        np.abs(
            [
                velocity_bias_power_identity_residual(state, params)
                for state in trace.state
            ]
        )
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
        "velocity_bias_power_identity_tolerance_w": 1.0e-6,
        "velocity_bias_power_identity_verified": bool(velocity_bias_residual < 1.0e-6),
    }
    return summary, accounting


def _balanced_main_effect_fractions(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, float]:
    """Return balanced-grid main-effect sums of squares divided by total SS."""
    response = np.asarray([row[metric] for row in rows], dtype=float)
    grand_mean = float(np.mean(response))
    total_ss = float(np.sum((response - grand_mean) ** 2))
    result: dict[str, float] = {}
    for factor in (
        "shaft_stiffness_nm_rad",
        "shaft_damping_nms_rad",
        "torque_cut_time_s",
    ):
        levels = sorted({str(row[factor]) for row in rows})
        factor_ss = 0.0
        for level in levels:
            indices = [
                index for index, row in enumerate(rows) if str(row[factor]) == level
            ]
            factor_ss += (
                len(indices) * (float(np.mean(response[indices])) - grand_mean) ** 2
            )
        result[factor] = factor_ss / total_ss
    result["unattributed_to_main_effects"] = 1.0 - sum(result.values())
    return result


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
        port = _shaft_port_accounting(trace, params)
        arrays.update(
            {f"{name}__shaft_port_{key}": value for key, value in port.items()}
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

    robustness_attribution = {
        metric: _balanced_main_effect_fractions(robustness_rows, metric)
        for metric in (
            "impact_speed_m_s",
            "impact_time_s",
            "peak_shaft_strain_energy_j",
            "shaft_flex_at_impact_deg",
        )
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "matched-rigid-flexible-shaft-contribution-study",
        "source_sha256": _source_hashes(),
        "provenance": {
            "model": "Three-link planar point-mass chain with one lumped linear shaft-flex coordinate",
            "rigid_contract": "Exact coordinate reduction phi2 = dphi2 = 0 with the same mass distribution",
            "integrator": "fixed-step RK4",
            "horizon_s": HORIZON_S,
            "robustness_horizon_s": ROBUSTNESS_HORIZON_S,
            "reference_dt_s": REFERENCE_DT_S,
            "json_reporting_precision_significant_digits": 10,
            "trace_reporting_absolute_resolution": 1.0e-7,
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
            "balanced_main_effect_fraction_of_total_ss": robustness_attribution,
            "attribution_boundary": (
                "Fractions are descriptive main-effect sums of squares for this "
                "balanced deterministic grid; the residual includes interactions "
                "and is not sampling uncertainty or a human causal share."
            ),
        },
    }
    canonical_arrays = {
        name: _canonicalize_array(values) for name, values in arrays.items()
    }
    return _canonicalize_json(record), canonical_arrays


def write_outputs(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Write deterministic JSON metrics and compressed selected traces."""
    record, arrays = build_outputs()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_PATH.name
    npz_path = output_dir / NPZ_PATH.name
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path


def main() -> None:
    """Write JSON metrics and compressed selected traces."""
    write_outputs()


if __name__ == "__main__":
    main()
