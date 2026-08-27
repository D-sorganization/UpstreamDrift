"""Generate spatial contact with paired native inertia-and-bias transport."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    SpatialContactParameters,
)
from scripts.research.proximal_distal_energy.spatial_forward_study import (
    SpatialForwardTrace,
    compare_engine_traces,
    engine_identity_record,
    run_engine_trace,
    summarize_trace,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
JSON_PATH = DATA_DIR / "spatial_forward_contact_study.json"
NPZ_PATH = DATA_DIR / "spatial_forward_contact_study.npz"
SCHEMA_VERSION = "spatial-forward-contact-evidence-v1"
STUDY_ID = "spatial-forward-two-engine-contact-v1"


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/spatial_forward_contract.py",
        "scripts/research/proximal_distal_energy/spatial_forward_engines.py",
        "scripts/research/proximal_distal_energy/spatial_forward_study.py",
        "scripts/research/proximal_distal_energy/run_spatial_forward_contact_study.py",
        "scripts/research/proximal_distal_energy/make_spatial_forward_contact_figures.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _trace_arrays(prefix: str, trace: SpatialForwardTrace) -> dict[str, np.ndarray]:
    names = (
        "time",
        "hand_positions",
        "club_position",
        "club_quaternion_wxyz",
        "club_axis",
        "club_angular_velocity",
        "contact_forces",
        "contact_points",
        "contact_wrench",
        "swing_normal_couple",
        "long_axis_couple",
        "swing_plane_tilt",
        "driver_forces",
        "ground_pathway_wrench",
        "driver_power",
        "contact_dissipation_power",
        "interface_storage_energy",
        "native_mechanical_energy",
        "total_energy",
        "action_reaction_force_residual",
        "interface_power_residual",
        "wrench_power_residual",
        "coincident_couple",
        "reversed_couple",
        "energy_balance_residual",
    )
    return {f"{prefix}_{name}": np.asarray(getattr(trace, name)) for name in names}


def _maximum_prebranch_state_difference(
    baseline: SpatialForwardTrace,
    killed: SpatialForwardTrace,
    killswitch_time: float,
) -> float:
    mask = baseline.time <= killswitch_time
    differences = (
        np.max(np.abs(baseline.club_position[mask] - killed.club_position[mask])),
        np.max(np.abs(baseline.hand_positions[mask] - killed.hand_positions[mask])),
        np.max(
            np.abs(
                baseline.club_quaternion_wxyz[mask] - killed.club_quaternion_wxyz[mask]
            )
        ),
    )
    return float(max(differences))


def _collect_study_traces(
    params: SpatialContactParameters,
) -> dict[str, SpatialForwardTrace]:
    traces: dict[str, SpatialForwardTrace] = {}
    for engine in ("mujoco", "pinocchio"):
        traces[f"{engine}_baseline"] = run_engine_trace(
            engine, params, disable_driver_after_killswitch=False
        )
        traces[f"{engine}_killswitch"] = run_engine_trace(
            engine, params, disable_driver_after_killswitch=True
        )
    return traces


def _evaluate_time_step_refinement(
    params: SpatialContactParameters,
    base_residual: float,
) -> tuple[list[float], list[float]]:
    energy_refinement = {params.time_step: base_residual}
    for time_step in (0.0005, 0.000125):
        refinement_params = replace(params, time_step=time_step)
        refinement_trace = run_engine_trace(
            "mujoco", refinement_params, disable_driver_after_killswitch=True
        )
        energy_refinement[time_step] = summarize_trace(
            refinement_trace, refinement_params
        )["energy_balance_residual_max_j"]
    ordered_steps = sorted(energy_refinement, reverse=True)
    ordered_residuals = [energy_refinement[value] for value in ordered_steps]
    return ordered_steps, ordered_residuals


def _build_numerical_gates(
    baseline_gate: dict[str, Any],
    killswitch_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trajectory_gate_passed": bool(
            baseline_gate["trajectory_gate_passed"]
            and killswitch_gate["trajectory_gate_passed"]
        ),
        "wrench_gate_passed": bool(
            baseline_gate["wrench_gate_passed"]
            and killswitch_gate["wrench_gate_passed"]
        ),
        "energy_gate_passed": bool(
            baseline_gate["energy_gate_passed"]
            and killswitch_gate["energy_gate_passed"]
        ),
        "baseline": baseline_gate,
        "same_state_killswitch": killswitch_gate,
    }


def _build_mechanism_tests(
    summaries: dict[str, Any],
    prebranch_difference: float,
) -> dict[str, Any]:
    return {
        "same_state_prebranch_max_state_difference": prebranch_difference,
        "same_state_killswitch_negative_duration_s": min(
            summaries["mujoco_killswitch"]["post_killswitch_negative_duration_s"],
            summaries["pinocchio_killswitch"]["post_killswitch_negative_duration_s"],
        ),
        "same_state_killswitch_minimum_couple_nm": min(
            summaries["mujoco_killswitch"]["minimum_post_killswitch_couple_nm"],
            summaries["pinocchio_killswitch"]["minimum_post_killswitch_couple_nm"],
        ),
        "coincident_grip_couple_max_nm": max(
            summary["coincident_grip_couple_max_nm"] for summary in summaries.values()
        ),
        "reversed_geometry_sign_residual_max_nm": max(
            summary["reversed_geometry_sign_residual_nm"]
            for summary in summaries.values()
        ),
        "direct_club_force_or_torque_command": "none",
        "baseline_and_killswitch_trace_summaries": summaries,
    }


def _build_study_record(
    params: SpatialContactParameters,
    identities: dict[str, Any],
    digests: dict[str, str],
    numerical_gates: dict[str, Any],
    ordered_steps: list[float],
    ordered_residuals: list[float],
    mechanism_tests: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "model_tier": "reduced_spatial_two_hand_forward_compliant_contact",
        "trajectory_kind": (
            "shared_projected_contact_and_integrator_with_native_inertia_bias_transport"
        ),
        "engine_identities": identities,
        "model_contract": {
            "canonical_record": params.canonical_record(),
            "sha256": params.model_digest(),
            "adapter_digests": digests,
            "digest_match": len(set(digests.values())) == 1,
        },
        "contact_contract": {
            "kind": "paired Kelvin-Voigt point interfaces",
            "force_origin": "achieved relative hand-club displacement and velocity",
            "action_reaction": "equal and opposite at every evaluation",
            "driver_pathway": "grounded spring-damper force on finite-mass hand carriages",
            "club_direct_actuation": "none",
        },
        "interventions": {
            "same_state_driver_killswitch_s": params.killswitch_time,
            "coincident_grips": "same achieved forces transported through club reference",
            "reversed_geometry": "same achieved forces with both moment arms negated",
        },
        "numerical_gates": numerical_gates,
        "timestep_refinement": {
            "time_step_s": ordered_steps,
            "maximum_work_energy_residual_j": ordered_residuals,
            "monotone_residual_reduction": bool(
                np.all(np.diff(ordered_residuals) < 0.0)
            ),
        },
        "mechanism_tests": mechanism_tests,
        "claim_status": {
            "passive_post_killswitch_negative_force_couple": (
                "supported_in_declared_reduced_spatial_contact_model"
            ),
            "native_inertia_bias_transport": (
                "supported_for_mujoco_and_pinocchio_under_shared_contact_and_update"
            ),
            "long_axis_rotation_and_swing_plane_evolution": (
                "executed_in_both_engines"
            ),
            "ground_pathway_accounting": "executed_as_reduced_driver_reaction_proxy",
            "subject_specific_anatomy": "untested",
            "muscle_coordination": "untested",
            "human_strategy": "untested",
        },
        "claim_boundary": {
            "human_strategy": "untested",
            "muscle_mechanism": "untested",
            "equipment_calibration": "untested",
            "physiological_effort": "untested",
        },
        "limitations": [
            "The two hand bodies are finite-mass translational carriages, not anatomical arms.",
            "The grounded driver is a declared reference-force pathway, not muscle activation.",
            "The Kelvin-Voigt interface is a reduced grip law, not measured tissue calibration.",
            "The branches share one contact law and state update while native libraries independently supply kinematics, mass, bias, gravity, spatial-force mapping, and continuous-time acceleration.",
            "Neither native contact solver nor native integrator is exercised in this transport study.",
            "The experiment does not use participant data and cannot support a coaching prescription.",
        ],
        "source_sha256": _source_hashes(),
        "array_artifact": NPZ_PATH.name,
    }


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute baseline and same-state driver-killswitch branches."""

    params = SpatialContactParameters()
    traces = _collect_study_traces(params)

    baseline_gate = compare_engine_traces(
        traces["mujoco_baseline"], traces["pinocchio_baseline"], params
    )
    killswitch_gate = compare_engine_traces(
        traces["mujoco_killswitch"], traces["pinocchio_killswitch"], params
    )
    summaries = {key: summarize_trace(value, params) for key, value in traces.items()}
    ordered_steps, ordered_residuals = _evaluate_time_step_refinement(
        params,
        summaries["mujoco_killswitch"]["energy_balance_residual_max_j"],
    )
    prebranch_difference = max(
        _maximum_prebranch_state_difference(
            traces[f"{engine}_baseline"],
            traces[f"{engine}_killswitch"],
            params.killswitch_time,
        )
        for engine in ("mujoco", "pinocchio")
    )
    identities = {
        engine: engine_identity_record(traces[f"{engine}_baseline"].engine_identity)
        for engine in ("mujoco", "pinocchio")
    }
    digests = {
        key: trace.model_digest
        for key, trace in traces.items()
        if key.endswith("baseline")
    }
    numerical_gates = _build_numerical_gates(baseline_gate, killswitch_gate)
    mechanism_tests = _build_mechanism_tests(summaries, prebranch_difference)
    record = _build_study_record(
        params,
        identities,
        digests,
        numerical_gates,
        ordered_steps,
        ordered_residuals,
        mechanism_tests,
    )
    arrays: dict[str, np.ndarray] = {}
    for prefix, trace in traces.items():
        arrays.update(_trace_arrays(prefix, trace))
    return record, arrays


def write_outputs() -> tuple[Path, Path]:
    """Write deterministic JSON metadata and lossless arrays."""

    record, arrays = run_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> None:
    json_path, npz_path = write_outputs()
    print(json_path)
    print(npz_path)


if __name__ == "__main__":
    main()


__all__ = ["run_study", "write_outputs"]
