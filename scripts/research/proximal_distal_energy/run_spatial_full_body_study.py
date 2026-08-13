"""Generate the spatial full-body common-state evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialExperimentConfig,
    build_spatial_model,
    evaluate_hand_wrenches,
    generalized_hand_load_power_residual,
    prescribed_state,
    run_cross_formulation_experiment,
    wrench_reference_power_residual,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"
JSON_PATH = DATA_DIR / "spatial_full_body_study.json"
NPZ_PATH = DATA_DIR / "spatial_full_body_study.npz"
SCHEMA_VERSION = "spatial-full-body-common-state-evidence-v1"
STUDY_ID = "spatial-full-body-common-state-v1"


def _source_hashes() -> dict[str, str]:
    paths = (
        "scripts/research/proximal_distal_energy/spatial_full_body.py",
        "scripts/research/proximal_distal_energy/run_spatial_full_body_study.py",
    )
    return {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _first_negative_time(time: np.ndarray, values: np.ndarray) -> float | None:
    indices = np.flatnonzero(values < -1e-10)
    return None if indices.size == 0 else float(time[int(indices[0])])


def run_study() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute the registered comparison and intervention controls."""

    model = build_spatial_model()
    fine = run_cross_formulation_experiment(
        SpatialExperimentConfig(derivative_step=1.0e-6)
    )
    coarse = run_cross_formulation_experiment(
        SpatialExperimentConfig(derivative_step=2.0e-6)
    )
    reversed_couple = np.empty_like(fine.force_generated_couple_nm)
    coincident_couple = np.empty_like(fine.force_generated_couple_nm)
    club_position = np.empty((fine.time_s.size, 3))
    club_rotation = np.empty((fine.time_s.size, 3))
    club_wrench = np.empty((fine.time_s.size, 6))
    compatible_twist = np.empty((fine.time_s.size, 6))
    power_residual = np.empty(fine.time_s.size)
    generalized_load_power_residual = np.empty(fine.time_s.size)
    reference_transport_power_residual = np.empty(fine.time_s.size)
    for index, time_s in enumerate(fine.time_s):
        baseline = evaluate_hand_wrenches(model, float(time_s), coincident_hands=False)
        reversed_sample = evaluate_hand_wrenches(
            model,
            float(time_s),
            coincident_hands=False,
            reverse_geometry=True,
        )
        coincident = evaluate_hand_wrenches(model, float(time_s), coincident_hands=True)
        q, _, _ = prescribed_state(model, float(time_s))
        club_position[index] = q[14:17]
        club_rotation[index] = q[17:20]
        club_wrench[index] = baseline.club_wrench
        compatible_twist[index] = baseline.compatible_twist
        power_residual[index] = baseline.action_reaction_power_residual_w
        generalized_load_power_residual[index] = generalized_hand_load_power_residual(
            model, float(time_s)
        )
        reference_transport_power_residual[index] = wrench_reference_power_residual(
            model, float(time_s)
        )
        reversed_couple[index] = reversed_sample.force_generated_couple_nm
        coincident_couple[index] = coincident.force_generated_couple_nm

    fine_pair_difference = float(
        np.max(
            np.abs(fine.inverse_dynamics_lagrange - coarse.inverse_dynamics_lagrange)
        )
    )
    sign_reversal_residual = float(
        np.max(np.abs(fine.force_generated_couple_nm + reversed_couple))
    )
    arrays = {
        "time_s": fine.time_s,
        "inverse_dynamics_lagrange": fine.inverse_dynamics_lagrange,
        "inverse_dynamics_mujoco": fine.inverse_dynamics_mujoco,
        "force_generated_couple_nm": fine.force_generated_couple_nm,
        "reverse_geometry_couple_nm": reversed_couple,
        "coincident_hands_couple_nm": coincident_couple,
        "club_position_m": club_position,
        "club_rotation_rad": club_rotation,
        "club_wrench": club_wrench,
        "compatible_twist": compatible_twist,
        "action_reaction_power_residual_w": power_residual,
        "generalized_load_power_residual_w": generalized_load_power_residual,
        "reference_transport_power_residual_w": reference_transport_power_residual,
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "model_tier": "reduced_full_body_3d_common_state_inverse_dynamics",
        "trajectory_kind": "prescribed_common_state_inverse_dynamics",
        "formulations": list(fine.formulation_names),
        "common_model_sha256": model.canonical_hash,
        "model_dimensions": {
            "generalized_coordinates": model.nq,
            "body_inertia_elements": len(model.bodies),
            "club_coordinates": 6,
        },
        "intervention": {
            "description": (
                "Reverse the signed two-hand contact moment arm while retaining "
                "force histories and common achieved state."
            ),
            "direct_club_torque_command_nm": 0.0,
            "contact_load_origin": "prescribed action-reaction input",
            "force_generated_couple_minimum_nm": float(
                np.min(fine.force_generated_couple_nm)
            ),
            "first_negative_time_s": _first_negative_time(
                fine.time_s, fine.force_generated_couple_nm
            ),
            "reversed_geometry_couple_maximum_nm": float(np.max(reversed_couple)),
            "sign_reversal_residual_max_nm": sign_reversal_residual,
            "coincident_hands_couple_max_abs_nm": float(
                np.max(np.abs(coincident_couple))
            ),
        },
        "cross_formulation": {
            "classification": fine.classification,
            "maximum_absolute_generalized_force_error": (
                fine.max_absolute_generalized_force_error
            ),
            "maximum_relative_inverse_dynamics_error": (
                fine.max_relative_inverse_dynamics_error
            ),
            "maximum_absolute_mass_matrix_error": (fine.max_absolute_mass_matrix_error),
            "maximum_relative_mass_matrix_error": (fine.max_relative_mass_matrix_error),
            "maximum_absolute_bias_force_error": (fine.max_absolute_bias_force_error),
            "maximum_relative_bias_force_error": (fine.max_relative_bias_force_error),
            "external_load_convention_mismatch_relative_error": (
                fine.external_load_convention_mismatch_relative_error
            ),
            "intervention_event_grid_error_s": (fine.intervention_event_grid_error_s),
            "tolerance": {
                "absolute_generalized_force": fine.tolerance.absolute,
                "relative": fine.tolerance.relative,
                "event_time_s": fine.tolerance.event_time_s,
                "calibration": fine.tolerance.calibration,
            },
            "finite_difference_finest_pair_difference": fine_pair_difference,
        },
        "spatial_checks": {
            "out_of_plane_club_motion_m": fine.out_of_plane_motion_m,
            "maximum_abs_action_reaction_power_residual_w": float(
                np.max(np.abs(power_residual))
            ),
            "maximum_abs_generalized_load_power_residual_w": float(
                np.max(np.abs(generalized_load_power_residual))
            ),
            "maximum_abs_reference_transport_power_residual_w": float(
                np.max(np.abs(reference_transport_power_residual))
            ),
            "proper_axes_rank": 3,
        },
        "claim_status": {
            "H2_geometry_dependent_transfer": "supported_in_reduced_spatial_common_state_tier",
            "H3_passive_late_negative_couple": "inconclusive_contact_loads_prescribed",
            "H5_implementation_transport": (
                "supported_for_common_state_inverse_dynamics"
            ),
            "full_body_forward_closed_contact": "untested",
            "human_or_coaching_inference": "unsupported",
        },
        "limitations": [
            "The reduced full-body tree is not a subject-specific anatomical model.",
            "Hand contact forces are prescribed rather than solved from two-hand closure.",
            "The experiment compares identical achieved states, not divergent forward rollouts.",
            "Spherical inertia elements regularize a common model and do not represent distributed soft tissue.",
            "No measured human trajectory or force record is used as validation data.",
        ],
        "source_sha256": _source_hashes(),
        "array_artifact": NPZ_PATH.name,
    }
    return record, arrays


def write_outputs() -> tuple[Path, Path]:
    """Write deterministic JSON metadata and lossless trace arrays."""

    record, arrays = run_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    cast(Any, np.savez_compressed)(NPZ_PATH, **arrays)
    return JSON_PATH, NPZ_PATH


def main() -> None:
    json_path, npz_path = write_outputs()
    print(json_path)
    print(npz_path)


if __name__ == "__main__":
    main()
