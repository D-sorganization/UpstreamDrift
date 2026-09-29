"""Write deterministic evidence for the frame and biological bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.research.proximal_distal_energy.advanced_biological_bridge import (
    BiologicalProgramResult,
    build_biological_timestep_audit,
    build_frame_invariance_audit,
    build_pose_adapter_audit,
    build_redundancy_surface,
    simulate_biological_programs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer" / "data"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _program_record(result: BiologicalProgramResult) -> dict[str, object]:
    return {
        "preparation_duration_s": result.preparation_duration_s,
        "post_target_torque_nm": result.post_target_torque_nm,
        "post_transition_error_impulse_nms": (result.post_transition_error_impulse_nms),
        "minimum_total_tendon_force_n": result.minimum_tendon_force_n,
        "maximum_series_elastic_energy_j": float(
            np.max(result.series_elastic_energy_j)
        ),
    }


def _program_arrays(
    prefix: str, result: BiologicalProgramResult
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__time_s": result.time_s,
        f"{prefix}__target_arm_torque_nm": result.target_arm_torque_nm,
        f"{prefix}__target_wrist_torque_nm": result.target_wrist_torque_nm,
        f"{prefix}__transmitted_arm_torque_nm": result.transmitted_arm_torque_nm,
        f"{prefix}__transmitted_wrist_torque_nm": result.transmitted_wrist_torque_nm,
        f"{prefix}__arm_activation": result.arm_activation,
        f"{prefix}__wrist_activation": result.wrist_activation,
        f"{prefix}__tendon_force_n": result.tendon_force_n,
        f"{prefix}__series_elastic_energy_j": result.series_elastic_energy_j,
    }


def build_study() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Return the JSON record and trace arrays for publication."""
    frame = build_frame_invariance_audit()
    pose_adapters = build_pose_adapter_audit()
    redundancy = build_redundancy_surface()
    program_study = simulate_biological_programs()
    timestep_audit = build_biological_timestep_audit()
    programs = program_study.programs
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "epic": "https://github.com/D-sorganization/UpstreamDrift/issues/8505",
        "provenance": {
            "git_sha": _git_sha(),
            "shared_mechanics": "InteractionSample",
            "shared_biological_models": ["HillMuscleModel", "ActivationDynamics"],
        },
        "frame_invariance": frame,
        "pose_adapter_round_trips": pose_adapters,
        "redundancy_surface": {
            "target_torque_nm": 10.0,
            "sample_count": int(redundancy.coactivation.size),
            "maximum_torque_closure_error_nm": float(
                np.max(np.abs(redundancy.net_torque_nm - 10.0))
            ),
            "stiffness_proxy_range_nm_rad": [
                float(np.min(redundancy.stiffness_proxy_nm_rad)),
                float(np.max(redundancy.stiffness_proxy_nm_rad)),
            ],
            "series_elastic_energy_range_j": [
                float(np.min(redundancy.series_elastic_energy_j)),
                float(np.max(redundancy.series_elastic_energy_j)),
            ],
        },
        "biological_programs": {
            name: _program_record(result) for name, result in programs.items()
        },
        "biological_timestep_audit": {
            "step_s": timestep_audit.step_s.tolist(),
            "persistent_error_impulse_nms": (
                timestep_audit.persistent_error_impulse_nms.tolist()
            ),
            "reversal_error_impulse_nms": (
                timestep_audit.reversal_error_impulse_nms.tolist()
            ),
            "reversal_minus_persistent_nms": (
                timestep_audit.reversal_minus_persistent_nms.tolist()
            ),
            "direction_preserved": bool(
                np.all(timestep_audit.reversal_minus_persistent_nms > 0.0)
            ),
            "published_step_relative_difference": float(
                timestep_audit.reversal_minus_persistent_nms[2]
                / timestep_audit.reversal_error_impulse_nms[2]
            ),
            "numerical_boundary": (
                "The direction is preserved over the declared step grid, but the "
                "difference magnitude is not step-converged and is not a robust "
                "physiological or performance effect."
            ),
        },
        "engine_ladder": {
            "mujoco": {
                "status": "executed_reduced_forward_contact_in_prior_tier",
                "role": "contact-rich forward dynamics and achieved contact wrenches",
                "required_common_observables": ["q", "v", "contact_wrench", "power"],
            },
            "pinocchio": {
                "status": "executed_reduced_forward_contact_in_prior_tier",
                "role": "fast rigid-body Jacobians, RNEA, ABA, and sensitivity checks",
                "required_common_observables": ["q", "v", "tau", "jacobian"],
            },
            "drake": {
                "status": "repository_capability_and_proposed_advanced_validation",
                "role": "constrained optimization, trajectory design, and contact alternatives",
                "required_common_observables": ["q", "v", "u", "constraint_wrench"],
            },
            "opensim": {
                "status": "repository_capability_and_proposed_subject_scaled_validation",
                "role": "muscle paths, moment arms, inverse dynamics, and induced acceleration",
                "required_common_observables": [
                    "activation",
                    "muscle_force",
                    "moment_arm",
                    "tau",
                ],
            },
            "myosuite": {
                "status": "repository_capability_and_proposed_activation_driven_validation",
                "role": "activation-driven muscle/contact forward simulation and control",
                "required_common_observables": [
                    "excitation",
                    "activation",
                    "muscle_force",
                    "contact_wrench",
                ],
            },
        },
        "claim_boundary": (
            "The reduced biological bridge does not identify anatomical muscles, "
            "scapular actions, neural commands, tissue parameters, or a preferred "
            "human technique. Optional engine capability is not human validation."
        ),
    }
    arrays = {
        "redundancy__coactivation": redundancy.coactivation,
        "redundancy__positive_activation": redundancy.positive_activation,
        "redundancy__negative_activation": redundancy.negative_activation,
        "redundancy__net_torque_nm": redundancy.net_torque_nm,
        "redundancy__activation_sum": redundancy.activation_sum,
        "redundancy__stiffness_proxy_nm_rad": redundancy.stiffness_proxy_nm_rad,
        "redundancy__series_elastic_energy_j": redundancy.series_elastic_energy_j,
        **_program_arrays("persistent_direction", programs["persistent_direction"]),
        **_program_arrays("complete_role_reversal", programs["complete_role_reversal"]),
        "timestep_audit__step_s": timestep_audit.step_s,
        "timestep_audit__persistent_error_impulse_nms": (
            timestep_audit.persistent_error_impulse_nms
        ),
        "timestep_audit__reversal_error_impulse_nms": (
            timestep_audit.reversal_error_impulse_nms
        ),
        "timestep_audit__reversal_minus_persistent_nms": (
            timestep_audit.reversal_minus_persistent_nms
        ),
    }
    return record, arrays


def main() -> None:
    """Write the study record and compressed trace archive."""
    record, arrays = build_study()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "advanced_biological_bridge.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(DATA_DIR / "advanced_biological_bridge.npz", **arrays)


if __name__ == "__main__":
    main()
