"""Generate independent articulated manufactured-solution evidence (#8910)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.articulated_manufactured_solution import (
    evaluate_manufactured_constrained_motion,
    evaluate_manufactured_free_body,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data"
OUTPUT = DATA / "articulated_manufactured_solution.json"
SOURCE_PATHS = (
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.json",
    "docs/research/proximal_distal_energy_transfer/data/subject_scaled_closed_contact.npz",
    "scripts/research/proximal_distal_energy/articulated_manufactured_solution.py",
    "scripts/research/proximal_distal_energy/run_articulated_manufactured_solution.py",
    "scripts/research/proximal_distal_energy/register_articulated_manufactured_solution_claims.py",
    "scripts/research/proximal_distal_energy/spatial_full_body.py",
    "tests/research/test_articulated_manufactured_solution.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _free_record(result: Any) -> dict[str, Any]:
    return {
        "all_gates_pass": result.closed_form_check_passed,
        "independent_engine_difference_detected": (
            result.independent_engine_difference_detected
        ),
        "inverse_dynamics_relative_error": {
            "lagrange_mujoco": result.lagrange_mujoco_relative_error,
            "lagrange_pinocchio": result.lagrange_pinocchio_relative_error,
            "mujoco_pinocchio": result.mujoco_pinocchio_relative_error,
            "maximum": result.inverse_dynamics_residual,
        },
        "integration_step_error_rad": {
            str(step): error
            for step, error in sorted(result.integration_step_errors.items())
        },
        "richardson_orders": list(result.richardson_orders),
        "gravity_free_zero_torque_relative_drift": {
            "linear_momentum": result.linear_momentum_conservation_error,
            "angular_momentum": result.angular_momentum_conservation_error,
            "kinetic_energy": result.mechanical_energy_conservation_error,
        },
    }


def _constrained_record(result: Any) -> dict[str, Any]:
    return {
        "all_gates_pass": result.closed_form_check_passed,
        "independent_engine_difference_detected": (
            result.independent_engine_difference_detected
        ),
        "position_residual_m": result.constraint_residual,
        "velocity_residual_m_s": result.constraint_velocity_residual,
        "virtual_power_residual_w": result.constraint_virtual_power_w,
        "multiplier_relative_residual": result.lagrange_multiplier_residual,
        "cross_engine_multiplier_relative_residual": (
            result.action_reaction_residual_n
        ),
        "equilibrium_relative_residual": result.equilibrium_residual,
    }


def build_record() -> dict[str, Any]:
    """Execute the registered controls and return a release record."""

    import mujoco
    import pinocchio as pin

    pinocchio_version = require_robotics_pinocchio(pin)
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    free = evaluate_manufactured_free_body(
        model, q, duration_s=0.01, time_steps_s=(0.002, 0.001, 0.0005)
    )
    constrained = evaluate_manufactured_constrained_motion(
        model,
        q,
        duration_s=0.01,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=float(metadata["hand_contact_local_x_m"]),
    )
    return {
        "schema_version": "1.0.0",
        "study_id": "articulated-manufactured-solution-independent-v1",
        "classification": "synthetic_numerical_verification_not_human_evidence",
        "model": {
            "canonical_sha256": model.canonical_hash,
            "coordinate_count": model.nq,
            "profile": default_synthetic_profiles()[0].profile_id,
            "closed_state_index": [0, 6],
        },
        "engines": {
            "analytical": "lagrange_christoffel_finite_difference_mass_gradient",
            "mujoco": str(mujoco.__version__),
            "pinocchio": pinocchio_version,
        },
        "design": {
            "duration_s": 0.01,
            "time_steps_s": [0.002, 0.001, 0.0005],
            "registered_richardson_order_interval": [0.9, 1.1],
            "inverse_dynamics_relative_tolerance": 0.05,
            "conservation_relative_tolerance": 0.02,
            "conservation_scope": "gravity_free_zero_torque_free_floating_club_subtree",
            "killswitch": "add_10_nm_to_mujoco_inverse_and_require_gate_failure",
        },
        "free_body": _free_record(free),
        "constrained_motion": _constrained_record(constrained),
        "all_gates_pass": bool(
            free.closed_form_check_passed and constrained.closed_form_check_passed
        ),
        "limitations": [
            "The trajectories are manufactured and are not measured golf swings.",
            "The conservation rollout isolates the free club subtree because the pelvis tree is world-supported.",
            "Agreement verifies the declared model and operators, not anatomy or coaching strategy.",
        ],
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
    }


def write_record(path: Path = OUTPUT) -> Path:
    """Write the deterministic evidence record."""

    record = build_record()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    print(write_record())


if __name__ == "__main__":
    main()
