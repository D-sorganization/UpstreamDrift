"""Generate or validate double-pendulum parameter-identifiability evidence."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.research.proximal_distal_energy.double_pendulum_identifiability import (
    BASE_COEFFICIENT_NAMES,
    PHYSICAL_PARAMETER_NAMES,
    DoublePendulumPhysicalParameters,
    nondimensional_regressor,
    parameter_map_jacobian,
    physical_parameter_rank_witness,
    stacked_inverse_dynamics_regressor,
)
from scripts.research.proximal_distal_energy.double_pendulum_identifiability_contract import (
    INFERENCE_BOUNDARY,
    RANK_TOLERANCE,
    SCHEMA_VERSION,
)
from scripts.research.proximal_distal_energy.double_pendulum_identifiability_reporting import (
    counterexample_payload,
    cumulative_rank_payload,
    noise_aware_lower_bound_payload,
    parameter_record,
    rank_payload,
    registered_scales,
    scale_contract_payload,
    scale_sensitivity_payload,
    unit_invariance_payload,
)
from scripts.research.proximal_distal_energy.double_pendulum_identifiability_validation import (
    validate_report_contract,
)
from scripts.research.proximal_distal_energy.local_linear_diagnostics import (
    rank_diagnostic,
)
from scripts.research.proximal_distal_energy.numeric_evidence import (
    CANONICAL_SIGNIFICANT_DIGITS,
    canonicalize_published_numbers,
)
from scripts.research.proximal_distal_energy.run_experiments import (
    DT,
    HORIZON,
    INITIAL_Q,
    rollout_program,
    trace_accelerations,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.simulation_backends import GolfModelParams


REPORT_PATH = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "double_pendulum_identifiability.json"
)
PROGRAM = {
    "onset_s": 0.10,
    "shoulder_torque_nm": 60.0,
    "wrist_drive_nm": 15.0,
    "wrist_restrain_nm": 10.0,
}


def build_report() -> dict[str, object]:
    """Build deterministic exact-map and finite-record rank evidence."""
    model = GolfModelParams.default()
    physical = DoublePendulumPhysicalParameters.from_model(model)
    program = restrain_then_drive_program(
        PROGRAM["shoulder_torque_nm"],
        PROGRAM["wrist_drive_nm"],
        PROGRAM["wrist_restrain_nm"],
        PROGRAM["onset_s"],
    )
    time, q, velocity, controls = rollout_program(model, program)
    impact = find_impact(time, q, velocity, PlanarInertials.from_params(model))
    if impact is None:
        raise RuntimeError("registered reference rollout has no delivery event")
    impact_time_s, impact_speed_m_s, impact_arm_angle_rad = impact
    delivery_index = int(np.argmin(np.abs(time - impact_time_s)))
    retained = slice(0, delivery_index + 1)
    acceleration = trace_accelerations(
        model, q[retained], velocity[retained], controls[retained]
    )
    regressor = stacked_inverse_dynamics_regressor(
        q[retained], velocity[retained], acceleration
    )
    base_coefficients = physical.base_coefficients()
    scales = registered_scales(base_coefficients, controls[retained])
    dimensionless = nondimensional_regressor(regressor, scales)
    record_rank = rank_diagnostic(dimensionless, RANK_TOLERANCE)
    zero_regressor = stacked_inverse_dynamics_regressor(
        np.zeros((8, 2)), np.zeros((8, 2)), np.zeros((8, 2))
    )
    zero_rank = rank_diagnostic(
        nondimensional_regressor(zero_regressor, scales), RANK_TOLERANCE
    )
    physical_jacobian = parameter_map_jacobian(physical)
    structural_witness = physical_parameter_rank_witness(physical)
    report = {
        "base_coefficient_finite_record_status": (
            "full_rank_for_registered_synthetic_record"
        ),
        "base_coefficients": dict(
            zip(BASE_COEFFICIENT_NAMES, base_coefficients, strict=True)
        ),
        "classification": (
            "exact_reduced_physical_map_with_finite_record_regressor_rank"
        ),
        "exact_counterexamples": counterexample_payload(physical),
        "falsifiers": [
            "The analytical regressor fails to reconstruct canonical ODE inverse dynamics.",
            "Any registered distinct physical counterexample changes a base coefficient.",
            "The declared physical-map Jacobian does not have rank seven and nullity four.",
            "The zero-motion killswitch retains a nonzero regressor rank.",
            "The oracle-kinematics noise bound is promoted to practical or participant identifiability.",
            "Equivalent coefficient units change the dimensionless rank decision.",
        ],
        "finite_record_regressor": {
            **rank_payload(record_rank),
            "cumulative_rank": cumulative_rank_payload(dimensionless),
            "full_base_coefficient_rank": record_rank.rank
            == len(BASE_COEFFICIENT_NAMES),
            "raw_dimensional_conditioning_status": "not_interpreted",
            "rank_basis": "nondimensional_base_coefficient_regressor",
            "sample_count": delivery_index + 1,
            "scale_sensitivity": scale_sensitivity_payload(regressor, scales),
            "time_span_s": [float(time[0]), float(time[delivery_index])],
            "unit_invariance": unit_invariance_payload(regressor, scales),
        },
        "inference_boundary": INFERENCE_BOUNDARY,
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9104",
        "model_tier": "analytical_double_pendulum",
        "noise_aware_lower_bound_screen": noise_aware_lower_bound_payload(
            regressor, base_coefficients, scales
        ),
        "nondimensional_scale_contract": scale_contract_payload(scales),
        "numeric_representation_contract": {
            "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS,
            "decision_precision": "full_precision_before_publication_rounding",
            "scope": "published_json_floats_only",
        },
        "parent_issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9027",
        "participant_status": "not_evaluated",
        "physical_parameter_map": {
            "jacobian": physical_jacobian.tolist(),
            "jacobian_conditioning_status": "not_interpreted_dimensioned_columns",
            "nullity": len(PHYSICAL_PARAMETER_NAMES) - len(BASE_COEFFICIENT_NAMES),
            "rank": len(BASE_COEFFICIENT_NAMES),
            "rank_basis": "analytic_nonzero_minor",
            "structural_rank_witness": {
                "closed_form": structural_witness.closed_form,
                "determinant": structural_witness.determinant,
                "parameter_columns": list(structural_witness.parameter_columns),
            },
        },
        "physical_parameter_nonuniqueness_status": (
            "established_by_exact_invariance_families"
        ),
        "physical_parameter_status": (
            "structurally_non_identifiable_under_declared_model"
        ),
        "physical_parameters": parameter_record(physical),
        "practical_identifiability_status": (
            "not_established_oracle_kinematics_lower_bound_only"
        ),
        "reference_rollout": {
            "delivery_event": {
                "arm_angle_rad": impact_arm_angle_rad,
                "clubhead_speed_m_s": impact_speed_m_s,
                "nearest_sample_index": delivery_index,
                "nearest_sample_time_s": float(time[delivery_index]),
                "time_s": impact_time_s,
            },
            "dt_s": DT,
            "horizon_steps": HORIZON,
            "initial_q_rad": list(INITIAL_Q),
            "model_parameters": "GolfModelParams.default()",
            "program": PROGRAM,
            "scope": "synthetic reference trajectory; not participant evidence",
        },
        "schema_version": SCHEMA_VERSION,
        "zero_motion_killswitch": rank_payload(zero_rank),
    }
    return cast(
        dict[str, object],
        canonicalize_published_numbers(
            report, context="published double-pendulum identifiability evidence"
        ),
    )


def validate_report(report: dict[str, Any]) -> dict[str, int]:
    """Validate the artifact against fail-closed and deterministic contracts."""
    return validate_report_contract(report, build_report())


def main(argv: list[str] | None = None) -> int:
    """Write or validate the registered evidence artifact."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    args = parser.parse_args(argv)
    if args.action == "write":
        payload = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
        REPORT_PATH.write_text(payload, encoding="utf-8")
        print(REPORT_PATH)
        return 0
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_report(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
