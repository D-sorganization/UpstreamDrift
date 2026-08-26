"""Generate or validate double-pendulum parameter-identifiability evidence."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
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
    CoefficientScaleContract,
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
    RankDiagnostic,
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


@dataclass(frozen=True)
class _ReferenceEvidence:
    physical: DoublePendulumPhysicalParameters
    time: np.ndarray
    delivery_index: int
    impact_time_s: float
    impact_speed_m_s: float
    impact_arm_angle_rad: float
    regressor: np.ndarray
    base_coefficients: np.ndarray
    scales: CoefficientScaleContract
    record_rank: RankDiagnostic
    zero_rank: RankDiagnostic


def _build_reference_evidence() -> _ReferenceEvidence:
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
    return _ReferenceEvidence(
        physical=physical,
        time=time,
        delivery_index=delivery_index,
        impact_time_s=impact_time_s,
        impact_speed_m_s=impact_speed_m_s,
        impact_arm_angle_rad=impact_arm_angle_rad,
        regressor=regressor,
        base_coefficients=base_coefficients,
        scales=scales,
        record_rank=record_rank,
        zero_rank=zero_rank,
    )


def _physical_parameter_map(evidence: _ReferenceEvidence) -> dict[str, object]:
    structural_witness = physical_parameter_rank_witness(evidence.physical)
    return {
        "jacobian": parameter_map_jacobian(evidence.physical).tolist(),
        "jacobian_conditioning_status": "not_interpreted_dimensioned_columns",
        "nullity": len(PHYSICAL_PARAMETER_NAMES) - len(BASE_COEFFICIENT_NAMES),
        "rank": len(BASE_COEFFICIENT_NAMES),
        "rank_basis": "analytic_nonzero_minor",
        "structural_rank_witness": {
            "closed_form": structural_witness.closed_form,
            "determinant": structural_witness.determinant,
            "parameter_columns": list(structural_witness.parameter_columns),
        },
    }


def _reference_rollout(evidence: _ReferenceEvidence) -> dict[str, object]:
    index = evidence.delivery_index
    return {
        "delivery_event": {
            "arm_angle_rad": evidence.impact_arm_angle_rad,
            "clubhead_speed_m_s": evidence.impact_speed_m_s,
            "nearest_sample_index": index,
            "nearest_sample_time_s": float(evidence.time[index]),
            "time_s": evidence.impact_time_s,
        },
        "dt_s": DT,
        "horizon_steps": HORIZON,
        "initial_q_rad": list(INITIAL_Q),
        "model_parameters": "GolfModelParams.default()",
        "program": PROGRAM,
        "scope": "synthetic reference trajectory; not participant evidence",
    }


def build_report() -> dict[str, object]:
    """Build deterministic exact-map and finite-record rank evidence."""
    evidence = _build_reference_evidence()
    index = evidence.delivery_index
    report = {
        "base_coefficient_finite_record_status": (
            "full_rank_for_registered_synthetic_record"
        ),
        "base_coefficients": dict(
            zip(BASE_COEFFICIENT_NAMES, evidence.base_coefficients, strict=True)
        ),
        "classification": (
            "exact_reduced_physical_map_with_finite_record_regressor_rank"
        ),
        "exact_counterexamples": counterexample_payload(evidence.physical),
        "falsifiers": [
            "The analytical regressor fails to reconstruct canonical ODE inverse dynamics.",
            "Any registered distinct physical counterexample changes a base coefficient.",
            "The declared physical-map Jacobian does not have rank seven and nullity four.",
            "The zero-motion killswitch retains a nonzero regressor rank.",
            "The oracle-kinematics noise bound is promoted to practical or participant identifiability.",
            "Equivalent coefficient units change the dimensionless rank decision.",
        ],
        "finite_record_regressor": {
            **rank_payload(evidence.record_rank),
            "cumulative_rank": cumulative_rank_payload(
                nondimensional_regressor(evidence.regressor, evidence.scales)
            ),
            "full_base_coefficient_rank": evidence.record_rank.rank
            == len(BASE_COEFFICIENT_NAMES),
            "raw_dimensional_conditioning_status": "not_interpreted",
            "rank_basis": "nondimensional_base_coefficient_regressor",
            "sample_count": index + 1,
            "scale_sensitivity": scale_sensitivity_payload(
                evidence.regressor, evidence.scales
            ),
            "time_span_s": [float(evidence.time[0]), float(evidence.time[index])],
            "unit_invariance": unit_invariance_payload(
                evidence.regressor, evidence.scales
            ),
        },
        "inference_boundary": INFERENCE_BOUNDARY,
        "issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9104",
        "model_tier": "analytical_double_pendulum",
        "noise_aware_lower_bound_screen": noise_aware_lower_bound_payload(
            evidence.regressor, evidence.base_coefficients, evidence.scales
        ),
        "nondimensional_scale_contract": scale_contract_payload(evidence.scales),
        "numeric_representation_contract": {
            "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS,
            "decision_precision": "full_precision_before_publication_rounding",
            "scope": "published_json_floats_only",
        },
        "parent_issue": "https://github.com/D-sorganization/UpstreamDrift/issues/9027",
        "participant_status": "not_evaluated",
        "physical_parameter_map": _physical_parameter_map(evidence),
        "physical_parameter_nonuniqueness_status": (
            "established_by_exact_invariance_families"
        ),
        "physical_parameter_status": (
            "structurally_non_identifiable_under_declared_model"
        ),
        "physical_parameters": parameter_record(evidence.physical),
        "practical_identifiability_status": (
            "not_established_oracle_kinematics_lower_bound_only"
        ),
        "reference_rollout": _reference_rollout(evidence),
        "schema_version": SCHEMA_VERSION,
        "zero_motion_killswitch": rank_payload(evidence.zero_rank),
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
