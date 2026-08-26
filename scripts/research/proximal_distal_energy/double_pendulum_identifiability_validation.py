"""Fail-closed validation for registered identifiability evidence."""

from __future__ import annotations

from typing import Any

from scripts.research.proximal_distal_energy.double_pendulum_identifiability_contract import (
    COUNTEREXAMPLE_NAMES,
    SCHEMA_VERSION,
    TORQUE_NOISE_LEVELS_NM,
)
from scripts.research.proximal_distal_energy.numeric_evidence import (
    CANONICAL_SIGNIFICANT_DIGITS,
)


def validate_report_contract(
    report: dict[str, Any], expected: dict[str, object]
) -> dict[str, int]:
    """Reject lost counterexamples, dimensional rank, or promoted inference."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("physical_parameter_status") != (
        "structurally_non_identifiable_under_declared_model"
    ):
        raise ValueError(
            "physical parameters must remain structurally non-identifiable"
        )
    if report.get("practical_identifiability_status") != (
        "not_established_oracle_kinematics_lower_bound_only"
    ):
        raise ValueError("practical identifiability must remain unestablished")
    lower_bound = report.get("noise_aware_lower_bound_screen", {})
    if lower_bound.get("status") != "conditional_lower_bound_only":
        raise ValueError("noise-aware result must remain a conditional lower bound")
    cases = lower_bound.get("full_record_cases")
    if not isinstance(cases, list) or [
        case.get("torque_noise_sd_nm") for case in cases
    ] != list(TORQUE_NOISE_LEVELS_NM):
        raise ValueError("registered torque-noise cases are incomplete")
    if any(case.get("full_rank") is not True for case in cases):
        raise ValueError("registered lower-bound cases must retain base rank")
    physical_map = report.get("physical_parameter_map", {})
    if physical_map.get("rank") != 7 or physical_map.get("nullity") != 4:
        raise ValueError(
            "physical parameter map must retain rank seven and nullity four"
        )
    witness = physical_map.get("structural_rank_witness", {})
    if (
        physical_map.get("rank_basis") != "analytic_nonzero_minor"
        or witness.get("determinant", 0.0) <= 0.0
        or len(witness.get("parameter_columns", [])) != 7
    ):
        raise ValueError("physical parameter rank requires a positive analytic witness")
    record = report.get("finite_record_regressor", {})
    if record.get("rank") != 7 or record.get("full_base_coefficient_rank") is not True:
        raise ValueError("registered finite record must retain seven-column rank")
    if (
        record.get("rank_basis") != "nondimensional_base_coefficient_regressor"
        or record.get("raw_dimensional_conditioning_status") != "not_interpreted"
    ):
        raise ValueError("finite-record rank must retain its nondimensional basis")
    unit_fixture = record.get("unit_invariance", {})
    if (
        unit_fixture.get("max_abs_dimensionless_regressor_difference", float("inf"))
        > 1e-12
        or unit_fixture.get("rank_before_conversion") != 7
        or unit_fixture.get("rank_after_conversion") != 7
    ):
        raise ValueError(
            "equivalent coefficient units must preserve dimensionless rank"
        )
    scale_trials = record.get("scale_sensitivity")
    if (
        not isinstance(scale_trials, list)
        or len(scale_trials) != 3
        or any(trial.get("rank") != 7 for trial in scale_trials)
    ):
        raise ValueError("registered scale-sensitivity ranks must remain stable")
    scale_contract = report.get("nondimensional_scale_contract", {})
    if (
        len(scale_contract.get("coefficient_scales", [])) != 7
        or scale_contract.get("torque_scale_nm", 0.0) <= 0.0
        or scale_contract.get("rank_basis")
        != "nondimensional_base_coefficient_regressor"
    ):
        raise ValueError("a positive nondimensional scale contract is required")
    if report.get("numeric_representation_contract") != {
        "canonical_significant_digits": CANONICAL_SIGNIFICANT_DIGITS,
        "decision_precision": "full_precision_before_publication_rounding",
        "scope": "published_json_floats_only",
    }:
        raise ValueError("numeric representation contract is missing or stale")
    if report.get("zero_motion_killswitch", {}).get("rank") != 0:
        raise ValueError("zero-motion killswitch must have rank zero")
    counterexamples = report.get("exact_counterexamples")
    if not isinstance(counterexamples, list) or [
        item.get("name") for item in counterexamples
    ] != list(COUNTEREXAMPLE_NAMES):
        raise ValueError("three registered exact counterexamples are required")
    if any(
        item.get("base_coefficient_max_abs_difference", float("inf")) > 1e-10
        for item in counterexamples
    ):
        raise ValueError("counterexamples must preserve every base coefficient")
    if report != expected:
        raise ValueError("registered report differs from deterministic recomputation")
    return {
        "counterexample_count": len(counterexamples),
        "finite_record_rank": int(record["rank"]),
        "physical_parameter_nullity": int(physical_map["nullity"]),
    }
