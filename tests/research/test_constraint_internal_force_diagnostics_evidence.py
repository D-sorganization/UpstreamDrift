"""Evidence contracts for the constraint and internal-force diagnostic."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.run_constraint_internal_force_diagnostics import (
    build_report,
    validate_report,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "constraint_internal_force_diagnostics.json"
)
REGISTRY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
)


def test_registered_report_is_deterministic_and_source_bound() -> None:
    registered = json.loads(REPORT.read_text(encoding="utf-8"))

    assert registered == build_report()
    assert validate_report(registered)["source_authority_count"] == 3


def test_report_separates_nullspace_meanings_and_adverse_cases() -> None:
    report = build_report()

    assert set(report["nullspace_semantics"]) == {
        "kinematic_velocity_nullspace",
        "point_force_measurement_nullspace",
        "full_hand_wrench_measurement_nullspace",
    }
    planar = report["planar_closed_loop"]
    assert planar["coordinate_scale_contract"] == {
        "angular_coordinate_scale_rad": 1.0,
        "translation_coordinate_scale_m": 0.75,
    }
    assert planar["regular_case"]["rank"] == 4
    assert planar["exact_singular_case"]["rank"] == 3
    assert planar["exact_singular_case"]["nullity"] == 2
    assert len(planar["scale_sensitivity_cases"]) == 3
    bilateral = report["bilateral_point_force"]
    assert bilateral["coincident_contact_case"]["rank"] == 3
    assert bilateral["coincident_contact_case"]["nullity"] == 3
    assert bilateral["registered_span_case"]["rank"] == 5
    assert bilateral["registered_span_case"]["nullity"] == 1
    assert bilateral["near_coincident_threshold_sensitivity"][-1]["rank"] == 3


def test_validation_rejects_conflation_scale_drift_and_human_promotion() -> None:
    conflated = deepcopy(build_report())
    conflated["nullspace_semantics"]["kinematic_velocity_nullspace"][
        "identifies_individual_hand_force"
    ] = True
    with pytest.raises(ValueError, match="kinematic nullspace"):
        validate_report(conflated)

    scale_drift = deepcopy(build_report())
    scale_drift["planar_closed_loop"]["coordinate_scale_contract"][
        "translation_coordinate_scale_m"
    ] = 1.0
    with pytest.raises(ValueError, match="coordinate scale"):
        validate_report(scale_drift)

    promoted = deepcopy(build_report())
    promoted["inference_status"]["human_strategy"] = "supported"
    with pytest.raises(ValueError, match="human strategy"):
        validate_report(promoted)

    lost = deepcopy(build_report())
    lost["bilateral_point_force"]["coincident_contact_case"]["rank"] = 5
    with pytest.raises(ValueError, match="coincident-contact"):
        validate_report(lost)


def test_atomic_claims_retain_scales_adverse_cases_and_human_boundary() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}

    assert {"PD-CLAIM-311", "PD-CLAIM-312"} <= set(claims)
    planar = claims["PD-CLAIM-311"]
    contact = claims["PD-CLAIM-312"]
    assert planar["adjudication_outcome"] == "supported"
    assert contact["adjudication_outcome"] == "supported"
    assert "human" in planar["uncertainty_boundary"].lower()
    assert "human" in contact["uncertainty_boundary"].lower()
    assert planar["numeric_evidence"]
    assert contact["numeric_evidence"]
