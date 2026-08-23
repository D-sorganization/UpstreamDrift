"""Contracts for articulated uncertainty and structural-authority claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
CLAIM_IDS = {f"PD-CLAIM-{value}" for value in range(305, 315)}


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads((DATA / "claim_audit_registry.json").read_text(encoding="utf-8"))


def test_current_candidate_census_is_fully_adjudicated(registry) -> None:
    inventory = json.loads(
        (DATA / "claim_candidate_inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["candidate_count"] == 1126
    assert registry["paper"]["source_digest"] == inventory["source_digest"]
    assert len(registry["candidate_reviews"]) == inventory["candidate_count"]
    assert {row["candidate_id"] for row in registry["candidate_reviews"]} == {
        row["candidate_id"] for row in inventory["candidates"]
    }


def test_uncertainty_and_structural_claim_set_is_complete(registry) -> None:
    claims = {
        row["claim_id"]: row
        for row in registry["claims"]
        if row["claim_id"] in CLAIM_IDS
    }
    assert set(claims) == CLAIM_IDS
    assert claims["PD-CLAIM-306"]["published_status"] == (
        "finite_energy_closed_but_contact_domain_adverse"
    )
    assert claims["PD-CLAIM-310"]["published_status"] == (
        "qualified_with_one_retained_boundary_failure"
    )
    assert claims["PD-CLAIM-311"]["published_status"] == (
        "registered_but_not_yet_executed"
    )
    assert claims["PD-CLAIM-312"]["published_status"] == (
        "completed_with_retained_failures"
    )
    assert claims["PD-CLAIM-314"]["published_status"] == (
        "completed_model_sensitivity_result"
    )


def test_current_main_claim_identities_are_not_overwritten(registry) -> None:
    claims = {row["claim_id"]: row for row in registry["claims"]}

    assert claims["PD-CLAIM-297"]["classification"] == (
        "articulated_manufactured_solution_design"
    )
    assert claims["PD-CLAIM-304"]["classification"] == (
        "native_constraint_formulation_inference_boundary"
    )


def test_completed_headline_claims_cite_the_qualified_result(registry) -> None:
    claims = {row["claim_id"]: row for row in registry["claims"]}
    result = (
        "docs/research/proximal_distal_energy_transfer/data/"
        "articulated_headline_uncertainty.json"
    )
    for claim_id in ("PD-CLAIM-312", "PD-CLAIM-313", "PD-CLAIM-314"):
        claim = claims[claim_id]
        assert result in claim["evidence_artifacts"]
        boundary = claim["uncertainty_boundary"].lower()
        assert any(word in boundary for word in ("human", "population", "participant"))


def test_completed_headline_result_preserves_counts_and_failures(registry) -> None:
    claim = next(row for row in registry["claims"] if row["claim_id"] == "PD-CLAIM-314")
    assert "80--182" in claim["statement"]
    assert "-46 to +56" in claim["statement"]
    assert "0/384" in claim["statement"]
    assert "grip-damping" in claim["statement"]


def test_structural_failure_and_dynamic_boundary_remain_distinct(registry) -> None:
    claims = {row["claim_id"]: row for row in registry["claims"]}
    result = claims["PD-CLAIM-310"]
    boundary = claims["PD-CLAIM-311"]
    assert "51/52" in result["statement"]
    assert "case-0 phase-12" in result["statement"]
    assert "does not establish sensitivity" in boundary["statement"]
    assert "not_yet_executed" in boundary["published_status"]


def test_validity_horizon_claim_separates_model_tiers(registry) -> None:
    claim = next(row for row in registry["claims"] if row["claim_id"] == "PD-CLAIM-273")
    assert "point-contact tier" in claim["statement"]
    assert "distributed-grip, shaft, and ground tiers" in claim["statement"]
    assert "cannot be pooled" in claim["uncertainty_boundary"]
