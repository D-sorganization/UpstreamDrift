"""Contracts for articulated uncertainty and structural-authority claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
CLAIM_IDS = {f"PD-CLAIM-{value}" for value in range(297, 306)}


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads((DATA / "claim_audit_registry.json").read_text(encoding="utf-8"))


def test_current_candidate_census_is_fully_adjudicated(registry) -> None:
    inventory = json.loads(
        (DATA / "claim_candidate_inventory.json").read_text(encoding="utf-8")
    )
    assert inventory["candidate_count"] == 1092
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
    assert claims["PD-CLAIM-298"]["published_status"] == (
        "finite_energy_closed_but_contact_domain_adverse"
    )
    assert claims["PD-CLAIM-302"]["published_status"] == (
        "qualified_with_one_retained_boundary_failure"
    )
    assert claims["PD-CLAIM-303"]["published_status"] == (
        "registered_but_not_yet_executed"
    )
    assert claims["PD-CLAIM-304"]["published_status"] == (
        "registered_execution_in_progress"
    )


def test_in_progress_headline_claims_do_not_cite_partial_result(registry) -> None:
    claims = {row["claim_id"]: row for row in registry["claims"]}
    partial = (
        "docs/research/proximal_distal_energy_transfer/data/"
        "articulated_headline_uncertainty.json"
    )
    for claim_id in ("PD-CLAIM-304", "PD-CLAIM-305"):
        claim = claims[claim_id]
        assert partial not in claim["evidence_artifacts"]
        boundary = claim["uncertainty_boundary"].lower()
        assert any(word in boundary for word in ("human", "population", "participant"))


def test_structural_failure_and_dynamic_boundary_remain_distinct(registry) -> None:
    claims = {row["claim_id"]: row for row in registry["claims"]}
    result = claims["PD-CLAIM-302"]
    boundary = claims["PD-CLAIM-303"]
    assert "51/52" in result["statement"]
    assert "case-0 phase-12" in result["statement"]
    assert "does not establish sensitivity" in boundary["statement"]
    assert "not_yet_executed" in boundary["published_status"]


def test_validity_horizon_claim_separates_model_tiers(registry) -> None:
    claim = next(row for row in registry["claims"] if row["claim_id"] == "PD-CLAIM-273")
    assert "point-contact tier" in claim["statement"]
    assert "distributed-grip, shaft, and ground tiers" in claim["statement"]
    assert "cannot be pooled" in claim["uncertainty_boundary"]
