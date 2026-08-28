from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    audit_claim_numeric_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRY = ARTICLE / "data/claim_audit_registry.json"
pytestmark = pytest.mark.scientific


def test_nonlinear_controller_claims_are_numeric_and_scope_bounded() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    expected = {"PD-CLAIM-324", "PD-CLAIM-325", "PD-CLAIM-326"}
    assert expected <= set(claims)
    for claim_id in expected:
        claim = claims[claim_id]
        result = audit_claim_numeric_evidence(claim, repository_root=ROOT)
        assert result["literal_count"] == result["verified_count"]
        assert "human" in claim["uncertainty_boundary"].lower()


def test_nonlinear_controller_release_claim_is_numerical_prerequisite_only() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    releases = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }
    assert releases["nonlinear_controller_numerical_qualification"] == {
        "release_claim_key": "nonlinear_controller_numerical_qualification",
        "published_status": (
            "supported_as_registered_numerical_prerequisite_without_evaluation"
        ),
        "audit_state": "reviewed_as_numerical_prerequisite_without_ranking",
    }
