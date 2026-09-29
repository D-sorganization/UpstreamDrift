"""Claim-ledger contracts for issue #9104."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    register_double_pendulum_identifiability_claims as registration,
)
from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    audit_claim_numeric_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
pytestmark = pytest.mark.scientific


def test_identifiability_claim_registration_is_complete_and_idempotent() -> None:
    registry = json.loads((DATA / "claim_audit_registry.json").read_text("utf-8"))
    inventory = json.loads((DATA / "claim_candidate_inventory.json").read_text("utf-8"))

    for _ in range(2):
        built_claims, assignments, abstract_id = registration._build_claims(
            inventory["candidates"]
        )
        registration._reconcile(
            registry,
            inventory,
            copy.deepcopy(built_claims),
            assignments,
            abstract_id,
        )

    claims_by_id = {claim["claim_id"]: claim for claim in registry["claims"]}
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    assert claims_by_id.keys() >= registration.CLAIM_IDS
    assert registration.OLD_ABSTRACT_ID not in reviews
    assert registration.OLD_APPENDIX_ID not in reviews
    assert registration.OLD_ABSTRACT_ID not in {
        candidate_id
        for claim in claims_by_id.values()
        for candidate_id in claim["candidate_ids"]
    }
    for candidate_id, claim_ids in assignments.items():
        assert candidate_id in reviews
        assert set(claim_ids) <= set(reviews[candidate_id]["claim_ids"])
    for claim_id in registration.CLAIM_IDS:
        result = audit_claim_numeric_evidence(
            claims_by_id[claim_id], repository_root=ROOT
        )
        assert result["literal_count"] == result["verified_count"]


def test_current_registry_contains_identifiability_claims_and_release_keys() -> None:
    registry = json.loads((DATA / "claim_audit_registry.json").read_text("utf-8"))
    claim_ids = {claim["claim_id"] for claim in registry["claims"]}
    release_keys = {
        item["release_claim_key"] for item in registry["release_claim_inventory"]
    }

    assert claim_ids >= registration.CLAIM_IDS
    assert {
        "double_pendulum_base_coefficient_excitation",
        "double_pendulum_physical_parameter_identifiability",
        "double_pendulum_practical_identifiability",
    } <= release_keys
