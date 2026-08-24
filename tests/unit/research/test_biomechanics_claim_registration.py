"""Tests for biomechanics evidence-bridge claim registration."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.register_biomechanics_evidence_bridge_claims import (
    CHAPTER,
    CLAIM_ID,
    register_claims,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.unit


def _load(name: str) -> dict:
    return json.loads((ARTICLE / "data" / name).read_text(encoding="utf-8"))


def test_registration_is_complete_reciprocal_and_idempotent() -> None:
    registry = _load("claim_audit_registry.json")
    inventory = _load("claim_candidate_inventory.json")
    sources = _load("biomechanics_source_register.json")

    once = register_claims(copy.deepcopy(registry), inventory, sources)
    twice = register_claims(copy.deepcopy(once), inventory, sources)

    assert once == twice
    claim = next(item for item in once["claims"] if item["claim_id"] == CLAIM_ID)
    reviews = {item["candidate_id"]: item for item in once["candidate_reviews"]}
    assert len(claim["candidate_ids"]) == 36
    assert all(
        CLAIM_ID in reviews[item]["claim_ids"] for item in claim["candidate_ids"]
    )
    assert once["paper"]["source_digest"] == inventory["source_digest"]


def test_registration_fails_closed_when_generated_candidate_count_changes() -> None:
    registry = _load("claim_audit_registry.json")
    inventory = _load("claim_candidate_inventory.json")
    sources = _load("biomechanics_source_register.json")
    removed = False
    retained = []
    for candidate in inventory["candidates"]:
        if candidate["source_path"] == CHAPTER and not removed:
            removed = True
            continue
        retained.append(candidate)
    inventory["candidates"] = retained

    with pytest.raises(ValueError, match="candidate count changed"):
        register_claims(registry, inventory, sources)
