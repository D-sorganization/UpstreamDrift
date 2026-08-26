"""Contracts for coordinate-force chapter claim adjudication."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import register_force_source_claims


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
CHAPTER = (
    "docs/research/proximal_distal_energy_transfer/chapters/"
    "_ch03ba_coordinate_force_sources.qmd"
)
pytestmark = pytest.mark.scientific


def test_force_source_claim_registration_is_complete_and_idempotent() -> None:
    registry = json.loads((DATA / "claim_audit_registry.json").read_text("utf-8"))
    inventory = json.loads((DATA / "claim_candidate_inventory.json").read_text("utf-8"))

    for _ in range(2):
        claims, assignments = register_force_source_claims._build_claims(
            inventory["candidates"]
        )
        register_force_source_claims._reconcile(
            registry,
            inventory,
            copy.deepcopy(claims),
            assignments,
        )

    chapter_candidates = {
        candidate["candidate_id"]
        for candidate in inventory["candidates"]
        if candidate["source_path"] == CHAPTER
    }
    reviews = {
        review["candidate_id"]: review for review in registry["candidate_reviews"]
    }
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}

    assert chapter_candidates
    assert chapter_candidates <= reviews.keys()
    assert {"PD-CLAIM-305", "PD-CLAIM-306", "PD-CLAIM-307"} <= claims.keys()
    assert set(claims["PD-CLAIM-307"]["evidence_artifacts"]) >= {
        "docs/research/proximal_distal_energy_transfer/data/force_source_optimization.json",
        "scripts/research/proximal_distal_energy/force_source_optimization.py",
        "tests/research/test_force_source_optimization.py",
    }
    for review in reviews.values():
        assert len(review["claim_ids"]) == len(set(review["claim_ids"]))
