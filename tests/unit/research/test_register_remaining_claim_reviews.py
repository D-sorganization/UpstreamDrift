from __future__ import annotations

import pytest

from scripts.research.proximal_distal_energy.register_remaining_claim_reviews import (
    _reconcile_reciprocal_claim_reviews,
)

pytestmark = pytest.mark.unit


def test_reconciliation_prunes_candidate_ids_absent_from_current_inventory() -> None:
    current_id = "PD-CAND-current"
    stale_id = "PD-CAND-stale"
    registry = {
        "candidate_reviews": [
            {
                "candidate_id": current_id,
                "disposition": "material_claims_mapped",
                "claim_ids": ["PD-CLAIM-001"],
            }
        ],
        "claims": [
            {
                "claim_id": "PD-CLAIM-001",
                "candidate_ids": [stale_id, current_id],
                "source_locations": ["old.qmd:1", "current.qmd:12"],
            }
        ],
    }
    by_id = {
        current_id: {
            "source_path": "current.qmd",
            "line_start": 12,
        }
    }
    claims = {"PD-CLAIM-001": registry["claims"][0]}

    _reconcile_reciprocal_claim_reviews(registry, by_id, claims)

    assert registry["claims"][0]["candidate_ids"] == [current_id]
    assert registry["claims"][0]["source_locations"] == ["current.qmd:12"]
    assert registry["candidate_reviews"][0]["claim_ids"] == ["PD-CLAIM-001"]
