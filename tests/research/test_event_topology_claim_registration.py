from __future__ import annotations

import json
from pathlib import Path

from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    audit_claim_numeric_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
)


def test_event_topology_claims_are_registered_and_numerically_traceable() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}

    assert {"PD-CLAIM-321", "PD-CLAIM-322", "PD-CLAIM-323"} <= set(claims)
    for claim_id in ("PD-CLAIM-321", "PD-CLAIM-322", "PD-CLAIM-323"):
        result = audit_claim_numeric_evidence(claims[claim_id], repository_root=ROOT)
        assert result["literal_count"] == result["verified_count"]


def test_event_topology_release_claim_preserves_synthetic_boundary() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    release = {
        item["release_claim_key"]: item for item in registry["release_claim_inventory"]
    }

    assert release["global_event_topology_robustness"] == {
        "release_claim_key": "global_event_topology_robustness",
        "published_status": (
            "supported_for_registered_synthetic_topology_model_scenarios"
        ),
        "audit_state": "reviewed_as_synthetic_global_topology_robustness",
    }
