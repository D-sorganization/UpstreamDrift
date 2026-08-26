"""Release-level numeric claim authority checks (#8918)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.build_claim_numeric_comparison_evidence import (
    validate_record,
)
from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    audit_registry_numeric_evidence,
)
from scripts.research.proximal_distal_energy.register_numeric_claim_evidence import (
    register,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
)


@pytest.mark.scientific
def test_complete_registry_has_executable_numeric_traceability() -> None:
    assert register(check=True) == {
        "claim_count": 315,
        "numeric_contract_count": 134,
        "mode": "check",
    }
    result = audit_registry_numeric_evidence(REGISTRY, repository_root=ROOT)
    assert result == {
        "claim_count": 315,
        "numeric_claim_count": 134,
        "numeric_literal_count": 445,
        "verified_numeric_literal_count": 445,
        "nondegenerate_comparison_count": 1,
        "evidence_scope_counts": {
            "local_json_value": 232,
            "registered_claim_value_not_independently_recomputed": 149,
            "registered_protocol_or_notation": 7,
            "reported_external_value": 57,
        },
        "completion_status": "complete",
    }


@pytest.mark.scientific
def test_registered_comparison_evidence_is_current_and_nondegenerate() -> None:
    result = validate_record()
    assert result["comparison_sample_count"] == 961
    assert result["completion_status"] == "complete"
