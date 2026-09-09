"""Provenance refresh must preserve independently reviewed claim evidence."""

from copy import deepcopy

import pytest

from scripts.research.proximal_distal_energy import (
    register_articulated_manufactured_solution_claims as registration,
)

pytestmark = [pytest.mark.unit]


def test_refresh_preserves_outcomes_numeric_evidence_order_and_inputs() -> None:
    reviewed = {
        "claim_id": "PD-CLAIM-299",
        "statement": "bounded manufactured conservation",
        "adjudication_outcome": "supported",
        "numeric_evidence": [{"path": "free_body.energy", "tolerance": 0.02}],
        "evidence_artifacts": ["old-record.json"],
        "last_verified_on": "2026-08-21",
    }
    unrelated = {"claim_id": "PD-CLAIM-1", "statement": "separate claim"}
    original = [reviewed, unrelated]
    snapshot = deepcopy(original)
    update = {
        "claim_id": "PD-CLAIM-299",
        "statement": reviewed["statement"],
        "evidence_artifacts": ["current-record.json"],
        "last_verified_on": "2026-09-08",
    }
    result = registration._refresh_claims(original, [update])
    assert [item["claim_id"] for item in result] == ["PD-CLAIM-299", "PD-CLAIM-1"]
    assert result[0]["adjudication_outcome"] == "supported"
    assert result[0]["numeric_evidence"] == reviewed["numeric_evidence"]
    assert result[0]["evidence_artifacts"] == ["current-record.json"]
    assert result[0]["last_verified_on"] == "2026-09-08"
    assert result[1] == unrelated
    assert original == snapshot


@pytest.mark.parametrize("field", ["statement", "model_domain", "falsifier"])
def test_changed_scientific_claim_requires_new_review(field: str) -> None:
    original = {"claim_id": "PD-CLAIM-297", field: "reviewed meaning"}
    update = {"claim_id": "PD-CLAIM-297", field: "different meaning"}
    with pytest.raises(ValueError, match="scientific.*review"):
        registration._refresh_claims([original], [update])


def test_initial_registration_never_invents_adjudication() -> None:
    original = [{"claim_id": "PD-CLAIM-1", "statement": "separate"}]
    new = {"claim_id": "PD-CLAIM-297", "statement": "new manufactured claim"}
    result = registration._refresh_claims(original, [new])
    assert result == [*original, new]
    assert "adjudication_outcome" not in result[1]
