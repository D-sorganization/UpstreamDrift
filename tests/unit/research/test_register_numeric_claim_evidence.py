"""Contracts for reviewed numeric-claim registration (#8918)."""

from __future__ import annotations

import copy

import pytest

from scripts.research.proximal_distal_energy.register_numeric_claim_evidence import (
    apply_numeric_contracts,
    statement_digest,
)


STATEMENT = "The declared result is 12.5 N across 3 cases."


def _registry() -> dict[str, object]:
    return {
        "claims": [
            {
                "claim_id": "PD-CLAIM-TEST",
                "statement": STATEMENT,
                "evidence_artifacts": ["result.json"],
            },
            {
                "claim_id": "PD-CLAIM-NARRATIVE",
                "statement": "This statement has no number.",
                "evidence_artifacts": ["result.json"],
            },
        ]
    }


def _contracts() -> dict[str, object]:
    return {
        "schema_version": "claim-numeric-contract-v1",
        "claims": [
            {
                "claim_id": "PD-CLAIM-TEST",
                "statement_sha256": statement_digest(STATEMENT),
                "numeric_evidence": [
                    {
                        "literal_id": "12.5#1",
                        "artifact": "result.json",
                        "json_pointer": "/result",
                        "evidence_scope": "local_json_value",
                        "scale": 1.0,
                        "offset": 0.0,
                        "atol": 0.05,
                        "rtol": 0.0,
                    },
                    {
                        "literal_id": "3#1",
                        "artifact": "reported.json",
                        "json_pointer": "/claims/PD-CLAIM-TEST/0/value",
                        "evidence_scope": "registered_protocol_or_notation",
                        "scale": 1.0,
                        "offset": 0.0,
                        "atol": 0.0,
                        "rtol": 0.0,
                    },
                ],
                "numeric_comparisons": [
                    {
                        "comparison_id": "engine-parity",
                        "artifact": "comparison.json",
                        "reference_pointer": "/reference",
                        "candidate_pointer": "/candidate",
                        "require_nondegenerate": True,
                        "atol": 0.01,
                        "rtol": 0.0,
                    }
                ],
            }
        ],
    }


@pytest.mark.unit
def test_registration_applies_complete_maps_and_declares_new_artifact() -> None:
    source = _registry()
    updated = apply_numeric_contracts(source, _contracts())

    claim = updated["claims"][0]
    assert claim["numeric_evidence"] == _contracts()["claims"][0]["numeric_evidence"]
    assert claim["evidence_artifacts"] == [
        "result.json",
        "reported.json",
        "comparison.json",
    ]
    assert (
        claim["numeric_comparisons"] == _contracts()["claims"][0]["numeric_comparisons"]
    )
    assert source == _registry()


@pytest.mark.unit
def test_registration_rejects_changed_statement() -> None:
    registry = _registry()
    registry["claims"][0]["statement"] = "The declared result is 99 N."

    with pytest.raises(ValueError, match="stale numeric contract statement digest"):
        apply_numeric_contracts(registry, _contracts())


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_registration_requires_exact_numeric_claim_coverage(mutation: str) -> None:
    contracts = _contracts()
    if mutation == "missing":
        contracts["claims"] = []
    else:
        extra = copy.deepcopy(contracts["claims"][0])
        extra["claim_id"] = "PD-CLAIM-EXTRA"
        contracts["claims"].append(extra)

    with pytest.raises(ValueError, match="claim coverage mismatch"):
        apply_numeric_contracts(_registry(), contracts)


@pytest.mark.unit
def test_registration_rejects_stale_literal_inventory() -> None:
    contracts = _contracts()
    contracts["claims"][0]["numeric_evidence"].pop()

    with pytest.raises(ValueError, match="stale numeric contract literal inventory"):
        apply_numeric_contracts(_registry(), contracts)
