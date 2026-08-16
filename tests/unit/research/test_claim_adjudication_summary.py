"""Tests for the reviewer-facing normalized claim adjudication summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.claim_adjudication_summary import (
    build_summary,
    validate_summary,
)


@pytest.mark.unit
def test_summary_exposes_all_outcomes_and_open_evidence_boundaries() -> None:
    root = Path(__file__).resolve().parents[3]
    registry = root / (
        "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
    )

    summary = build_summary(registry)

    assert summary["claim_count"] == 295
    assert summary["outcome_counts"] == {
        "inconclusive": 5,
        "supported": 275,
        "untested": 15,
    }
    assert summary["evidence_qualification_counts"]["independent_followup_open"] > 0
    assert (
        summary["evidence_qualification_counts"]["governed_human_validation_open"] > 0
    )
    assert {row["claim_id"] for row in summary["claims"]} == {
        f"PD-CLAIM-{index:03d}" for index in range(2, 297)
    }


@pytest.mark.unit
def test_committed_summary_matches_registry() -> None:
    root = Path(__file__).resolve().parents[3]
    summary = validate_summary(root)

    assert (
        summary["paper_source_digest"]
        == (
            json.loads(
                (
                    root
                    / "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
                ).read_text(encoding="utf-8")
            )["paper"]["source_digest"]
        )
    )
