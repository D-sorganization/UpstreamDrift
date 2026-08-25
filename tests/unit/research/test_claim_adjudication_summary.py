"""Tests for the reviewer-facing normalized claim adjudication summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.claim_adjudication_summary import (
    build_summary,
    validate_summary,
)
from scripts.research.proximal_distal_energy.claim_audit import (
    build_candidate_inventory,
)


@pytest.mark.unit
def test_summary_exposes_all_outcomes_and_open_evidence_boundaries() -> None:
    root = Path(__file__).resolve().parents[3]
    registry = root / (
        "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
    )

    summary = build_summary(registry)

    assert summary["claim_count"] == 313
    assert summary["outcome_counts"] == {
        "inconclusive": 5,
        "supported": 293,
        "untested": 15,
    }
    assert summary["evidence_qualification_counts"]["independent_followup_open"] > 0
    assert (
        summary["evidence_qualification_counts"]["governed_human_validation_open"] > 0
    )
    assert summary["evidence_tier_counts"]["external_empirical_human"] > 0
    assert (
        summary["source_independence_counts"]["multiple_independent_external_support"]
        > 0
    )
    assert summary["model_tier_counts"]["articulated_spatial"] > 0
    assert (
        summary["unresolved_replication_counts"]["governed_human_data_unavailable"] > 0
    )
    assert {row["claim_id"] for row in summary["claims"]} == {
        f"PD-CLAIM-{index:03d}" for index in range(2, 315)
    }
    rows = {row["claim_id"]: row for row in summary["claims"]}
    assert rows["PD-CLAIM-093"]["source_independence"] == (
        "multiple_independent_external_support"
    )
    assert (
        "governed_human_data_unavailable"
        in rows["PD-CLAIM-199"]["unresolved_replication_classes"]
    )
    assert all(row["evidence_tiers"] for row in rows.values())
    assert all(row["model_tiers"] for row in rows.values())
    family_rows = summary["claim_family_source_concentration"]
    assert sum(row["claim_count"] for row in family_rows) == 313
    assert any(
        row["concentration_flag"] == "project_authored_only" for row in family_rows
    )


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


@pytest.mark.unit
def test_generated_reviewer_projection_is_in_the_claim_inventory() -> None:
    root = Path(__file__).resolve().parents[3]
    data = root / "docs/research/proximal_distal_energy_transfer/data"
    registry = json.loads(
        (data / "claim_audit_registry.json").read_text(encoding="utf-8")
    )
    inventory = build_candidate_inventory(
        root / registry["paper"]["source"], repository_root=root
    )

    assert inventory["source_digest"] == registry["paper"]["source_digest"]
    reviewed_ids = {review["candidate_id"] for review in registry["candidate_reviews"]}
    summary_ids = {
        candidate["candidate_id"]
        for candidate in inventory["candidates"]
        if candidate["source_path"].endswith("/_claim_adjudication_summary.qmd")
    }
    assert summary_ids <= reviewed_ids


@pytest.mark.unit
def test_reviewer_chapter_has_pdf_breaks_and_wrappable_family_rows() -> None:
    root = Path(__file__).resolve().parents[3]
    chapter = (
        root / "docs/research/proximal_distal_energy_transfer/chapters/"
        "_claim_adjudication_summary.qmd"
    ).read_text(encoding="utf-8")

    assert chapter.count("\\newpage") == 2
    assert "| Claim Family | Claims | Source-Category Tuple | Flag |" in chapter
    assert "| `ch01_introduction`" not in chapter
    assert (
        "tuple is project only / one independent work / two or more independent works"
        in chapter
    )


@pytest.mark.unit
def test_byte_governed_summary_is_excluded_from_prettier() -> None:
    root = Path(__file__).resolve().parents[3]
    precommit = (root / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "|claim_adjudication_summary|claim_audit_registry|" in precommit
