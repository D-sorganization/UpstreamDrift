"""Tests for the proximal-to-distal paper claim-audit authority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.claim_audit import (
    build_candidate_inventory,
    validate_registry,
)


@pytest.mark.unit
def test_candidate_inventory_expands_includes_with_source_locations(
    tmp_path: Path,
) -> None:
    chapter = tmp_path / "chapter.qmd"
    chapter.write_text(
        "# Result\n\nThe model produced 12.0 m/s [@source].\n",
        encoding="utf-8",
    )
    master = tmp_path / "paper.qmd"
    master.write_text(
        '---\ntitle: "Paper"\n---\n\n{{< include chapter.qmd >}}\n',
        encoding="utf-8",
    )

    inventory = build_candidate_inventory(master, repository_root=tmp_path)

    assert inventory["candidate_count"] == 1
    candidate = inventory["candidates"][0]
    assert candidate["source_path"] == "chapter.qmd"
    assert candidate["line_start"] == 3
    assert candidate["line_end"] == 3
    assert candidate["citation_keys"] == ["source"]
    assert candidate["has_numeric_content"] is True


@pytest.mark.unit
def test_candidate_inventory_includes_quarto_abstract(tmp_path: Path) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text(
        "---\n"
        'title: "Paper"\n'
        "abstract: |\n"
        "  The registered model reaches 18.2 m/s.\n"
        "  This result remains conditional [@study].\n"
        "keywords:\n"
        "  - mechanics\n"
        "---\n\n"
        "Body text without a separate quantitative claim.\n",
        encoding="utf-8",
    )

    inventory = build_candidate_inventory(master, repository_root=tmp_path)

    abstract = next(item for item in inventory["candidates"] if item["line_start"] == 4)
    assert abstract["line_end"] == 5
    assert abstract["citation_keys"] == ["study"]
    assert "18.2 m/s" in abstract["text"]


@pytest.mark.unit
def test_registry_rejects_duplicate_claim_identifiers(tmp_path: Path) -> None:
    registry = {
        "schema_version": "proximal-distal-claim-audit-v1",
        "paper": {"source": "paper.qmd", "source_digest": "a" * 64},
        "audit_scope": {"completion_status": "in_progress"},
        "dependencies": [],
        "research_collections": [],
        "release_claim_inventory": [],
        "claims": [
            {"claim_id": "PD-CLAIM-001"},
            {"claim_id": "PD-CLAIM-001"},
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate claim_id"):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
def test_repository_registry_matches_release_claims_and_remains_open() -> None:
    root = Path(__file__).resolve().parents[3]
    registry_path = root / (
        "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
    )

    result = validate_registry(registry_path, repository_root=root)

    assert result["release_claim_count"] >= 18
    assert result["registered_claim_count"] >= 5
    assert result["completion_status"] == "in_progress"
    assert result["unadjudicated_candidate_count"] > 0
