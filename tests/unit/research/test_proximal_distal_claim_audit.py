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
def test_candidate_inventory_excludes_cross_references_from_citations(
    tmp_path: Path,
) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text(
        "A result is shown in @sec-results and @fig-speed, and agrees with "
        "[@study2026].\n",
        encoding="utf-8",
    )

    inventory = build_candidate_inventory(master, repository_root=tmp_path)

    candidate = inventory["candidates"][0]
    assert candidate["citation_keys"] == ["study2026"]
    assert candidate["priority_score"] >= 4
    assert "assertive_language" in candidate["triage_flags"]


@pytest.mark.unit
def test_candidate_id_is_stable_when_unrelated_lines_are_inserted_above(
    tmp_path: Path,
) -> None:
    master = tmp_path / "paper.qmd"
    claim = "The declared model produces 12.0 m/s [@study].\n"
    master.write_text(claim, encoding="utf-8")
    before = build_candidate_inventory(master, repository_root=tmp_path)["candidates"][
        0
    ]

    master.write_text("Unrelated context appears here.\n\n" + claim, encoding="utf-8")
    after = next(
        candidate
        for candidate in build_candidate_inventory(master, repository_root=tmp_path)[
            "candidates"
        ]
        if "12.0 m/s" in candidate["text"]
    )

    assert after["line_start"] != before["line_start"]
    assert after["candidate_id"] == before["candidate_id"]


@pytest.mark.unit
def test_candidate_inventory_resumes_after_labeled_display_math(
    tmp_path: Path,
) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text(
        "Narrative before the registered equation.\n\n"
        "$$\n"
        "M(q) \\ddot q = \\tau - b(q, \\dot q).\n"
        "$$ {#eq-motion}\n\n"
        "Narrative after the equation remains auditable [@study].\n",
        encoding="utf-8",
    )

    inventory = build_candidate_inventory(master, repository_root=tmp_path)

    assert inventory["candidate_count"] == 2
    assert [candidate["line_start"] for candidate in inventory["candidates"]] == [
        1,
        7,
    ]
    assert inventory["candidates"][1]["citation_keys"] == ["study"]


@pytest.mark.unit
def test_source_digest_is_invariant_to_checkout_line_endings(tmp_path: Path) -> None:
    master = tmp_path / "paper.qmd"
    source = "First auditable paragraph.\n\nSecond auditable paragraph.\n"
    master.write_bytes(source.encode("utf-8"))
    lf_digest = build_candidate_inventory(master, repository_root=tmp_path)[
        "source_digest"
    ]

    master.write_bytes(source.replace("\n", "\r\n").encode("utf-8"))
    crlf_digest = build_candidate_inventory(master, repository_root=tmp_path)[
        "source_digest"
    ]

    assert crlf_digest == lf_digest


@pytest.mark.unit
def test_registry_rejects_duplicate_claim_identifiers(tmp_path: Path) -> None:
    registry = {
        "schema_version": "proximal-distal-claim-audit-v1",
        "paper": {"source": "paper.qmd", "source_digest": "a" * 64},
        "audit_scope": {"completion_status": "in_progress"},
        "dependencies": [],
        "research_collections": [],
        "release_claim_inventory": [],
        "candidate_reviews": [],
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
def test_registry_rejects_missing_local_evidence_artifact(tmp_path: Path) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text("", encoding="utf-8")
    inventory = build_candidate_inventory(master, repository_root=tmp_path)
    registry = {
        "schema_version": "proximal-distal-claim-audit-v1",
        "paper": {
            "source": "paper.qmd",
            "source_digest": inventory["source_digest"],
        },
        "audit_scope": {"completion_status": "in_progress"},
        "release_claim_inventory": [],
        "candidate_reviews": [],
        "claims": [
            {
                "claim_id": "PD-CLAIM-001",
                "statement": "A test claim.",
                "classification": "test",
                "published_status": "untested",
                "audit_status": "provisional",
                "source_locations": ["paper.qmd:1"],
                "evidence_artifacts": ["evidence/missing.json"],
                "model_domain": "Test domain.",
                "uncertainty_boundary": "Test uncertainty.",
                "competing_explanations": ["Alternative"],
                "negative_controls": ["Negative control"],
                "falsifier": "Evidence is absent.",
                "adjudication": "Test adjudication.",
                "reviewer": "Test reviewer",
                "last_verified_on": "2026-08-13",
            }
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="missing local evidence artifact"):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
def test_registry_requires_reciprocal_candidate_claim_mapping(tmp_path: Path) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text("The model produces 12 m/s.\n", encoding="utf-8")
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    inventory = build_candidate_inventory(master, repository_root=tmp_path)
    candidate_id = inventory["candidates"][0]["candidate_id"]
    registry = {
        "schema_version": "proximal-distal-claim-audit-v1",
        "paper": {
            "source": "paper.qmd",
            "source_digest": inventory["source_digest"],
        },
        "audit_scope": {"completion_status": "in_progress"},
        "release_claim_inventory": [],
        "candidate_reviews": [],
        "claims": [
            {
                "claim_id": "PD-CLAIM-001",
                "candidate_ids": [candidate_id],
                "statement": "The declared model produces 12 m/s.",
                "classification": "model_result",
                "published_status": "supported",
                "audit_status": "provisional",
                "source_locations": ["paper.qmd:1"],
                "evidence_artifacts": ["result.json"],
                "model_domain": "Declared model.",
                "uncertainty_boundary": "No population inference.",
                "competing_explanations": ["Implementation error"],
                "negative_controls": ["Zero input"],
                "falsifier": "Recomputation disagrees.",
                "adjudication": "Independent review remains open.",
                "reviewer": "Test reviewer",
                "last_verified_on": "2026-08-12",
            }
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks a reciprocal candidate review"):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
def test_complete_registry_rejects_candidates_that_still_require_split(
    tmp_path: Path,
) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text("This paragraph may contain two claims.\n", encoding="utf-8")
    inventory = build_candidate_inventory(master, repository_root=tmp_path)
    candidate_id = inventory["candidates"][0]["candidate_id"]
    registry = {
        "schema_version": "proximal-distal-claim-audit-v1",
        "paper": {
            "source": "paper.qmd",
            "source_digest": inventory["source_digest"],
        },
        "audit_scope": {"completion_status": "complete"},
        "release_claim_inventory": [],
        "candidate_reviews": [
            {
                "candidate_id": candidate_id,
                "disposition": "requires_split",
                "claim_ids": [],
                "rationale": "Atomic coverage is unfinished.",
                "reviewer": "Test reviewer",
                "last_verified_on": "2026-08-12",
            }
        ],
        "claims": [],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="require splitting"):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
def test_repository_registry_matches_release_claims_and_is_complete() -> None:
    root = Path(__file__).resolve().parents[3]
    registry_path = root / (
        "docs/research/proximal_distal_energy_transfer/data/claim_audit_registry.json"
    )

    result = validate_registry(registry_path, repository_root=root)

    assert result["release_claim_count"] >= 18
    assert result["registered_claim_count"] >= 5
    assert result["reviewed_candidate_count"] >= 5
    assert result["completion_status"] == "complete"
    assert result["unadjudicated_candidate_count"] == 0
