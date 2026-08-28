"""Tests for the proximal-to-distal paper claim-audit authority."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.research.proximal_distal_energy.claim_audit import (
    build_candidate_inventory,
    validate_registry,
    write_candidate_inventory,
)
from scripts.research.proximal_distal_energy.migrate_claim_adjudication_v2 import (
    PRE_ADJUDICATION_SOURCE_DIGEST,
    PRIOR_REVIEWED_SOURCE_DIGEST,
    PRIOR_REVIEWER_PROJECTION_CANDIDATE_IDS,
    REVIEWER_PROJECTION_CANDIDATE_IDS,
    migrate,
)


def _copy_reviewed_snapshot(root: Path, target_root: Path) -> Path:
    relative = Path("docs/research/proximal_distal_energy_transfer")
    source_article = root / relative
    target_article = target_root / relative
    target_data = target_article / "data"
    target_data.mkdir(parents=True)
    for name in ("claim_audit_registry.json", "claim_candidate_inventory.json"):
        shutil.copy2(source_article / "data" / name, target_data / name)
    shutil.copy2(source_article / "proximal_distal_energy_transfer.qmd", target_article)
    target_chapters = target_article / "chapters"
    target_chapters.mkdir()
    for source in (source_article / "chapters").glob("*.qmd"):
        shutil.copy2(source, target_chapters / source.name)
    return target_data


@pytest.mark.unit
def test_candidate_inventory_writer_preserves_scientific_unicode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "inventory.json"

    write_candidate_inventory(output, {"statement": "P→D — 30°"})

    rendered = output.read_text(encoding="utf-8")
    assert '"statement": "P→D — 30°"' in rendered
    assert "\\u2192" not in rendered


def _minimal_registry(tmp_path: Path, source_location: str) -> Path:
    master = tmp_path / "paper.qmd"
    master.write_text("The registered result remains conditional.\n", encoding="utf-8")
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    inventory = build_candidate_inventory(master, repository_root=tmp_path)
    registry = {
        "schema_version": "proximal-distal-claim-audit-v2",
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
                "statement": "The registered result remains conditional.",
                "classification": "model_result",
                "published_status": "conditional",
                "audit_status": "provisional",
                "adjudication_outcome": "supported",
                "source_locations": [source_location],
                "evidence_artifacts": ["result.json"],
                "model_domain": "Declared test model.",
                "uncertainty_boundary": "No population inference.",
                "competing_explanations": ["Implementation error"],
                "negative_controls": ["Zero input"],
                "falsifier": "Independent recomputation disagrees.",
                "adjudication": "The claim remains conditional.",
                "reviewer": "Test reviewer",
                "last_verified_on": "2026-08-14",
            }
        ],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


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
        "schema_version": "proximal-distal-claim-audit-v2",
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
    master.write_text("Evidence source line.\n", encoding="utf-8")
    inventory = build_candidate_inventory(master, repository_root=tmp_path)
    registry = {
        "schema_version": "proximal-distal-claim-audit-v2",
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
                "adjudication_outcome": "untested",
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
@pytest.mark.parametrize("outcome", [None, "provisional", "SUPPORTED"])
def test_registry_requires_normalized_adjudication_outcome(
    tmp_path: Path, outcome: str | None
) -> None:
    path = _minimal_registry(tmp_path, "paper.qmd:1")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if outcome is None:
        registry["claims"][0].pop("adjudication_outcome")
    else:
        registry["claims"][0]["adjudication_outcome"] = outcome
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="adjudication_outcome"):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
def test_supported_outcome_requires_narrow_scope_when_validation_is_open(
    tmp_path: Path,
) -> None:
    path = _minimal_registry(tmp_path, "paper.qmd:1")
    registry = json.loads(path.read_text(encoding="utf-8"))
    claim = registry["claims"][0]
    claim["audit_status"] = "human_validation_blocked"
    claim["published_status"] = "supported"
    claim["adjudication"] = "Evidence was checked."
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly narrower scope"):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("artifact", "fixture_name", "fixture_text", "expected_type"),
    [
        (
            "references.bib#source2026",
            "references.bib",
            "@article{source2026,}\n",
            "bibliography_key",
        ),
        ("chapter.qmd#eq-power", "chapter.qmd", "$$ {#eq-power}\n", "local_anchor"),
        ("result.json", "result.json", "{}\n", "generated_artifact"),
        ("https://doi.org/10.1000/example", None, None, "doi"),
        ("https://example.org/paper", None, None, "external_url"),
    ],
)
def test_registry_types_and_validates_evidence_locators(
    tmp_path: Path,
    artifact: str,
    fixture_name: str | None,
    fixture_text: str | None,
    expected_type: str,
) -> None:
    path = _minimal_registry(tmp_path, "paper.qmd:1")
    if fixture_name is not None and fixture_text is not None:
        (tmp_path / fixture_name).write_text(fixture_text, encoding="utf-8")
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["claims"][0]["evidence_artifacts"] = [artifact]
    path.write_text(json.dumps(registry), encoding="utf-8")

    result = validate_registry(
        path, repository_root=tmp_path, check_release_manifest=False
    )

    assert result["evidence_locator_type_counts"] == {expected_type: 1}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("artifact", "fixture_name", "fixture_text", "message"),
    [
        (
            "references.bib#missing",
            "references.bib",
            "@article{present,}\n",
            "missing bibliography key",
        ),
        (
            "chapter.qmd#missing",
            "chapter.qmd",
            "# Chapter\n",
            "missing local anchor",
        ),
    ],
)
def test_registry_rejects_broken_evidence_fragments(
    tmp_path: Path,
    artifact: str,
    fixture_name: str,
    fixture_text: str,
    message: str,
) -> None:
    path = _minimal_registry(tmp_path, "paper.qmd:1")
    (tmp_path / fixture_name).write_text(fixture_text, encoding="utf-8")
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["claims"][0]["evidence_artifacts"] = [artifact]
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_location", "message"),
    [
        ("paper.qmd", "path:line"),
        ("missing.qmd:1", "missing source location file"),
        ("paper.qmd:2", "source location line is out of range"),
        ("../outside.qmd:1", "source location escapes repository root"),
    ],
)
def test_registry_rejects_unresolvable_source_locations(
    tmp_path: Path, source_location: str, message: str
) -> None:
    path = _minimal_registry(tmp_path, source_location)

    with pytest.raises(ValueError, match=message):
        validate_registry(path, repository_root=tmp_path, check_release_manifest=False)


@pytest.mark.unit
def test_registry_requires_reciprocal_candidate_claim_mapping(tmp_path: Path) -> None:
    master = tmp_path / "paper.qmd"
    master.write_text("The model produces 12 m/s.\n", encoding="utf-8")
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    inventory = build_candidate_inventory(master, repository_root=tmp_path)
    candidate_id = inventory["candidates"][0]["candidate_id"]
    registry = {
        "schema_version": "proximal-distal-claim-audit-v2",
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
                "adjudication_outcome": "supported",
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
        "schema_version": "proximal-distal-claim-audit-v2",
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
    assert result["release_review_completion_status"] == "complete"
    assert result["open_release_claim_count"] == 0
    assert result["open_release_claim_keys"] == []
    assert result["adjudication_outcome_counts"] == {
        "inconclusive": 5,
        "supported": 308,
        "untested": 15,
    }
    assert result["evidence_locator_type_counts"]["bibliography_key"] >= 3
    assert result["evidence_locator_type_counts"]["local_anchor"] >= 1


@pytest.mark.unit
def test_v2_migration_is_snapshot_locked_and_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    target_data = _copy_reviewed_snapshot(root, tmp_path)

    assert migrate(tmp_path) == {
        "supported": 308,
        "contradicted": 0,
        "inconclusive": 5,
        "untested": 15,
    }

    registry_path = target_data / "claim_audit_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["paper"]["source_digest"] = "0" * 64
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly reviewed v2 snapshot"):
        migrate(tmp_path)


@pytest.mark.unit
def test_v2_migration_reconciles_only_explicit_reviewer_projection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    target_data = _copy_reviewed_snapshot(root, tmp_path)
    registry_path = target_data / "claim_audit_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["paper"]["source_digest"] = PRE_ADJUDICATION_SOURCE_DIGEST
    registry["candidate_reviews"] = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in REVIEWER_PROJECTION_CANDIDATE_IDS
    ]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    migrate(tmp_path)

    migrated = json.loads(registry_path.read_text(encoding="utf-8"))
    migrated_reviews = {
        review["candidate_id"]: review for review in migrated["candidate_reviews"]
    }
    assert set(migrated_reviews) >= REVIEWER_PROJECTION_CANDIDATE_IDS
    assert all(
        migrated_reviews[candidate_id]["disposition"] == "editorial_or_navigation"
        for candidate_id in REVIEWER_PROJECTION_CANDIDATE_IDS
    )


@pytest.mark.unit
def test_v2_migration_reconciles_prior_reviewed_projection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    target_data = _copy_reviewed_snapshot(root, tmp_path)
    registry_path = target_data / "claim_audit_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["paper"]["source_digest"] = PRIOR_REVIEWED_SOURCE_DIGEST
    retained = [
        review
        for review in registry["candidate_reviews"]
        if review["candidate_id"] not in REVIEWER_PROJECTION_CANDIDATE_IDS
    ]
    retained.extend(
        {
            "candidate_id": candidate_id,
            "disposition": "editorial_or_navigation",
            "claim_ids": [],
            "rationale": "Prior reviewed projection.",
            "reviewer": "Codex technical audit",
            "last_verified_on": "2026-08-23",
        }
        for candidate_id in PRIOR_REVIEWER_PROJECTION_CANDIDATE_IDS
    )
    registry["candidate_reviews"] = retained
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    migrate(tmp_path)

    migrated = json.loads(registry_path.read_text(encoding="utf-8"))
    migrated_ids = {review["candidate_id"] for review in migrated["candidate_reviews"]}
    assert migrated_ids >= REVIEWER_PROJECTION_CANDIDATE_IDS
    obsolete_ids = (
        PRIOR_REVIEWER_PROJECTION_CANDIDATE_IDS - REVIEWER_PROJECTION_CANDIDATE_IDS
    )
    assert not (obsolete_ids & migrated_ids)


@pytest.mark.unit
def test_v2_migration_rejects_claim_without_explicit_review(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    target_data = _copy_reviewed_snapshot(root, tmp_path)

    registry_path = target_data / "claim_audit_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["claims"][0]["claim_id"] = "PD-CLAIM-999"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="without explicit reviewed outcomes"):
        migrate(tmp_path)
