from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.grf_source_reconciliation import (
    REQUIRED_TOPIC_IDS,
    load_and_validate_registry,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.scientific
REGISTRY = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/"
    "grf_source_reconciliation.json"
)


def _record() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _topic(record: dict[str, object], topic_id: str) -> dict[str, object]:
    return next(row for row in record["topics"] if row["topic_id"] == topic_id)


def test_committed_register_reconciles_every_required_topic() -> None:
    coverage = load_and_validate_registry(REGISTRY, repository_root=ROOT)

    assert set(coverage["topic_ids"]) == set(REQUIRED_TOPIC_IDS)
    assert coverage["unsupported_topic_ids"] == []
    assert coverage["open_gap_count"] >= 1
    assert coverage["recorded_disagreement_count"] >= 1


def test_corpus_is_recorded_as_blocked_without_credentials() -> None:
    record = _record()
    corpus = record["corpus_access"]

    assert corpus["status"] == "blocked_on_manual_authentication"
    assert corpus["notebook_ids"] == []
    assert corpus["source_inventory_ids"] == []
    assert record["evidence_policy"]["generated_prose_is_evidence"] is False
    assert record["coverage"]["corpus_derived_evidence_count"] == 0


def test_blocked_corpus_cannot_smuggle_notebook_identifiers() -> None:
    record = _record()
    record["corpus_access"]["notebook_ids"] = ["notebook-fabricated"]

    with pytest.raises(ValueError, match="blocked corpus cannot record"):
        validate_registry(record, repository_root=ROOT)


def test_generated_prose_can_never_be_declared_evidence() -> None:
    record = _record()
    record["evidence_policy"]["generated_prose_is_evidence"] = True

    with pytest.raises(ValueError, match="generated prose"):
        validate_registry(record, repository_root=ROOT)


def test_unreviewed_work_cannot_enter_the_reconciliation() -> None:
    record = _record()
    record["works"][0]["work_id"] = "doi:10.0000/not-reviewed"

    with pytest.raises(ValueError, match="absent from the external source review"):
        validate_registry(record, repository_root=ROOT)


def test_bib_key_must_exist_in_the_article_bibliography() -> None:
    record = _record()
    record["works"][0]["bib_key"] = "absent2026"

    with pytest.raises(ValueError, match="absent from references.bib"):
        validate_registry(record, repository_root=ROOT)


def test_topic_claim_requires_a_supporting_work_in_that_topic() -> None:
    record = _record()
    topic = _topic(record, "free_moment")
    topic["claim_ids"] = sorted([*topic["claim_ids"], "PD-CLAIM-078"])

    with pytest.raises(ValueError, match="no registered work supports"):
        validate_registry(record, repository_root=ROOT)


def test_model_internal_claim_cannot_be_given_external_support() -> None:
    record = _record()
    topic = _topic(record, "contact_model")
    topic["model_internal_claim_ids"] = sorted(
        [*topic["model_internal_claim_ids"], "PD-CLAIM-074"]
    )

    with pytest.raises(ValueError, match="carry external support"):
        validate_registry(record, repository_root=ROOT)


def test_open_topic_must_record_at_least_one_gap() -> None:
    record = _record()
    _topic(record, "segment_power")["gaps"] = []

    with pytest.raises(ValueError, match="must record at least one gap"):
        validate_registry(record, repository_root=ROOT)


def test_coverage_summary_must_reproduce_from_the_topics() -> None:
    record = _record()
    record["coverage"]["open_gap_count"] += 1

    with pytest.raises(ValueError, match="coverage summary does not reproduce"):
        validate_registry(record, repository_root=ROOT)


def test_required_topic_cannot_be_dropped() -> None:
    record = _record()
    record["topics"] = [
        row for row in record["topics"] if row["topic_id"] != "free_moment"
    ]
    record["coverage"]["topic_count"] -= 1
    record["coverage"]["topic_ids"].remove("free_moment")
    record["coverage"]["externally_bounded_topic_ids"].remove("free_moment")
    record["coverage"]["open_gap_count"] -= 2
    record["coverage"]["recorded_disagreement_count"] -= 1

    with pytest.raises(ValueError, match="missing required topics"):
        validate_registry(record, repository_root=ROOT)
