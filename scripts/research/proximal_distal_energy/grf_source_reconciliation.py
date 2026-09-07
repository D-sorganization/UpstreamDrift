"""Fail-closed reconciliation of ground-reaction topics with original sources.

Issue #8496 asked for the ground-reaction bibliography of #8493 to be
reconciled against a NotebookLM biomechanics corpus. The corpus is a research
index, never evidence, so this register maps each topic to works that were
already adjudicated in ``external_source_review.json`` and verified against
their own DOI, PubMed/PMC, or publisher record. It fails closed when a mapped
work is absent or ineligible there, when a claim said to be model-internal
actually carries external support, or when the committed coverage summary does
not reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .external_source_review import REVIEW_REL

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
REGISTRY_REL = ARTICLE_REL / "data/grf_source_reconciliation.json"
CLAIM_REGISTRY_REL = ARTICLE_REL / "data/claim_audit_registry.json"
REFERENCES_REL = ARTICLE_REL / "references.bib"

SCHEMA_VERSION = "grf-source-reconciliation/v1"
REGISTRY_ID = "grf-source-reconciliation-v1"

REQUIRED_TOPIC_IDS = frozenset(
    {
        "center_of_pressure",
        "contact_model",
        "free_moment",
        "ground_reaction_force",
        "segment_power",
    }
)
_TOP_KEYS = {
    "schema_version",
    "registry_id",
    "registered_at_utc",
    "authority",
    "scope_statement",
    "corpus_access",
    "evidence_policy",
    "works",
    "topics",
    "coverage",
    "inference_boundary",
}
_CORPUS_KEYS = {
    "provider",
    "collection",
    "status",
    "checked_on",
    "method",
    "observed",
    "notebook_ids",
    "source_inventory_ids",
    "credential_handling",
    "unblock_action",
    "consequence",
}
_WORK_KEYS = {
    "work_id",
    "bib_key",
    "verified_against",
    "verification_method",
    "checked_on",
    "contribution",
    "boundary",
}
_TOPIC_KEYS = {
    "topic_id",
    "question",
    "article_locations",
    "claim_ids",
    "model_internal_claim_ids",
    "work_ids",
    "resolution",
    "gaps",
    "disagreements",
}
_CORPUS_STATES = {"blocked_on_manual_authentication", "enumerated"}
_ORIGINAL_SOURCE_KINDS = {
    "dataset_record",
    "doi",
    "publisher_record",
    "pubmed_or_pmc",
    "repository_record",
}
_RESOLUTIONS = {
    "externally_bounded",
    "externally_bounded_with_open_gap",
    "unsupported_open_gap",
}
_CREDENTIAL_MARKERS = ("cookie", "token", "password", "secret", "session_id")


def _require_exact_keys(record: dict[str, Any], expected: set[str], name: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{name} fields do not match the registered schema")


def _require_text(record: dict[str, Any], field: str, name: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: {field} must be non-empty text")
    return value


def _require_sorted_unique(values: object, name: str) -> list[str]:
    if not isinstance(values, list) or not all(
        isinstance(item, str) and item.strip() for item in values
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be sorted and free of duplicates")
    return values


def _validate_corpus_access(corpus: object) -> bool:
    if not isinstance(corpus, dict):
        raise ValueError("corpus_access must be an object")
    _require_exact_keys(corpus, _CORPUS_KEYS, "corpus_access")
    for field in ("provider", "collection", "checked_on", "method"):
        _require_text(corpus, field, "corpus_access")
    for field in ("credential_handling", "unblock_action", "consequence"):
        _require_text(corpus, field, "corpus_access")
    if corpus["status"] not in _CORPUS_STATES:
        raise ValueError("corpus_access status is not a registered state")
    if not isinstance(corpus["observed"], dict) or not corpus["observed"]:
        raise ValueError("corpus_access requires the observed tool status")
    for key, value in corpus["observed"].items():
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"corpus_access observed.{key} must be a boolean or null")
    notebooks = _require_sorted_unique(corpus["notebook_ids"], "notebook_ids")
    inventories = _require_sorted_unique(
        corpus["source_inventory_ids"], "source_inventory_ids"
    )
    enumerated = corpus["status"] == "enumerated"
    if enumerated and not notebooks:
        raise ValueError("an enumerated corpus must record notebook identifiers")
    if not enumerated and (notebooks or inventories):
        raise ValueError(
            "a blocked corpus cannot record notebook or source identifiers"
        )
    lowered = json.dumps(corpus).lower()
    for marker in _CREDENTIAL_MARKERS:
        if f'"{marker}"' in lowered:
            raise ValueError(f"corpus_access must not carry a {marker} field")
    return enumerated


def _validate_evidence_policy(policy: object) -> None:
    if not isinstance(policy, dict):
        raise ValueError("evidence_policy must be an object")
    if policy.get("corpus_authority") != "research_index_only":
        raise ValueError("the corpus must stay a research index only")
    if policy.get("generated_prose_is_evidence") is not False:
        raise ValueError("generated prose can never be registered as evidence")
    for field in ("verification_rule", "gap_rule"):
        _require_text(policy, field, "evidence_policy")


def _validate_work(work: dict[str, Any], review_index: dict[str, str], bib: str) -> str:
    _require_exact_keys(work, _WORK_KEYS, "work")
    work_id = _require_text(work, "work_id", "work")
    for field in ("verification_method", "checked_on", "contribution", "boundary"):
        _require_text(work, field, work_id)
    if work["verified_against"] not in _ORIGINAL_SOURCE_KINDS:
        raise ValueError(f"{work_id}: verified_against must name an original source")
    if work_id not in review_index:
        raise ValueError(f"{work_id}: absent from the external source review")
    if review_index[work_id] != "eligible":
        raise ValueError(f"{work_id}: mapped work is not evidence-eligible")
    bib_key = _require_text(work, "bib_key", work_id)
    if f"{{{bib_key}," not in bib:
        raise ValueError(f"{work_id}: bib_key {bib_key} is absent from references.bib")
    return work_id


def _validate_topic(
    topic: dict[str, Any],
    work_ids: set[str],
    claim_ids: set[str],
    supported_by_work: dict[str, set[str]],
    externally_supported: set[str],
) -> None:
    _require_exact_keys(topic, _TOPIC_KEYS, "topic")
    topic_id = _require_text(topic, "topic_id", "topic")
    _require_text(topic, "question", topic_id)
    if not topic["article_locations"]:
        raise ValueError(f"{topic_id}: article_locations must not be empty")
    _require_sorted_unique(topic["article_locations"], f"{topic_id} article_locations")
    if topic["resolution"] not in _RESOLUTIONS:
        raise ValueError(f"{topic_id}: resolution is not registered")
    topic_works = _require_sorted_unique(topic["work_ids"], f"{topic_id} work_ids")
    if not set(topic_works) <= work_ids:
        raise ValueError(f"{topic_id}: work_ids are not registered works")
    topic_claims = _require_sorted_unique(topic["claim_ids"], f"{topic_id} claim_ids")
    internal = _require_sorted_unique(
        topic["model_internal_claim_ids"], f"{topic_id} model_internal_claim_ids"
    )
    if set(topic_claims) & set(internal):
        raise ValueError(f"{topic_id}: a claim cannot be external and model-internal")
    if not set(topic_claims) | set(internal) <= claim_ids:
        raise ValueError(f"{topic_id}: claim ids are absent from the claim registry")
    supported = {claim for wid in topic_works for claim in supported_by_work[wid]}
    unbacked = sorted(set(topic_claims) - supported)
    if unbacked:
        raise ValueError(f"{topic_id}: no registered work supports {unbacked}")
    overreach = sorted(set(internal) & externally_supported)
    if overreach:
        raise ValueError(f"{topic_id}: {overreach} carry external support")
    if topic["resolution"] == "unsupported_open_gap" and topic_works:
        raise ValueError(f"{topic_id}: an unsupported topic cannot list works")
    if not topic_works and topic["resolution"] != "unsupported_open_gap":
        raise ValueError(f"{topic_id}: a topic without works must be unsupported")
    for field in ("gaps", "disagreements"):
        if not isinstance(topic[field], list) or not all(
            isinstance(item, str) and item.strip() for item in topic[field]
        ):
            raise ValueError(f"{topic_id}: {field} must be a list of statements")
    if topic["resolution"] == "externally_bounded" and topic["gaps"]:
        raise ValueError(f"{topic_id}: a fully bounded topic cannot record gaps")
    if topic["resolution"] != "externally_bounded" and not topic["gaps"]:
        raise ValueError(f"{topic_id}: an open topic must record at least one gap")


def _load_review(root: Path) -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    review = json.loads((root / REVIEW_REL).read_text(encoding="utf-8"))
    dispositions = {
        work["work_id"]: work["evidence_disposition"] for work in review["works"]
    }
    supported = {
        work["work_id"]: set(work["supports_claims"]) for work in review["works"]
    }
    return dispositions, supported, {c for ids in supported.values() for c in ids}


def compute_coverage(record: dict[str, Any], corpus_enumerated: bool) -> dict[str, Any]:
    """Recompute the coverage summary from the registered topics and works."""

    topics = record["topics"]
    topic_ids = [topic["topic_id"] for topic in topics]
    return {
        "topic_count": len(topics),
        "work_count": len(record["works"]),
        "topic_ids": topic_ids,
        "externally_bounded_topic_ids": [
            topic["topic_id"]
            for topic in topics
            if topic["resolution"] != "unsupported_open_gap"
        ],
        "unsupported_topic_ids": [
            topic["topic_id"]
            for topic in topics
            if topic["resolution"] == "unsupported_open_gap"
        ],
        "open_gap_count": sum(len(topic["gaps"]) for topic in topics),
        "recorded_disagreement_count": sum(
            len(topic["disagreements"]) for topic in topics
        ),
        "corpus_enumerated": corpus_enumerated,
        "corpus_derived_evidence_count": 0,
    }


def validate_registry(
    record: dict[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    """Validate one reconciliation register and return its recomputed coverage.

    Raises ``ValueError`` unless every mapped work is an eligible entry of the
    external source review, every mapped claim is supported by one of the
    topic's own works, no model-internal claim carries external support, and
    the committed coverage summary reproduces exactly.
    """

    if not isinstance(record, dict):
        raise ValueError("registry must be an object")
    _require_exact_keys(record, _TOP_KEYS, "registry")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("registry schema_version is unsupported")
    if record["registry_id"] != REGISTRY_ID:
        raise ValueError("registry_id is not registered")
    for field in ("registered_at_utc", "authority", "scope_statement"):
        _require_text(record, field, "registry")
    boundary = _require_text(record, "inference_boundary", "registry")
    if "cannot" not in boundary.lower():
        raise ValueError("inference_boundary must state what this register cannot show")
    corpus_enumerated = _validate_corpus_access(record["corpus_access"])
    _validate_evidence_policy(record["evidence_policy"])

    root = Path(repository_root).resolve()
    dispositions, supported_by_work, externally_supported = _load_review(root)
    bib = (root / REFERENCES_REL).read_text(encoding="utf-8")
    claims = json.loads((root / CLAIM_REGISTRY_REL).read_text(encoding="utf-8"))
    claim_ids = {claim["claim_id"] for claim in claims["claims"]}

    works = record["works"]
    if not isinstance(works, list) or not works:
        raise ValueError("works must be a nonempty list")
    ids = [_validate_work(work, dispositions, bib) for work in works]
    _require_sorted_unique(ids, "work_id values")

    topics = record["topics"]
    if not isinstance(topics, list) or not topics:
        raise ValueError("topics must be a nonempty list")
    for topic in topics:
        _validate_topic(
            topic, set(ids), claim_ids, supported_by_work, externally_supported
        )
    topic_ids = _require_sorted_unique(
        [topic["topic_id"] for topic in topics], "topic_id values"
    )
    missing = sorted(REQUIRED_TOPIC_IDS - set(topic_ids))
    if missing:
        raise ValueError(f"the reconciliation is missing required topics: {missing}")

    coverage = compute_coverage(record, corpus_enumerated)
    if record["coverage"] != coverage:
        raise ValueError("committed coverage summary does not reproduce")
    return coverage


def load_and_validate_registry(
    path: Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Load the register without accepting duplicate JSON keys and validate it."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    record = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    return validate_registry(record, repository_root=repository_root)


def main() -> None:
    """Validate the committed ground-reaction source reconciliation."""
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Validate the GRF source register.")
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--registry", type=Path, default=root / REGISTRY_REL)
    args = parser.parse_args()
    coverage = load_and_validate_registry(args.registry, repository_root=root)
    print(json.dumps(coverage, sort_keys=True))


if __name__ == "__main__":
    main()
