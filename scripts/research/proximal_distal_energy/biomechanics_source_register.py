"""Validate the source-bounded biomechanics research register."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
SOURCE_REGISTER_REL = ARTICLE_REL / "data/biomechanics_source_register.json"
CLAIM_REGISTRY_REL = ARTICLE_REL / "data/claim_audit_registry.json"
EXTERNAL_REVIEW_REL = ARTICLE_REL / "data/external_source_review.json"
BIBLIOGRAPHY_REL = ARTICLE_REL / "references.bib"
SCHEMA_VERSION = "proximal-distal-biomechanics-source-register/v1"

REQUIRED_DOMAINS = {
    "pelvis",
    "thorax",
    "scapula",
    "glenohumeral",
    "elbow",
    "forearm",
    "wrist",
    "hand",
    "bilateral_grip",
    "ground_pathway",
    "muscle_tendon_dynamics",
    "fatigue",
    "injury",
    "population_variability",
    "equipment_variability",
    "task_variability",
}
SOURCE_ROLES = {
    "primary_human_experiment",
    "primary_human_observation",
    "primary_nonhuman_experiment",
    "primary_model_or_methods",
    "review_or_synthesis",
    "systematic_review",
}
EVIDENCE_USES = {"bounded_support", "contextual_only"}
CORRECTION_STATES = {
    "no_notice_found_in_checked_metadata",
    "not_applicable_non_periodical",
}
COVERAGE_STATES = {"bounded", "context_only", "gap"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_text(record: dict[str, Any], field: str, record_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_id}: {field} must be non-empty text")
    return value


def _require_text_list(
    record: dict[str, Any], field: str, record_id: str, *, allow_empty: bool = False
) -> list[str]:
    value = record.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{record_id}: {field} must be a text list")
    if not allow_empty and not value:
        raise ValueError(f"{record_id}: {field} must not be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{record_id}: {field} contains duplicates")
    return value


def _normalise(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _bibliography_entries(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^@[^{]+\{([^,]+),", text))
    entries: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries[match.group(1).strip()] = text[match.start() : end]
    return entries


def _validate_source(
    source: dict[str, Any],
    bibliography: dict[str, str],
    claim_ids: set[str],
) -> tuple[str, set[str]]:
    source_id = _require_text(source, "source_id", "source")
    expected_fields = {
        "source_id",
        "bibliography_key",
        "title",
        "year",
        "stable_url",
        "identifier",
        "source_role",
        "independence",
        "correction_status",
        "correction_check_method",
        "population",
        "apparatus",
        "estimand",
        "uncertainty",
        "limitations",
        "linked_claim_ids",
        "coverage_domains",
        "evidence_use",
        "raw_data_status",
    }
    unknown_fields = sorted(set(source) - expected_fields)
    if unknown_fields:
        raise ValueError(f"{source_id}: unknown fields: {unknown_fields}")
    for field in (
        "bibliography_key",
        "title",
        "stable_url",
        "identifier",
        "correction_check_method",
        "population",
        "apparatus",
        "estimand",
        "uncertainty",
        "raw_data_status",
    ):
        _require_text(source, field, source_id)
    if not isinstance(source.get("year"), int):
        raise ValueError(f"{source_id}: year must be an integer")
    if source.get("source_role") not in SOURCE_ROLES:
        raise ValueError(f"{source_id}: invalid source_role")
    if source.get("independence") != "independent_of_project":
        raise ValueError(f"{source_id}: independence must be independent_of_project")
    if source.get("correction_status") not in CORRECTION_STATES:
        raise ValueError(f"{source_id}: invalid correction_status")
    if source.get("evidence_use") not in EVIDENCE_USES:
        raise ValueError(f"{source_id}: invalid evidence_use")
    limitations = _require_text_list(source, "limitations", source_id)
    if len(limitations) < 2:
        raise ValueError(f"{source_id}: at least two limitations are required")
    linked_claims = set(_require_text_list(source, "linked_claim_ids", source_id))
    unknown_claims = sorted(linked_claims - claim_ids)
    if unknown_claims:
        raise ValueError(f"{source_id}: unknown claim ids: {unknown_claims}")
    domains = set(_require_text_list(source, "coverage_domains", source_id))
    unknown_domains = sorted(domains - REQUIRED_DOMAINS)
    if unknown_domains:
        raise ValueError(f"{source_id}: unknown coverage domains: {unknown_domains}")
    key = source["bibliography_key"]
    if key not in bibliography:
        raise ValueError(f"{source_id}: bibliography key is missing: {key}")
    entry = _normalise(bibliography[key])
    if _normalise(source["identifier"].split(":", 1)[-1]) not in entry:
        raise ValueError(f"{source_id}: bibliography entry does not contain identifier")
    if _normalise(source["title"]) not in entry:
        raise ValueError(f"{source_id}: bibliography entry does not contain title")
    return source_id, domains


def validate_biomechanics_source_register(
    root: str | Path, register: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed on source, coverage, claim, bibliography, or digest drift."""
    root_path = Path(root).resolve()
    bibliography_path = root_path / BIBLIOGRAPHY_REL
    claim_path = root_path / CLAIM_REGISTRY_REL
    external_path = root_path / EXTERNAL_REVIEW_REL
    if register.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("biomechanics source register schema is stale")
    expected_root_fields = {
        "schema_version",
        "scope_statement",
        "reviewed_on",
        "reviewer",
        "bibliography_sha256",
        "claim_registry_sha256",
        "external_source_review_sha256",
        "sources",
        "coverage",
        "summary",
    }
    unknown_fields = sorted(set(register) - expected_root_fields)
    if unknown_fields:
        raise ValueError(f"source register contains unknown fields: {unknown_fields}")
    for field in ("scope_statement", "reviewed_on", "reviewer"):
        _require_text(register, field, "register")
    expected_digests = {
        "bibliography_sha256": _sha256(bibliography_path),
        "claim_registry_sha256": _sha256(claim_path),
        "external_source_review_sha256": _sha256(external_path),
    }
    for field, digest in expected_digests.items():
        if register.get(field) != digest:
            raise ValueError(f"{field} is stale")

    claim_registry = json.loads(claim_path.read_text(encoding="utf-8"))
    claim_ids = {record["claim_id"] for record in claim_registry["claims"]}
    bibliography = _bibliography_entries(bibliography_path.read_text(encoding="utf-8"))
    sources = register.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    source_domains: dict[str, set[str]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        source_id, domains = _validate_source(source, bibliography, claim_ids)
        if source_id in source_domains:
            raise ValueError(f"duplicate source_id: {source_id}")
        source_domains[source_id] = domains

    coverage = register.get("coverage")
    if not isinstance(coverage, list):
        raise ValueError("coverage must be a list")
    coverage_states: dict[str, str] = {}
    for record in coverage:
        if not isinstance(record, dict):
            raise ValueError("each coverage record must be an object")
        domain_id = _require_text(record, "domain_id", "coverage")
        if set(record) != {
            "domain_id",
            "source_ids",
            "evidence_status",
            "current_answer",
            "limitations",
            "required_observables",
            "data_gate",
        }:
            raise ValueError(f"{domain_id}: coverage fields are incomplete or unknown")
        if domain_id in coverage_states:
            raise ValueError(f"duplicate coverage domain: {domain_id}")
        state = record.get("evidence_status")
        if state not in COVERAGE_STATES:
            raise ValueError(f"{domain_id}: invalid evidence_status")
        coverage_states[domain_id] = state
        source_ids = set(
            _require_text_list(
                record, "source_ids", domain_id, allow_empty=state == "gap"
            )
        )
        unknown_sources = sorted(source_ids - set(source_domains))
        if unknown_sources:
            raise ValueError(f"{domain_id}: unknown source_ids: {unknown_sources}")
        incoherent = sorted(
            source_id
            for source_id in source_ids
            if domain_id not in source_domains[source_id]
        )
        if incoherent:
            raise ValueError(
                f"{domain_id}: source coverage is not reciprocal: {incoherent}"
            )
        for field in ("current_answer", "data_gate"):
            _require_text(record, field, domain_id)
        _require_text_list(record, "limitations", domain_id)
        _require_text_list(record, "required_observables", domain_id)
    if set(coverage_states) != REQUIRED_DOMAINS:
        raise ValueError("coverage domains are incomplete or stale")

    state_counts = Counter(coverage_states.values())
    expected_summary = {
        "source_count": len(sources),
        "independent_source_count": len(sources),
        "coverage_domain_count": len(coverage),
        "source_bounded_domain_count": state_counts["bounded"],
        "context_only_domain_count": state_counts["context_only"],
        "source_gap_domain_count": state_counts["gap"],
    }
    if register.get("summary") != expected_summary:
        raise ValueError("biomechanics source register summary is stale")
    return {"valid": True, **expected_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate",))
    parser.parse_args()
    root = _repository_root()
    register = json.loads((root / SOURCE_REGISTER_REL).read_text(encoding="utf-8"))
    print(json.dumps(validate_biomechanics_source_register(root, register), indent=2))


if __name__ == "__main__":
    main()
