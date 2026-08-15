"""Validate the governed review of external proximal-distal sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .claim_evidence_integrity import MANIFEST_REL

ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
REVIEW_REL = ARTICLE_REL / "data/external_source_review.json"
SCHEMA_VERSION = "proximal-distal-external-source-review-v1"

SOURCE_TYPES = {
    "book_chapter",
    "journal_article",
    "proceedings_article",
    "repository_record",
    "web_publication",
}
EVIDENCE_ROLES = {
    "empirical_human",
    "methods_or_measurement",
    "modeling_or_simulation",
    "review_or_synthesis",
    "theory_or_definition",
}
INDEPENDENCE_STATES = {
    "independent_of_project",
    "project_author_overlap",
    "unclear",
}
CORRECTION_STATES = {
    "correction_present",
    "expression_of_concern",
    "no_notice_found_in_reviewed_metadata",
    "not_applicable_non_periodical",
    "retracted",
}
DISPOSITIONS = {"contextual_only", "eligible", "excluded"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_text(record: dict[str, Any], field: str, work_id: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{work_id}: {field} must be non-empty text")


def validate_external_source_review(
    root: str | Path,
    review: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless every external URL has one complete adjudication."""
    root_path = Path(root).resolve()
    evidence_manifest = manifest or json.loads(
        (root_path / MANIFEST_REL).read_text(encoding="utf-8")
    )
    if review.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("external source review schema version is missing or stale")
    for field in ("reviewed_on", "reviewer", "scope_statement"):
        _require_text(review, field, "review")
    availability = review.get("availability_snapshot")
    if not isinstance(availability, dict):
        raise ValueError("external source review requires an availability snapshot")
    _require_text(availability, "checked_on", "availability_snapshot")
    _require_text(availability, "method", "availability_snapshot")
    url_checks = availability.get("url_checks")
    if not isinstance(url_checks, dict):
        raise ValueError("availability snapshot requires per-URL checks")

    works = review.get("works")
    if not isinstance(works, list) or not works:
        raise ValueError("external source review must contain works")
    expected_urls = set(evidence_manifest["external_urls"])
    expected_claims = set(evidence_manifest["claims"])
    seen_urls: dict[str, str] = {}
    seen_ids: set[str] = set()
    eligible = 0
    contextual = 0
    excluded = 0
    externally_linked_claims: set[str] = set()
    supported_claims: set[str] = set()
    contextual_claims: set[str] = set()

    for work in works:
        work_id = work.get("work_id", "<missing-work-id>")
        _require_text(work, "work_id", work_id)
        if work_id in seen_ids:
            raise ValueError(f"duplicate external work id: {work_id}")
        seen_ids.add(work_id)
        for field in (
            "title",
            "canonical_metadata_url",
            "scope_assessment",
            "correction_check_method",
        ):
            _require_text(work, field, work_id)
        if not isinstance(work.get("year"), int):
            raise ValueError(f"{work_id}: year must be an integer")
        if work.get("source_type") not in SOURCE_TYPES:
            raise ValueError(f"{work_id}: invalid source_type")
        if work.get("evidence_role") not in EVIDENCE_ROLES:
            raise ValueError(f"{work_id}: invalid evidence_role")
        if work.get("independence") not in INDEPENDENCE_STATES:
            raise ValueError(f"{work_id}: invalid independence")
        if work.get("correction_status") not in CORRECTION_STATES:
            raise ValueError(f"{work_id}: invalid correction_status")
        disposition = work.get("evidence_disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"{work_id}: invalid evidence_disposition")
        if (
            work["correction_status"] in {"retracted", "expression_of_concern"}
            and disposition == "eligible"
        ):
            raise ValueError(
                f"{work_id}: affected work cannot remain evidence-eligible"
            )
        urls = work.get("urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError(f"{work_id}: urls must be a non-empty list")
        for url in urls:
            if url in seen_urls:
                raise ValueError(
                    f"external URL assigned to both {seen_urls[url]} and {work_id}: {url}"
                )
            seen_urls[url] = work_id
        linked_claims = set(work.get("linked_claims", []))
        if not linked_claims or not linked_claims <= expected_claims:
            raise ValueError(f"{work_id}: linked_claims are missing or invalid")
        expected_linked = {
            claim_id
            for url in urls
            for claim_id in evidence_manifest["external_urls"]
            .get(url, {})
            .get("referenced_by", [])
        }
        if linked_claims != expected_linked:
            raise ValueError(f"{work_id}: linked_claims do not match URL inventory")
        supported = set(work.get("supports_claims", []))
        if not supported <= linked_claims:
            raise ValueError(f"{work_id}: supports_claims exceed linked_claims")
        if disposition == "eligible" and not supported:
            raise ValueError(f"{work_id}: eligible work supports no linked claim")
        assessments = work.get("claim_assessments")
        if not isinstance(assessments, dict) or set(assessments) != linked_claims:
            raise ValueError(f"{work_id}: claim_assessments must cover linked_claims")
        assessment_supported: set[str] = set()
        for claim_id, assessment in assessments.items():
            assessment_disposition = assessment.get("disposition")
            if assessment_disposition not in {
                "contextual_only",
                "excluded",
                "supports_with_declared_boundary",
            }:
                raise ValueError(f"{work_id}: invalid claim assessment disposition")
            _require_text(assessment, "use", f"{work_id}/{claim_id}")
            _require_text(assessment, "boundary", f"{work_id}/{claim_id}")
            if assessment_disposition == "supports_with_declared_boundary":
                assessment_supported.add(claim_id)
            elif assessment_disposition == "contextual_only":
                contextual_claims.add(claim_id)
        if assessment_supported != supported:
            raise ValueError(f"{work_id}: supports_claims disagree with assessments")
        externally_linked_claims.update(linked_claims)
        supported_claims.update(supported)
        limitations = work.get("limitations")
        if (
            not isinstance(limitations, list)
            or not limitations
            or not all(isinstance(item, str) and item.strip() for item in limitations)
        ):
            raise ValueError(f"{work_id}: limitations must be explicit")
        if disposition == "eligible":
            eligible += 1
        elif disposition == "contextual_only":
            contextual += 1
        else:
            excluded += 1

    actual_urls = set(seen_urls)
    if actual_urls != expected_urls:
        missing = sorted(expected_urls - actual_urls)
        extra = sorted(actual_urls - expected_urls)
        raise ValueError(
            f"external URL coverage mismatch: missing={missing}, extra={extra}"
        )
    if set(url_checks) != expected_urls:
        raise ValueError("availability snapshot URL coverage is stale")
    availability_counts = {"resolves": 0, "automated_access_restricted": 0}
    for url, check in url_checks.items():
        status = check.get("status")
        if status not in availability_counts:
            raise ValueError(f"unacceptable availability status for {url}: {status}")
        if not isinstance(check.get("http_status"), int):
            raise ValueError(f"availability HTTP status missing for {url}")
        _require_text(check, "final_url", url)
        availability_counts[status] += 1
    if availability.get("summary") != availability_counts:
        raise ValueError("availability snapshot summary is stale")
    summary = review.get("summary")
    expected_summary = {
        "external_url_count": len(expected_urls),
        "canonical_work_count": len(works),
        "eligible_work_count": eligible,
        "contextual_only_work_count": contextual,
        "excluded_work_count": excluded,
        "externally_linked_claim_count": len(externally_linked_claims),
        "supported_claim_count": len(supported_claims),
        "contextual_only_claim_count": len(contextual_claims - supported_claims),
        "unsupported_claim_count": len(
            externally_linked_claims - supported_claims - contextual_claims
        ),
    }
    if summary != expected_summary:
        raise ValueError("external source review summary is stale")
    return {"valid": True, **expected_summary, "availability": availability_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate",))
    parser.parse_args()
    root = _repository_root()
    review = json.loads((root / REVIEW_REL).read_text(encoding="utf-8"))
    print(json.dumps(validate_external_source_review(root, review), indent=2))


if __name__ == "__main__":
    main()
