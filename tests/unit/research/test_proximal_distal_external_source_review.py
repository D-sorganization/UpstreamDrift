"""Tests for the external-source scientific review authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.external_source_review import (
    REVIEW_REL,
    validate_external_source_review,
)

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def _review() -> dict:
    return json.loads((ROOT / REVIEW_REL).read_text(encoding="utf-8"))


def test_external_source_review_covers_every_url_and_consolidates_mirrors() -> None:
    review = _review()
    report = validate_external_source_review(ROOT, review)

    assert report["valid"] is True
    assert report["external_url_count"] >= 75
    assert report["canonical_work_count"] < report["external_url_count"]
    assert report["unsupported_claim_count"] == 0
    assert sum(report["availability"].values()) == report["external_url_count"]
    golf_model = next(
        work for work in review["works"] if work["work_id"] == "doi:10.1115/1.3426951"
    )
    assert len(golf_model["urls"]) == 2


def test_external_source_review_requires_explicit_correction_status() -> None:
    review = _review()
    review["works"][0].pop("correction_status")

    with pytest.raises(ValueError, match="invalid correction_status"):
        validate_external_source_review(ROOT, review)


def test_external_source_review_rejects_duplicate_url_assignment() -> None:
    review = _review()
    duplicate = review["works"][0]["urls"][0]
    review["works"][1]["urls"].append(duplicate)

    with pytest.raises(ValueError, match="assigned to both"):
        validate_external_source_review(ROOT, review)


def test_external_source_review_fails_when_inventory_gains_an_unreviewed_url() -> None:
    manifest = json.loads(
        (
            ROOT
            / "docs/research/proximal_distal_energy_transfer/data/claim_evidence_manifest.json"
        ).read_text(encoding="utf-8")
    )
    drifted = copy.deepcopy(manifest)
    drifted["external_urls"]["https://example.invalid/unreviewed"] = {
        "scheme": "https",
        "host": "example.invalid",
        "referenced_by": [next(iter(drifted["claims"]))],
    }

    with pytest.raises(ValueError, match="external URL coverage mismatch"):
        validate_external_source_review(ROOT, _review(), drifted)


def test_affected_source_cannot_be_marked_eligible() -> None:
    review = _review()
    work = review["works"][0]
    work["correction_status"] = "retracted"
    work["evidence_disposition"] = "eligible"

    with pytest.raises(ValueError, match="cannot remain evidence-eligible"):
        validate_external_source_review(ROOT, review)


def test_broken_or_unchecked_link_cannot_pass_availability_review() -> None:
    review = _review()
    first_url = next(iter(review["availability_snapshot"]["url_checks"]))
    review["availability_snapshot"]["url_checks"][first_url]["status"] = "broken"

    with pytest.raises(ValueError, match="unacceptable availability status"):
        validate_external_source_review(ROOT, review)
