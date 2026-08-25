"""Tests for the biomechanics model-to-measurement authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.biomechanics_evidence_bridge import (
    BRIDGE_REL,
    EXTERNAL_REVIEW_REL,
    _sha256,
    validate_biomechanics_evidence_bridge,
)

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def _load(relative: Path) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _bridge() -> dict:
    return _load(BRIDGE_REL)


def _external_review() -> dict:
    return _load(EXTERNAL_REVIEW_REL)


def test_bridge_digests_are_stable_across_text_line_endings(tmp_path: Path) -> None:
    """Text provenance must identify repository content, not checkout style."""
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(b'{\n  "claim": "bounded"\n}\n')
    crlf_path.write_bytes(b'{\r\n  "claim": "bounded"\r\n}\r\n')

    assert _sha256(lf_path) == _sha256(crlf_path)


def test_bridge_covers_required_modalities_and_retains_open_gates() -> None:
    report = validate_biomechanics_evidence_bridge(ROOT, _bridge(), _external_review())

    assert report["valid"] is True
    assert report["modality_count"] == 7
    assert report["mechanism_count"] >= 8
    assert report["source_registered_modality_count"] >= 2
    assert report["source_gap_modality_count"] >= 4
    assert report["human_validation_status"] == "externally_blocked"
    assert report["transportability_dimension_count"] == 9


def test_bridge_rejects_missing_required_modality() -> None:
    bridge = _bridge()
    bridge["modalities"] = bridge["modalities"][1:]

    with pytest.raises(ValueError, match="required modality coverage"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())


def test_registered_sources_must_exist_in_external_review() -> None:
    bridge = _bridge()
    bridge["modalities"][0]["source_ids"].append("doi:missing")

    with pytest.raises(ValueError, match="unknown source_ids"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())


def test_registered_sources_must_be_independent_and_evidence_eligible() -> None:
    bridge = _bridge()
    review = _external_review()
    source_id = bridge["modalities"][0]["source_ids"][0]
    source = next(work for work in review["works"] if work["work_id"] == source_id)
    source["independence"] = "project_author_overlap"

    with pytest.raises(ValueError, match="independent evidence-eligible"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, review)


def test_source_gap_requires_an_explicit_data_gate() -> None:
    bridge = _bridge()
    gap = next(item for item in bridge["modalities"] if item["source_status"] == "gap")
    gap["data_gate"] = ""

    with pytest.raises(ValueError, match="data_gate"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())


def test_bilateral_allocation_cannot_be_promoted_from_motion_only() -> None:
    bridge = _bridge()
    allocation = next(
        item
        for item in bridge["mechanisms"]
        if item["mechanism_id"] == "bilateral_hand_wrench_allocation"
    )
    allocation["identifiability"] = "directly_observed"

    with pytest.raises(ValueError, match="bilateral hand allocation"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())


def test_schema_rejects_unknown_fields_and_requires_processing_authority() -> None:
    bridge = _bridge()
    bridge["modalities"][0]["undeclared_extension"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())

    missing_processing = _bridge()
    missing_processing["modalities"][0]["processing_method"] = ""
    with pytest.raises(ValueError, match="processing_method"):
        validate_biomechanics_evidence_bridge(
            ROOT, missing_processing, _external_review()
        )


def test_transportability_dimensions_are_complete_and_fail_closed() -> None:
    bridge = _bridge()
    bridge["transportability"] = bridge["transportability"][:-1]

    with pytest.raises(ValueError, match="transportability coverage"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())


def test_summary_and_external_review_digest_are_fail_closed() -> None:
    stale_summary = _bridge()
    stale_summary["summary"]["mechanism_count"] += 1
    with pytest.raises(ValueError, match="summary is stale"):
        validate_biomechanics_evidence_bridge(ROOT, stale_summary, _external_review())

    stale_digest = copy.deepcopy(_bridge())
    stale_digest["external_source_review_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="external source review digest"):
        validate_biomechanics_evidence_bridge(ROOT, stale_digest, _external_review())

    stale_source_digest = copy.deepcopy(_bridge())
    stale_source_digest["source_register_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="biomechanics source register digest"):
        validate_biomechanics_evidence_bridge(
            ROOT, stale_source_digest, _external_review()
        )


def test_every_mechanism_requires_an_observable_discriminator() -> None:
    bridge = _bridge()
    bridge["mechanisms"][0]["observable_discriminator"] = ""

    with pytest.raises(ValueError, match="observable_discriminator"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())


def test_source_register_claim_links_must_be_reciprocal() -> None:
    bridge = _bridge()
    for mechanism in bridge["mechanisms"]:
        mechanism["claim_ids"] = [
            claim_id
            for claim_id in mechanism["claim_ids"]
            if claim_id != "PD-CLAIM-200"
        ]
    for dimension in bridge["transportability"]:
        dimension["claim_ids"] = [
            claim_id
            for claim_id in dimension["claim_ids"]
            if claim_id != "PD-CLAIM-200"
        ]

    with pytest.raises(ValueError, match="not reciprocated"):
        validate_biomechanics_evidence_bridge(ROOT, bridge, _external_review())
