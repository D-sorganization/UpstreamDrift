"""Tests for deterministic claim-support integrity records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.claim_evidence_integrity import (
    build_claim_evidence_manifest,
    validate_claim_evidence_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.unit


def test_builder_hash_pins_all_local_support_and_inventories_external_links() -> None:
    registry = json.loads((ARTICLE / "data/claim_audit_registry.json").read_text())
    expected_references = sum(
        len(claim["evidence_artifacts"]) for claim in registry["claims"]
    )

    manifest = build_claim_evidence_manifest(ROOT)

    assert manifest["summary"]["claim_count"] == len(registry["claims"])
    assert manifest["summary"]["evidence_reference_count"] == expected_references
    assert manifest["summary"]["local_artifact_count"] >= 190
    assert manifest["summary"]["external_url_count"] >= 80
    assert manifest["scope"]["external_url_semantics"] == (
        "inventory_only_not_scientific_validation"
    )
    assert all(
        len(record["sha256"]) == 64 and record["bytes"] >= 0
        for record in manifest["local_artifacts"].values()
    )
    assert set(manifest["claims"]) == {
        claim["claim_id"] for claim in registry["claims"]
    }


def test_committed_claim_evidence_manifest_matches_builder_and_validates() -> None:
    committed = json.loads(
        (ARTICLE / "data/claim_evidence_manifest.json").read_text(encoding="utf-8")
    )

    assert committed == build_claim_evidence_manifest(ROOT)
    report = validate_claim_evidence_manifest(ROOT, committed)

    assert report["valid"] is True
    assert report["mismatches"] == []


def test_claim_evidence_validator_fails_closed_on_hash_tampering() -> None:
    manifest = build_claim_evidence_manifest(ROOT)
    first_path = next(iter(manifest["local_artifacts"]))
    manifest["local_artifacts"][first_path]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="claim evidence manifest validation failed"):
        validate_claim_evidence_manifest(ROOT, manifest)


def test_claim_evidence_validator_fails_closed_on_omitted_claim() -> None:
    manifest = build_claim_evidence_manifest(ROOT)
    manifest["claims"].pop(next(iter(manifest["claims"])))

    with pytest.raises(ValueError, match="claim evidence manifest validation failed"):
        validate_claim_evidence_manifest(ROOT, manifest)
