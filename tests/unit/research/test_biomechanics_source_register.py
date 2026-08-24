"""Tests for the governed biomechanics source register."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.biomechanics_source_register import (
    BIBLIOGRAPHY_REL,
    CLAIM_REGISTRY_REL,
    EXTERNAL_REVIEW_REL,
    SOURCE_REGISTER_REL,
    validate_biomechanics_source_register,
)
from scripts.research.proximal_distal_energy.release_bundle import artifact_sha256

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def _register() -> dict:
    return json.loads((ROOT / SOURCE_REGISTER_REL).read_text(encoding="utf-8"))


def test_source_register_covers_biological_and_transport_domains() -> None:
    report = validate_biomechanics_source_register(ROOT, _register())

    assert report["valid"] is True
    assert report["source_count"] >= 15
    assert report["coverage_domain_count"] == 16
    assert report["independent_source_count"] == report["source_count"]
    assert report["source_gap_domain_count"] == 0


def test_source_register_rejects_bibliography_identifier_mismatch() -> None:
    register = _register()
    register["sources"][0]["identifier"] = "doi:10.0000/not-the-cited-work"

    with pytest.raises(ValueError, match="bibliography entry does not contain"):
        validate_biomechanics_source_register(ROOT, register)


def test_source_register_rejects_project_authored_evidence() -> None:
    register = _register()
    register["sources"][0]["independence"] = "project_author_overlap"

    with pytest.raises(ValueError, match="independent_of_project"):
        validate_biomechanics_source_register(ROOT, register)


def test_source_register_requires_complete_domain_coverage() -> None:
    register = _register()
    register["coverage"] = register["coverage"][1:]

    with pytest.raises(ValueError, match="coverage domains are incomplete"):
        validate_biomechanics_source_register(ROOT, register)


def test_source_register_summary_is_recomputed() -> None:
    register = _register()
    register["summary"]["source_count"] += 1

    with pytest.raises(ValueError, match="summary is stale"):
        validate_biomechanics_source_register(ROOT, register)


def test_source_register_digest_is_stable_across_text_line_endings(
    tmp_path: Path,
) -> None:
    register = _register()
    for relative_path in (CLAIM_REGISTRY_REL, EXTERNAL_REVIEW_REL):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative_path).read_bytes())

    bibliography_path = tmp_path / BIBLIOGRAPHY_REL
    bibliography_path.parent.mkdir(parents=True, exist_ok=True)
    bibliography_text = (ROOT / BIBLIOGRAPHY_REL).read_text(encoding="utf-8")
    bibliography_path.write_bytes(bibliography_text.replace("\n", "\r\n").encode())
    register["bibliography_sha256"] = artifact_sha256(bibliography_path)

    report = validate_biomechanics_source_register(tmp_path, register)

    assert report["valid"] is True
