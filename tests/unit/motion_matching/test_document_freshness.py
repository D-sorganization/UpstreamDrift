"""Tests for anthropometric document freshness against de Leva table (HO-11 #10250)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.anthropometry import (
    DE_LEVA_MALE,
    de_leva_table_dict,
    de_leva_table_sha256,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs" / "development" / "full_body_models"
COMMITTED_ANTHRO_DOCS = (
    DOCS_DIR / "full_body_spec_anthro_driver.json",
    DOCS_DIR / "full_body_spec_anthro_iron7.json",
)


def validate_document_freshness(document: Mapping[str, Any]) -> None:
    """Validate that an anthropometric document matches the current DE_LEVA_MALE table.

    Design by Contract:
    - Precondition: `document` is a Mapping.
    - Precondition: `document` has a `de_leva_table_sha256` field matching `de_leva_table_sha256()`.
    - Precondition: `document` has an `anthropometry` block with all segments covered in `DE_LEVA_MALE`.
    - Postcondition: Every segment in the document's anthropometry block matches
      `DE_LEVA_MALE` parameters (length, mass fraction, CoM fraction, and radii of gyration).
    """
    if not isinstance(document, Mapping):
        raise TypeError("Document must be a mapping")

    table_sha = document.get("de_leva_table_sha256")
    if table_sha is None:
        raise ValueError("Document is missing required 'de_leva_table_sha256' field")
    current_sha = de_leva_table_sha256()
    if table_sha != current_sha:
        raise ValueError(
            f"Document de_leva_table_sha256 '{table_sha}' does not match current table sha256 '{current_sha}'"
        )

    anthro = document.get("anthropometry")
    if anthro is None or not isinstance(anthro, Mapping):
        raise ValueError("Document is missing required 'anthropometry' mapping")

    segments = anthro.get("segments")
    if segments is None or not isinstance(segments, Mapping):
        raise ValueError("Document anthropometry block is missing 'segments' mapping")

    expected_segments = set(DE_LEVA_MALE.keys())
    missing_segments = expected_segments - set(segments.keys())
    if missing_segments:
        raise ValueError(
            f"Anthropometry block is missing required segments: {sorted(missing_segments)}"
        )

    for seg_name, expected in DE_LEVA_MALE.items():
        actual = segments[seg_name]
        if not isinstance(actual, Mapping):
            raise TypeError(f"Segment '{seg_name}' data must be a mapping")

        for key in ("length_m", "mass_fraction", "com_fraction", "radii"):
            if key not in actual:
                raise ValueError(f"Segment '{seg_name}' missing field '{key}'")

        if actual["com_fraction"] != pytest.approx(expected.com_fraction, abs=1e-6):
            raise ValueError(
                f"Segment '{seg_name}' com_fraction {actual['com_fraction']} "
                f"does not match expected {expected.com_fraction}"
            )
        if actual["radii"] != pytest.approx(list(expected.radii), abs=1e-6):
            raise ValueError(
                f"Segment '{seg_name}' radii {actual['radii']} "
                f"does not match expected {list(expected.radii)}"
            )
        if actual["mass_fraction"] != pytest.approx(expected.mass_fraction, abs=1e-6):
            raise ValueError(
                f"Segment '{seg_name}' mass_fraction {actual['mass_fraction']} "
                f"does not match expected {expected.mass_fraction}"
            )
        if actual["length_m"] != pytest.approx(expected.length_m, abs=1e-6):
            raise ValueError(
                f"Segment '{seg_name}' length_m {actual['length_m']} "
                f"does not match expected {expected.length_m}"
            )


@pytest.mark.parametrize("doc_path", COMMITTED_ANTHRO_DOCS)
def test_committed_anthropometric_documents_are_fresh(doc_path: Path) -> None:
    """Committed full-body anthropometric documents match the current DE_LEVA_MALE table."""
    assert doc_path.is_file(), f"Document not found at {doc_path}"
    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    validate_document_freshness(doc)


def test_validate_document_freshness_type_error() -> None:
    """DbC: validate_document_freshness raises TypeError when input is not a Mapping."""
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_document_freshness("not a mapping")  # type: ignore[arg-type]


def test_validate_document_freshness_missing_hash() -> None:
    """DbC: validate_document_freshness raises ValueError when de_leva_table_sha256 is missing."""
    doc = {"anthropometry": {"segments": de_leva_table_dict()}}
    with pytest.raises(ValueError, match="de_leva_table_sha256"):
        validate_document_freshness(doc)


def test_validate_document_freshness_stale_hash() -> None:
    """DbC: validate_document_freshness raises ValueError when de_leva_table_sha256 does not match."""
    doc = {
        "de_leva_table_sha256": "0" * 64,
        "anthropometry": {"segments": de_leva_table_dict()},
    }
    with pytest.raises(ValueError, match="does not match current table sha256"):
        validate_document_freshness(doc)


def test_validate_document_freshness_missing_anthropometry() -> None:
    """DbC: validate_document_freshness raises ValueError when anthropometry block is missing."""
    doc = {"de_leva_table_sha256": de_leva_table_sha256()}
    with pytest.raises(ValueError, match="missing required 'anthropometry'"):
        validate_document_freshness(doc)


def test_validate_document_freshness_missing_segments() -> None:
    """DbC: validate_document_freshness raises ValueError when segments are missing."""
    doc = {
        "de_leva_table_sha256": de_leva_table_sha256(),
        "anthropometry": {"segments": {"head": de_leva_table_dict()["head"]}},
    }
    with pytest.raises(ValueError, match="missing required segments"):
        validate_document_freshness(doc)


def test_validate_document_freshness_stale_shank_com() -> None:
    """DbC: validate_document_freshness catches stale shank com_fraction (pre-HO-9 value 0.4459)."""
    tbl = de_leva_table_dict()
    tbl["shank"]["com_fraction"] = 0.4459  # Old pre-HO-9 value
    doc = {
        "de_leva_table_sha256": de_leva_table_sha256(),
        "anthropometry": {"segments": tbl},
    }
    with pytest.raises(ValueError, match=r"Segment 'shank' com_fraction"):
        validate_document_freshness(doc)


def test_validate_document_freshness_stale_shank_radii() -> None:
    """DbC: validate_document_freshness catches stale shank radii (pre-HO-9 values)."""
    tbl = de_leva_table_dict()
    tbl["shank"]["radii"] = [0.255, 0.249, 0.103]  # Old pre-HO-9 radii
    doc = {
        "de_leva_table_sha256": de_leva_table_sha256(),
        "anthropometry": {"segments": tbl},
    }
    with pytest.raises(ValueError, match=r"Segment 'shank' radii"):
        validate_document_freshness(doc)
