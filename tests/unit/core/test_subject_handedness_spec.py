"""Spec/receipt handedness (HRG-GR0, #13201): one accessor, sparse default."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.motion_matching.anthropometric_geometry import (
    build_upper_body,
)

REPO = Path(__file__).resolve().parents[3]
COMMITTED = REPO / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def test_missing_field_resolves_to_right() -> None:
    assert subject_handedness({"subject": {}}) == "right"
    assert subject_handedness({}) == "right"


def test_declared_left_and_invalid() -> None:
    assert subject_handedness({"subject": {"handedness": "left"}}) == "left"
    with pytest.raises(ValueError, match="handedness"):
        subject_handedness({"subject": {"handedness": "ambi"}})


def test_committed_spec_unchanged_by_accessor() -> None:
    before = hashlib.sha256(COMMITTED.read_bytes()).hexdigest()
    doc = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert "handedness" not in doc["subject"]
    assert subject_handedness(doc) == "right"
    assert hashlib.sha256(COMMITTED.read_bytes()).hexdigest() == before


def test_builder_writes_field_only_when_not_default() -> None:
    default = build_upper_body(stature_m=1.8, mass_kg=80.0)
    assert "handedness" not in default["subject"]
    left = build_upper_body(stature_m=1.8, mass_kg=80.0, handedness="left")
    assert left["subject"]["handedness"] == "left"
    with pytest.raises(ValueError, match="handedness"):
        build_upper_body(stature_m=1.8, mass_kg=80.0, handedness="x")


def test_receipt_schema_records_optional_handedness() -> None:
    from src.shared.python.motion_matching.pipeline.receipt_schema import (
        validate_receipt,
    )

    path = REPO / "docs/development/full_body_models/evidence/ground_support"
    doc = json.loads((path / "anthro_driver/receipt.json").read_text("utf-8"))
    assert "handedness" not in doc  # existing receipt: untouched, still valid
    assert validate_receipt(doc).handedness is None
    assert validate_receipt({**doc, "handedness": "left"}).handedness == "left"
    with pytest.raises(ValueError):
        validate_receipt({**doc, "handedness": "ambi"})
