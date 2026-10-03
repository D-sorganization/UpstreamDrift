"""Tests for GlyphSet serialization and schema round-trip (#11288)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    build_glyphs,
)
from scripts.generate_glyph_set_examples import build_example_cases

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def schema(repo_root: Path) -> dict:
    schema_path = repo_root / "schemas" / "glyph-set-v1.json"
    assert schema_path.exists(), f"Schema file not found: {schema_path}"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fixtures(repo_root: Path) -> dict:
    fixtures_path = repo_root / "schemas" / "glyph-set-examples.json"
    assert fixtures_path.exists(), f"Examples file not found: {fixtures_path}"
    with open(fixtures_path, encoding="utf-8") as f:
        return json.load(f)


def test_glyph_set_to_from_dict_round_trip() -> None:
    """GlyphSet round-trips through to_dict and from_dict preserving all values."""
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        source="test",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="joint:motor",
        body="link1",
        point_m=(1.0, 1.0, 1.0),
        torque_nm=(10.0, 0.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.5, engine="test_eng", wrenches=(w1, w2))
    glyph_set = build_glyphs(frame, ForceGlyphStyle())

    data = glyph_set.to_dict()
    assert data["schema_version"] == "glyph-set-v1"
    assert data["time_s"] == 0.5
    assert len(data["arrows"]) == 1
    assert len(data["torque_arcs"]) == 1

    restored = GlyphSet.from_dict(data)
    assert restored.time_s == glyph_set.time_s
    assert len(restored.arrows) == len(glyph_set.arrows)
    assert len(restored.torque_arcs) == len(glyph_set.torque_arcs)
    assert restored.arrows[0].label == "contact:ground"
    assert restored.torque_arcs[0].label == "joint:motor"
    assert restored == glyph_set


def test_glyph_set_from_dict_validation() -> None:
    """GlyphSet.from_dict rejects invalid schemas and unknown keys."""
    w = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="c:1",
        body="b1",
        point_m=(0.0, 0.0, 0.0),
        force_n=(100.0, 0.0, 0.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.0, engine="test", wrenches=(w,))
    data = build_glyphs(frame, ForceGlyphStyle()).to_dict()

    # Wrong schema version
    bad_version = dict(data)
    bad_version["schema_version"] = "wrong-v1"
    with pytest.raises(ValueError):
        GlyphSet.from_dict(bad_version)

    # Unknown key
    bad_key = dict(data)
    bad_key["unknown_extra_field"] = "bad"
    with pytest.raises(ValueError):
        GlyphSet.from_dict(bad_key)


def test_glyph_schema_valid_draft_2020_12(schema: dict) -> None:
    """glyph-set-v1.json must be valid according to Draft 2020-12."""
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)


def test_glyph_examples_validate_against_schema(schema: dict, fixtures: dict) -> None:
    """All valid cases in glyph-set-examples.json must validate against schema."""
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)

    cases = fixtures.get("cases", [])
    assert len(cases) >= 4, f"Expected >= 4 cases, found {len(cases)}"

    for case in cases:
        data = case["data"]
        is_valid = case.get("valid", True)
        errors = list(validator.iter_errors(data))
        if is_valid:
            assert not errors, (
                f"Case {case['name']!r} failed schema validation: {[e.message for e in errors]}"
            )
        else:
            assert errors, (
                f"Invalid case {case['name']!r} unexpectedly passed schema validation"
            )


def test_glyph_examples_freshness(repo_root: Path, fixtures: dict) -> None:
    """Committed fixtures must match freshly generated cases (freshness guard)."""
    fresh_cases = build_example_cases()
    committed_cases = fixtures.get("cases", [])
    assert len(fresh_cases) == len(committed_cases)

    for fresh, committed in zip(fresh_cases, committed_cases, strict=True):
        assert fresh["name"] == committed["name"]
        assert fresh["valid"] == committed["valid"]
        assert fresh["data"] == committed["data"]
