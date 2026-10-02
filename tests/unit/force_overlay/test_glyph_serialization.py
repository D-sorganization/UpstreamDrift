"""Unit tests for GlyphSet JSON serialization and schema validation (FTO-3, #11288)."""

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

pytestmark = pytest.mark.unit

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "glyph-set-v1.json"
EXAMPLES_PATH = (
    Path(__file__).resolve().parents[3] / "schemas" / "glyph-set-examples.json"
)


def test_glyph_set_to_dict_and_from_dict_roundtrip() -> None:
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 600.0),
        torque_nm=(0.0, 0.0, 10.0),
        source="test",
    )
    frame = ForceTorqueFrame(time_s=0.05, engine="mujoco", wrenches=(w1,))
    glyphs = build_glyphs(frame)

    d = glyphs.to_dict()
    assert d["schema_version"] == "glyph-set-v1"
    assert d["time_s"] == 0.05
    assert len(d["arrows"]) == 1
    assert len(d["torque_arcs"]) == 1

    restored = GlyphSet.from_dict(d)
    assert restored.time_s == glyphs.time_s
    assert restored.schema_version == glyphs.schema_version
    assert len(restored.arrows) == len(glyphs.arrows)
    assert len(restored.torque_arcs) == len(glyphs.torque_arcs)
    assert restored.arrows[0].label == glyphs.arrows[0].label
    assert restored.arrows[0].tip_m == pytest.approx(glyphs.arrows[0].tip_m)
    assert restored.torque_arcs[0].radius_m == pytest.approx(
        glyphs.torque_arcs[0].radius_m
    )
    assert restored.legend.engine == glyphs.legend.engine


def test_glyph_set_rejects_unknown_keys_and_wrong_version() -> None:
    valid_dict = {
        "schema_version": "glyph-set-v1",
        "time_s": 0.0,
        "arrows": [],
        "torque_arcs": [],
        "legend": {
            "force_reference_n": None,
            "force_reference_length_m": None,
            "torque_reference_nm": None,
            "torque_reference_radius_m": None,
            "kinds_present": [],
            "unavailable_labels": [],
            "engine": "test",
            "source_labels": [],
        },
    }

    # Wrong version
    bad_version = valid_dict.copy()
    bad_version["schema_version"] = "glyph-set-v2"
    with pytest.raises(ValueError, match="schema_version must be 'glyph-set-v1'"):
        GlyphSet.from_dict(bad_version)

    # Unknown key
    bad_key = valid_dict.copy()
    bad_key["unexpected"] = 123
    with pytest.raises(ValueError, match="Unknown keys in GlyphSet"):
        GlyphSet.from_dict(bad_key)


def test_examples_against_schema_and_freshness() -> None:
    if not SCHEMA_PATH.exists() or not EXAMPLES_PATH.exists():
        pytest.skip("Schema or examples fixture not yet created")

    import jsonschema

    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    with EXAMPLES_PATH.open("r", encoding="utf-8") as f:
        examples = json.load(f)

    validator = jsonschema.Draft202012Validator(schema)

    for name, fixture in examples.items():
        errors = list(validator.iter_errors(fixture))
        assert not errors, (
            f"Fixture {name} failed schema: {[e.message for e in errors]}"
        )
        restored = GlyphSet.from_dict(fixture)
        assert restored.schema_version == "glyph-set-v1"
