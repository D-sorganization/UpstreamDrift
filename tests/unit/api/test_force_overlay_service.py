"""Unit tests for ForceOverlayService (#11307, FTO-22)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from src.shared.python.force_overlay import (
    ForceGlyphStyle,
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
    build_glyphs,
)
from src.api.services.force_overlay_service import (
    current_force_frame,
    force_overlay_payload,
    style_from_request_params,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
GLYPH_SCHEMA_PATH = REPO_ROOT / "schemas" / "glyph-set-v1.json"
FRAME_SCHEMA_PATH = REPO_ROOT / "schemas" / "force-torque-frame-v1.json"


def _make_sample_frame(time_s: float = 0.5) -> ForceTorqueFrame:
    """Construct a deterministic sample ForceTorqueFrame for testing."""
    wrenches = (
        OverlayWrench(
            kind=WrenchKind.JOINT_ACTUATOR,
            label="joint_actuator:shoulder",
            body="arm",
            point_m=(0.0, 1.0, 0.0),
            torque_nm=(0.0, 0.0, 15.0),
            source="test_engine",
        ),
        OverlayWrench(
            kind=WrenchKind.JOINT_REACTION,
            label="joint_reaction:shoulder",
            body="arm",
            point_m=(0.0, 1.0, 0.0),
            force_n=(0.0, 0.0, -100.0),
            source="test_engine",
        ),
        OverlayWrench(
            kind=WrenchKind.CONTACT,
            label="contact:foot_ground",
            body="foot",
            point_m=(0.1, 0.0, 0.0),
            force_n=(0.0, 0.0, 800.0),
            source="test_engine",
        ),
    )
    return ForceTorqueFrame(time_s=time_s, engine="test_engine", wrenches=wrenches)


class FakeProvider:
    """Mock engine satisfying ForceTorqueProvider."""

    def __init__(self, frame: ForceTorqueFrame | None = None) -> None:
        self._frame = frame or _make_sample_frame()

    def get_force_torque_frame(self) -> ForceTorqueFrame | None:
        return self._frame


class NonProvider:
    """Mock engine not satisfying ForceTorqueProvider."""

    def get_state(self) -> dict[str, Any]:
        return {"time": 0.5}


def test_current_force_frame_with_provider() -> None:
    expected = _make_sample_frame(1.23)
    provider = FakeProvider(expected)
    result = current_force_frame(provider)
    assert result is not None
    assert result.time_s == 1.23
    assert len(result.wrenches) == 3


def test_current_force_frame_non_provider() -> None:
    non_provider = NonProvider()
    assert current_force_frame(non_provider) is None
    assert current_force_frame(None) is None


def test_current_force_frame_provider_returns_none() -> None:
    provider = FakeProvider(None)
    provider._frame = None
    assert current_force_frame(provider) is None


def test_force_overlay_payload_equals_build_glyphs() -> None:
    frame = _make_sample_frame(0.42)
    style = ForceGlyphStyle(min_length_m=0.03)

    payload = force_overlay_payload(frame, style)
    assert payload is not None
    assert "glyphs" in payload
    assert "frame" in payload
    assert "style" in payload

    expected_glyphs = build_glyphs(frame, style).to_dict()
    assert payload["glyphs"] == expected_glyphs
    assert payload["frame"] == frame.to_dict()
    assert payload["style"] == style.to_dict()


def test_force_overlay_payload_none_frame() -> None:
    payload = force_overlay_payload(None)
    assert payload["glyphs"] is None
    assert payload["frame"] is None
    assert payload.get("unavailable_reason") is not None


def test_force_overlay_payload_body_filter() -> None:
    frame = _make_sample_frame()
    style = ForceGlyphStyle()
    payload = force_overlay_payload(frame, style, body_filter=["foot"])

    glyphs = payload["glyphs"]
    assert glyphs is not None
    # Only the contact wrench on 'foot' should be present
    assert len(glyphs["arrows"]) == 1
    assert glyphs["arrows"][0]["label"] == "contact:foot_ground"
    assert len(glyphs["torque_arcs"]) == 0


def test_style_from_request_params_mapping() -> None:
    style = style_from_request_params(
        force_types=["applied", "contact"],
        scale_factor=0.05,
        show_labels=True,
    )
    assert WrenchKind.JOINT_ACTUATOR in style.kinds
    assert WrenchKind.CONTACT in style.kinds
    assert WrenchKind.GRAVITY not in style.kinds
    assert style.show_labels is True
    assert style.force_scale_m_per_n > 0


def test_payload_schema_validation() -> None:
    frame = _make_sample_frame(0.5)
    style = ForceGlyphStyle()
    payload = force_overlay_payload(frame, style)

    with open(GLYPH_SCHEMA_PATH, encoding="utf-8") as f:
        glyph_schema = json.load(f)
    with open(FRAME_SCHEMA_PATH, encoding="utf-8") as f:
        frame_schema = json.load(f)

    jsonschema.validate(instance=payload["glyphs"], schema=glyph_schema)
    jsonschema.validate(instance=payload["frame"], schema=frame_schema)
