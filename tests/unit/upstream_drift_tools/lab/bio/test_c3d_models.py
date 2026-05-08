"""Unit tests for ``_c3d_models`` dataclass validation and constants."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.lab.bio._c3d_models import (
    BIOMECHANICAL_MARKER_MAX_M,
    BIOMECHANICAL_MARKER_MIN_M,
    SCHEMA_VERSION,
    C3DEvent,
    C3DMetadata,
)


def test_schema_version_invariants() -> None:
    assert SCHEMA_VERSION == "1.0"
    assert isinstance(SCHEMA_VERSION, str)


def test_biomechanical_constants() -> None:
    assert BIOMECHANICAL_MARKER_MIN_M == 0.001
    assert BIOMECHANICAL_MARKER_MAX_M == 10.0


def test_c3d_event_valid() -> None:
    event = C3DEvent(label="FootStrike", time=1.5)
    assert event.label == "FootStrike"
    assert event.time == 1.5


def test_c3d_event_empty_label_rejected() -> None:
    with pytest.raises(ValueError, match="Event label cannot be empty"):
        C3DEvent(label="", time=0.0)


def _valid_kwargs() -> dict:
    return {
        "marker_labels": ["M1", "M2"],
        "frame_count": 10,
        "frame_rate": 100.0,
        "units": "m",
        "analog_labels": ["A1"],
        "analog_units": ["V"],
        "analog_rate": 1000.0,
        "events": [],
    }


def test_metadata_valid() -> None:
    md = C3DMetadata(**_valid_kwargs())
    assert md.marker_count == 2
    assert md.analog_count == 1
    assert md.duration == pytest.approx(10 / 100.0)


def test_metadata_duration_zero_rate() -> None:
    kw = _valid_kwargs()
    kw["frame_rate"] = 0.0
    md = C3DMetadata(**kw)
    assert md.duration == 0.0


def test_metadata_negative_frame_count() -> None:
    kw = _valid_kwargs()
    kw["frame_count"] = -1
    with pytest.raises(ValueError, match="Frame count cannot be negative"):
        C3DMetadata(**kw)


def test_metadata_negative_frame_rate() -> None:
    kw = _valid_kwargs()
    kw["frame_rate"] = -1.0
    with pytest.raises(ValueError, match="Frame rate cannot be negative"):
        C3DMetadata(**kw)


def test_metadata_negative_analog_rate() -> None:
    kw = _valid_kwargs()
    kw["analog_rate"] = -1.0
    with pytest.raises(ValueError, match="Analog rate cannot be negative"):
        C3DMetadata(**kw)


def test_metadata_analog_rate_none_allowed() -> None:
    kw = _valid_kwargs()
    kw["analog_rate"] = None
    md = C3DMetadata(**kw)
    assert md.analog_rate is None


def test_metadata_analog_units_label_mismatch() -> None:
    kw = _valid_kwargs()
    kw["analog_units"] = ["V", "V"]
    with pytest.raises(ValueError, match="analog_units and analog_labels"):
        C3DMetadata(**kw)
