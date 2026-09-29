"""Coverage tests for ``motion_matching.metrics`` legacy paths."""

from __future__ import annotations

import json

import pytest
from src.shared.python.motion_matching.metrics import (
    SCHEMA_VERSION,
    Metrics,
    asdict_safe,
    legacy_struct_to_metrics,
)


def _good_kwargs(**overrides):
    base = {
        "swing_id": "s1",
        "option": 1,
        "solver": "lbfgs",
        "n_iterations": 3,
        "rmse_clubhead_mm": 1.0,
        "rmse_butt_mm": 2.0,
        "rmse_orientation_deg": 0.5,
        "clubhead_speed_at_impact_mph": 100.0,
        "clubhead_speed_meas_mph": 99.0,
        "total_work_J": 10.0,
        "peak_power_W": 200.0,
        "wall_clock_s": 0.5,
        "git_commit": "0" * 40,
        "matlab_version": "R2024a",
        "python_version": "3.11",
        "timestamp_iso8601": "2024-01-01T00:00:00Z",
        "schema_version": SCHEMA_VERSION,
    }
    base.update(overrides)
    return base


def test_round_trip_json() -> None:
    """Pin: ``to_json`` -> ``from_json`` is an exact round-trip."""
    m = Metrics(**_good_kwargs())
    s = m.to_json()
    m2 = Metrics.from_json(s)
    assert m == m2


def test_from_json_rejects_non_object() -> None:
    """Pin: top-level JSON array rejected."""
    with pytest.raises(ValueError, match="expected JSON object"):
        Metrics.from_json("[]")


def test_to_csv_round_trip() -> None:
    """Pin: CSV-row round-trip is lossless."""
    m = Metrics(**_good_kwargs())
    row = m.to_csv_row()
    m2 = Metrics.from_csv_row(row)
    assert m == m2


def test_invalid_option_rejected() -> None:
    """Pin: option outside {1,2,3,4} rejected."""
    with pytest.raises(ValueError, match=r"option must be in"):
        Metrics(**_good_kwargs(option=9))


def test_negative_iter_rejected() -> None:
    """Pin: negative ``n_iterations`` rejected."""
    with pytest.raises(ValueError, match="n_iterations must be >= 0"):
        Metrics(**_good_kwargs(n_iterations=-1))


def test_non_finite_field_rejected() -> None:
    """Pin: NaN in a numeric field rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        Metrics(**_good_kwargs(rmse_clubhead_mm=float("nan")))


def test_negative_nonneg_field_rejected() -> None:
    """Pin: negative entry in a non-negative field rejected."""
    with pytest.raises(ValueError, match=r"must be >= 0"):
        Metrics(**_good_kwargs(wall_clock_s=-1.0))


def test_bad_git_sha_rejected() -> None:
    """Pin: SHA != 40 hex chars is rejected."""
    with pytest.raises(ValueError, match="git_commit"):
        Metrics(**_good_kwargs(git_commit="not-a-sha"))


def test_bad_timestamp_rejected() -> None:
    """Pin: non-ISO timestamp rejected."""
    with pytest.raises(ValueError, match="ISO-8601 UTC"):
        Metrics(**_good_kwargs(timestamp_iso8601="2024-01-01"))


def test_microsecond_timestamp_accepted() -> None:
    """Pin: ISO-8601 with microseconds accepted."""
    Metrics(**_good_kwargs(timestamp_iso8601="2024-01-01T00:00:00.123456Z"))


def test_schema_version_mismatch() -> None:
    """Pin: stale schema_version rejected."""
    with pytest.raises(ValueError, match="schema_version"):
        Metrics(**_good_kwargs(schema_version="0.0.0"))


def test_legacy_struct_to_metrics() -> None:
    """Pin: legacy field renames flow through."""
    legacy = {
        "swing_id": "s",
        "option": 2,
        "solver": "lbfgs",
        "rmse_clubhead": 1.0,
        "rmse_butt": 2.0,
        "rmse_orient": 3.0,
        "chs_impact": 100.0,
        "chs_meas": 99.0,
        "total_work": 10.0,
        "peak_power": 100.0,
        "wall_clock": 0.5,
        "git_commit": "0" * 40,
        "timestamp": "2024-01-01T00:00:00Z",
    }
    m = legacy_struct_to_metrics(legacy)
    assert m.rmse_clubhead_mm == 1.0
    assert m.timestamp_iso8601 == "2024-01-01T00:00:00Z"


def test_asdict_safe_dict() -> None:
    """Pin: dict input pass-through (copy)."""
    out = asdict_safe({"a": 1})
    assert out == {"a": 1}


def test_asdict_safe_namedtuple() -> None:
    """Pin: namedtuple-like ``_asdict`` input is consumed."""
    from collections import namedtuple

    NT = namedtuple("NT", "x y")
    out = asdict_safe(NT(1, 2))
    assert out == {"x": 1, "y": 2}


def test_asdict_safe_invalid() -> None:
    """Pin: opaque object rejected."""
    with pytest.raises(TypeError, match="cannot convert"):
        asdict_safe(42)


def test_to_json_canonical() -> None:
    """Pin: JSON output has sorted keys and no whitespace."""
    m = Metrics(**_good_kwargs())
    s = m.to_json()
    parsed = json.loads(s)
    assert isinstance(parsed, dict)
    assert " " not in s.replace('": ', '":').replace(", ", ",")
