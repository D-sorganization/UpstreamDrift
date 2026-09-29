"""Unit tests for the canonical motion-matching Metrics record."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
from src.shared.python.motion_matching.metrics import (
    SCHEMA_VERSION,
    Metrics,
    legacy_struct_to_metrics,
)

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "motion_matching"
    / "shared"
    / "fixtures"
    / "metrics_example.json"
)


def _good_kwargs() -> dict:
    return {
        "swing_id": "TW_ProV1",
        "option": 1,
        "solver": "fmincon-sqp",
        "n_iterations": 42,
        "rmse_clubhead_mm": 3.21,
        "rmse_butt_mm": 1.85,
        "rmse_orientation_deg": 2.4,
        "clubhead_speed_at_impact_mph": 112.3,
        "clubhead_speed_meas_mph": 113.5,
        "total_work_J": 310.5,
        "peak_power_W": 2845.0,
        "wall_clock_s": 17.4,
        "git_commit": "a" * 40,
        "matlab_version": "R2024b",
        "python_version": "",
        "timestamp_iso8601": "2026-05-05T17:34:21Z",
        "schema_version": SCHEMA_VERSION,
    }


def test_metrics_immutable() -> None:
    m = Metrics(**_good_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.option = 2  # type: ignore[misc]


def test_metrics_validates_finite_values() -> None:
    bad = _good_kwargs()
    bad["rmse_clubhead_mm"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        Metrics(**bad)
    bad["rmse_clubhead_mm"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        Metrics(**bad)


def test_metrics_validates_iso8601_timestamp() -> None:
    bad = _good_kwargs()
    bad["timestamp_iso8601"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        Metrics(**bad)
    bad["timestamp_iso8601"] = "2026-05-05T17:34:21+00:00"  # not 'Z' suffix
    with pytest.raises(ValueError, match="ISO-8601"):
        Metrics(**bad)


def test_metrics_validates_schema_version() -> None:
    bad = _good_kwargs()
    bad["schema_version"] = "0.9.0"
    with pytest.raises(ValueError, match="schema_version"):
        Metrics(**bad)


def test_metrics_validates_option_range() -> None:
    bad = _good_kwargs()
    bad["option"] = 5
    with pytest.raises(ValueError, match="option"):
        Metrics(**bad)


def test_metrics_validates_negative_rmse() -> None:
    bad = _good_kwargs()
    bad["rmse_clubhead_mm"] = -1.0
    with pytest.raises(ValueError, match=">= 0"):
        Metrics(**bad)


def test_metrics_validates_git_sha() -> None:
    bad = _good_kwargs()
    bad["git_commit"] = "ZZZ"
    with pytest.raises(ValueError, match="git_commit"):
        Metrics(**bad)


def test_to_json_round_trip() -> None:
    m = Metrics(**_good_kwargs())
    j = m.to_json()
    m2 = Metrics.from_json(j)
    assert m == m2
    # Canonical: sorted keys, no spaces.
    parsed = json.loads(j)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_to_csv_row_round_trip() -> None:
    m = Metrics(**_good_kwargs())
    row = m.to_csv_row()
    assert all(isinstance(v, str) for v in row.values())
    m2 = Metrics.from_csv_row(row)
    assert m == m2


def test_python_matlab_schema_equivalence() -> None:
    """Fixture written in MATLAB-equivalent canonical form deserialises."""
    text = FIXTURE.read_text(encoding="utf-8")
    m = Metrics.from_json(text)
    assert m.swing_id == "TW_ProV1"
    assert m.option == 1
    assert m.schema_version == SCHEMA_VERSION
    # Re-emit and verify semantic equivalence with the canonical fixture.
    assert json.loads(m.to_json()) == json.loads(text)


def test_legacy_result_struct_converted_to_metrics() -> None:
    legacy = {
        "swing_id": "TW_ProV1",
        "option": 2,
        "solver": "surrogate+fmincon",
        "rmse_clubhead": 4.2,
        "rmse_butt": 2.1,
        "rmse_orient": 3.0,
        "chs_impact": 110.0,
        "chs_meas": 111.5,
        "total_work": 295.0,
        "peak_power": 2700.0,
        "wall_clock": 12.3,
        "git_commit": "b" * 40,
        "timestamp": "2026-05-05T18:00:00Z",
    }
    m = legacy_struct_to_metrics(legacy)
    assert isinstance(m, Metrics)
    assert m.rmse_clubhead_mm == 4.2
    assert m.clubhead_speed_at_impact_mph == 110.0
    assert m.schema_version == SCHEMA_VERSION
    assert m.matlab_version == ""
