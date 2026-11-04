"""Tests for the append-only model-matching trend file (MATCH-4 #14446)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.model_matching_trend import append_record, render_delta, snapshot

pytestmark = pytest.mark.unit


def _entry(engine: str, marker: float | None, verified: bool = True) -> dict:
    return {
        "engine": engine,
        "marker_rms_mm": marker,
        "joint_angle_rms_deg": None,
        "verified": verified,
    }


def _ledger(mujoco: float | None, opensim: float | None) -> dict:
    return {"engines": [_entry("mujoco", mujoco), _entry("opensim", opensim, False)]}


def test_snapshot_keeps_metrics_and_verification_only() -> None:
    record = snapshot(_ledger(50.9, 32.5), "2026-11-04")
    assert record == {
        "date": "2026-11-04",
        "engines": {
            "mujoco": {
                "marker_rms_mm": 50.9,
                "joint_angle_rms_deg": None,
                "verified": True,
            },
            "opensim": {
                "marker_rms_mm": 32.5,
                "joint_angle_rms_deg": None,
                "verified": False,
            },
        },
    }


def test_append_record_is_append_only_and_dated(tmp_path: Path) -> None:
    trend = tmp_path / "trend.jsonl"
    append_record(trend, snapshot(_ledger(50.9, 32.5), "2026-11-04"))
    append_record(trend, snapshot(_ledger(40.0, 32.5), "2026-11-05"))
    lines = trend.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["date"] for line in lines] == ["2026-11-04", "2026-11-05"]
    before = trend.read_bytes()
    for stale in ("2026-11-05", "2026-11-01", "11/06/2026"):
        with pytest.raises(ValueError, match="date"):
            append_record(trend, snapshot(_ledger(1.0, 1.0), stale))
    assert trend.read_bytes() == before


def test_render_delta_reports_change_since_last_report(tmp_path: Path) -> None:
    trend = tmp_path / "trend.jsonl"
    append_record(trend, snapshot(_ledger(50.9, 32.5), "2026-11-04"))
    assert "No earlier record" in render_delta(trend)
    append_record(trend, snapshot(_ledger(40.4, None), "2026-11-05"))
    text = render_delta(trend)
    assert "2026-11-04 to 2026-11-05" in text
    assert "| mujoco | 50.9 | 40.4 | -10.5 |" in text
    assert "| opensim | 32.5 (UNVERIFIED) | not measured | n/a |" in text


def test_render_delta_rejects_empty_trend(tmp_path: Path) -> None:
    trend = tmp_path / "trend.jsonl"
    trend.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        render_delta(trend)


def test_committed_trend_matches_committed_ledger() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "docs" / "development" / "model_matching"
    ledger = json.loads((folder / "model_matching_status.json").read_text("utf-8"))
    last = json.loads(
        (folder / "model_matching_trend.jsonl").read_text("utf-8").splitlines()[-1]
    )
    assert last == snapshot(ledger, last["date"])
