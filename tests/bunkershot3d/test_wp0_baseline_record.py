"""Keep the WP-0 baseline record explicit about evidence and gaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "bunkershot3d"
    / "baselines"
    / "wp0_f0_reference.json"
)


def _record() -> dict[str, Any]:
    """Load the committed record without silently supplying missing fields."""
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_wp0_record_pins_the_issue_reported_baseline() -> None:
    record = _record()

    assert record["source_issue"] == 15805
    assert record["source_revision"] == "ffa19cf8ccc36b3ae1b492ef32b123ecf14e557a"
    assert record["preset"] == "acushnet_example_1"
    assert record["sand_preset"] == "default_sand"
    assert record["swing_speed_m_s"] == 25.0
    assert record["solver"] == {
        "package": "upstream-physics",
        "version": "2.1.3",
        "revision": 3,
    }
    assert record["reported_summary"] == {
        "verdict_status": "beyond_validation",
        "fidelity_tier": "F0",
        "refused": False,
        "peak_force_n": 300.19,
        "impulse_n_s": 2.22,
        "animation_frames": 60,
        "carry_m": 0.578,
        "apex_m": 0.255,
        "total_distance_m": 2.645,
    }
    assert record["summary_evidence"] == "issue_report_only"


def test_missing_render_capture_keeps_wp0_blocked() -> None:
    record = _record()

    assert record["status"] == "blocked_missing_render_capture"
    assert record["camera"] is None
    assert record["video"] is None
    assert record["blockers"] == [
        "original camera extrinsics and bounds are unavailable",
        "original MP4 bytes and container metadata are unavailable",
        "a revision-3 upstream-physics runtime is required to replay the summary",
    ]
    assert record["local_replay"]["installed_solver_revision"] == 0
    assert record["local_replay"]["status"] == "rejected_wrong_solver_revision"
    assert record["local_replay"]["reported_by_runtime"] == {
        "peak_force_n": 289.41713464673734,
        "impulse_n_s": 1.7644172666440627,
        "animation_frames": 60,
        "carry_m": None,
    }
