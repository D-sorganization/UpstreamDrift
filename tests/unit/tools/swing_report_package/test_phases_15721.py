"""Key swing phases for the swing report package (#15721, SRP-2).

Phases are found from segment directions: a segment is level when its unit
vector's vertical part changes sign. Which segment, and between which phases,
is a setting (``phase_settings.json``), never a literal in code.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.tools.swing_report_package.phases import (
    PHASE_SETTINGS_PATH,
    KeyPhase,
    detect_phases,
    load_phase_settings,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _synthetic(n: int = 401):
    """A swing whose shaft and lead arm turn through level at known times."""
    t = np.linspace(0.0, 2.0, n)
    # shaft vertical part: -0.8 at address, level at 0.5 s, up to the top
    # (1.0 s), level again on the way down at 1.25 s, down at impact 1.3 s,
    # level after impact at 1.6 s
    shaft_z = np.interp(
        t, [0.0, 0.5, 1.0, 1.25, 1.3, 1.6, 2.0], [-0.8, 0.0, 0.2, 0.0, -0.8, 0.0, 0.5]
    )
    arm_z = np.interp(t, [0.0, 1.0, 1.15, 1.3, 2.0], [-0.9, 0.5, 0.0, -0.9, 0.6])

    def segment(z):
        return np.stack([np.sqrt(1 - z**2), np.zeros_like(z), z], axis=1)

    grip = np.zeros((n, 3))
    shoulder = np.tile([0.0, 0.0, 1.4], (n, 1))
    head = grip + segment(shaft_z)
    points = {
        "grip": grip,
        "clubhead": head,
        "lead_shoulder": shoulder,
        "lead_wrist": shoulder + segment(arm_z) * 0.6,
    }
    return t, points


def test_the_default_phases_are_found_in_swing_order() -> None:
    t, points = _synthetic()
    phases = detect_phases(t, points, impact_time_s=1.3, top_time_s=1.0)
    by_id = {p.id: p for p in phases}
    assert [p.id for p in phases] == [
        "address",
        "shaft_parallel_back",
        "top",
        "lead_arm_parallel",
        "shaft_parallel_down",
        "impact",
        "follow_through",
    ]
    expected = {
        "address": 0.0,
        "shaft_parallel_back": 0.5,
        "top": 1.0,
        "lead_arm_parallel": 1.15,
        "shaft_parallel_down": 1.25,
        "impact": 1.3,
        "follow_through": 1.6,
    }
    for key, time_s in expected.items():
        assert by_id[key].time_s == pytest.approx(time_s, abs=0.006), key
        assert t[by_id[key].index] == pytest.approx(by_id[key].time_s, abs=0.006)
    assert all(isinstance(p.label, str) and p.label for p in phases)
    times = [p.time_s for p in phases]
    assert times == sorted(times)


def test_a_segment_that_never_levels_is_reported_not_guessed() -> None:
    t, points = _synthetic()
    points["clubhead"] = points["grip"] + np.array([0.0, 0.0, -1.0])
    with pytest.raises(ValueError, match="shaft_parallel_back"):
        detect_phases(t, points, impact_time_s=1.3, top_time_s=1.0)


def test_inputs_are_checked() -> None:
    t, points = _synthetic()
    with pytest.raises(ValueError, match="impact"):
        detect_phases(t, points, impact_time_s=5.0, top_time_s=1.0)
    with pytest.raises(ValueError, match="lead_wrist"):
        detect_phases(
            t,
            {k: v for k, v in points.items() if k != "lead_wrist"},
            impact_time_s=1.3,
            top_time_s=1.0,
        )


def test_the_phases_come_from_the_settings(tmp_path) -> None:
    doc = json.loads(PHASE_SETTINGS_PATH.read_text(encoding="utf-8"))
    doc["phases"] = [p for p in doc["phases"] if p["id"] in ("address", "impact")]
    path = tmp_path / "phases.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    t, points = _synthetic()
    settings = load_phase_settings(path)
    phases = detect_phases(t, points, 1.3, 1.0, settings)
    assert [p.id for p in phases] == ["address", "impact"]
    doc["phases"] = [{"id": "x", "label": "X", "rule": "teleport"}]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="rule"):
        load_phase_settings(path)


def test_a_key_phase_round_trips_to_a_dict() -> None:
    phase = KeyPhase("top", "Top of Backswing", 200, 1.0)
    assert KeyPhase(**phase.to_dict()) == phase
