"""Tests for PhysicalTimePlayback engine (MV-04 #10480)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from rate_of_closure.simulation.playback_transport import (
    DEFAULT_SPEED,
    PLAYBACK_SPEEDS,
    SCRUB_STEPS,
    advance_playback,
    scrub_value,
    time_at_scrub,
)
from src.shared.python.motion_matching.playback import (
    InterpolatedPlaybackState,
    PhysicalTimePlayback,
)

FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "vendor"
    / "ud-tools"
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "playback_transport_golden_v1.json"
)


@pytest.fixture(scope="module")
def golden_fixture() -> dict[str, Any]:
    assert FIXTURE_PATH.exists(), f"Golden fixture missing at {FIXTURE_PATH}"
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_golden_fixture_constants_match_tools(golden_fixture: dict[str, Any]) -> None:
    assert golden_fixture["scrub_steps"] == SCRUB_STEPS
    assert golden_fixture["speeds"] == list(PLAYBACK_SPEEDS)
    assert golden_fixture["default_speed"] == DEFAULT_SPEED


def test_golden_fixture_scrub_and_advances(golden_fixture: dict[str, Any]) -> None:
    for case in golden_fixture["scrub_values"]:
        assert (
            scrub_value(float(case["time_s"]), float(case["duration_s"]))
            == case["value"]
        )

    for case in golden_fixture["scrub_times"]:
        assert time_at_scrub(
            int(case["value"]), float(case["duration_s"])
        ) == pytest.approx(case["time_s"], abs=1e-4)

    for case in golden_fixture["advances"]:
        step = advance_playback(
            float(case["time_s"]),
            float(case["elapsed_s"]),
            float(case["speed"]),
            float(case["duration_s"]),
        )
        assert step.time_s == pytest.approx(case["next_time_s"], abs=1e-4)
        assert step.finished == case["finished"]


def test_physical_time_playback_duration_and_stepping() -> None:
    times_s = np.array([0.0, 0.25, 0.75, 1.25, 1.814], dtype=np.float64)
    q = np.zeros((len(times_s), 6), dtype=np.float64)
    for i in range(len(times_s)):
        q[i, 0] = float(i) * 0.5

    playback = PhysicalTimePlayback(times_s=times_s, q=q)
    assert playback.duration_s == pytest.approx(1.814)

    # Step forward
    assert playback.step_time(0.0, 1) == pytest.approx(0.25)
    assert playback.step_time(0.25, 1) == pytest.approx(0.75)
    assert playback.step_time(1.814, 1) == pytest.approx(1.814)

    # Step backward
    assert playback.step_time(0.75, -1) == pytest.approx(0.25)
    assert playback.step_time(0.25, -1) == pytest.approx(0.0)
    assert playback.step_time(0.0, -1) == pytest.approx(0.0)


def test_physical_time_playback_authority_wall_clock_rates() -> None:
    """1.814s source plays in 1.814s at 1x and 7.256s at 0.25x."""
    duration_s = 1.813889
    times_s = np.linspace(0.0, duration_s, 654)
    q = np.zeros((654, 3))
    playback = PhysicalTimePlayback(times_s=times_s, q=q)

    # Simulate 1x playback over discrete frame steps with variable wall clock intervals
    t = 0.0
    elapsed_total_wall_s = 0.0
    dt_wall = 0.03333333333333333  # 30 fps GUI tick
    while t < duration_s:
        adv = playback.advance(t, dt_wall, speed=1.0)
        t = adv.time_s
        elapsed_total_wall_s += dt_wall
        if adv.finished:
            break
    # Total elapsed wall clock time should match source duration within one tick
    assert elapsed_total_wall_s == pytest.approx(duration_s, abs=dt_wall)

    # Simulate 0.25x playback: wall clock time is 4x longer (7.255556s)
    t = 0.0
    elapsed_total_wall_s = 0.0
    while t < duration_s:
        adv = playback.advance(t, dt_wall, speed=0.25)
        t = adv.time_s
        elapsed_total_wall_s += dt_wall
        if adv.finished:
            break
    expected_slow_duration = duration_s / 0.25
    assert elapsed_total_wall_s == pytest.approx(expected_slow_duration, abs=dt_wall)


def test_quaternion_slerp_interpolation() -> None:
    # 2 timestamps with 90 degree rotation around Z axis
    times_s = np.array([0.0, 1.0], dtype=np.float64)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity
    # 90 deg around Z: w=cos(pi/4), z=sin(pi/4)
    q1 = np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)])
    q = np.vstack([q0, q1])

    playback = PhysicalTimePlayback(times_s=times_s, q=q, quat_indices=((0, 4),))

    # At t = 0.5, rotation should be 45 deg around Z
    state = playback.interpolate(0.5)
    assert isinstance(state, InterpolatedPlaybackState)
    assert state.is_solver_state is False
    assert state.time_s == pytest.approx(0.5)

    interp_q = state.q[0:4]
    # Unit norm preserved
    assert np.linalg.norm(interp_q) == pytest.approx(1.0, abs=1e-7)
    expected_q = np.array([np.cos(np.pi / 8.0), 0.0, 0.0, np.sin(np.pi / 8.0)])
    np.testing.assert_allclose(interp_q, expected_q, atol=1e-6)

    # At exact knot t = 0.0, is_solver_state is True
    state_knot = playback.interpolate(0.0)
    assert state_knot.is_solver_state is True
    np.testing.assert_allclose(state_knot.q[0:4], q0, atol=1e-7)


def test_dropped_draw_does_not_slow_playback() -> None:
    """If a GUI frame is delayed by 100ms, playback jumps 100ms * speed."""
    times_s = np.linspace(0.0, 2.0, 100)
    q = np.zeros((100, 3))
    playback = PhysicalTimePlayback(times_s=times_s, q=q)

    adv = playback.advance(current_time_s=0.5, elapsed_wall_s=0.1, speed=1.0)
    assert adv.time_s == pytest.approx(0.6)
    assert adv.finished is False


def test_nonuniform_timestamps_and_event_times() -> None:
    times_s = np.array([0.0, 0.05, 0.12, 0.85, 1.814])
    q = np.zeros((5, 2))
    event_indices = {"Address": 0, "Top": 2, "Impact": 3, "Finish": 4}
    playback = PhysicalTimePlayback(times_s=times_s, q=q, event_indices=event_indices)
    assert playback.event_times["Address"] == pytest.approx(0.0)
    assert playback.event_times["Top"] == pytest.approx(0.12)
    assert playback.event_times["Impact"] == pytest.approx(0.85)
    assert playback.event_times["Finish"] == pytest.approx(1.814)

    # Query within nonuniform interval
    state = playback.interpolate(0.085)
    assert state.is_solver_state is False
    assert state.lower_index == 1
    assert 0.0 < state.fraction < 1.0
