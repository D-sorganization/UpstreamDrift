"""Ball-flight reporting for the showpiece render (SHOW-C7, #14563).

Reporting layer only: impact state -> club/ball impact solver -> aerodynamic
flight.  Carry only: the repository has no bounce/roll model, so ``total`` is
reported unavailable with the reason, never invented.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries, TargetFrame
from src.shared.python.impact_parameters.conditions import ImpactConditions
from src.shared.python.impact_parameters.flight_report import (
    FlightSummary,
    flight_for_impact,
    overlay_lines,
    swing_state_from_series,
)

pytestmark = pytest.mark.unit
FRAME = TargetFrame()  # Z-up, target along -Y, golfer's left = +X


def _series(
    speed: float = 49.0,
    loft_deg: float = 12.0,
    face_open_deg: float = 0.0,
    path_deg: float = 0.0,
):
    """Three samples, clubhead moving down-range (-Y) with a given face."""
    loft, op = math.radians(loft_deg), math.radians(face_open_deg)
    # face normal: down-range, tilted up by loft, rotated open (toward golfer's right = -X)
    normal = np.array(
        [-math.cos(loft) * math.sin(op), -math.cos(loft) * math.cos(op), math.sin(loft)]
    )
    t = np.array([0.0, 0.001, 0.002])
    pos = np.array([[0.0, 0.1 - speed * s, 0.0] for s in t])
    pa = math.radians(path_deg)  # + = in-to-out = toward golfer's right (-X)
    vel = np.tile([-speed * math.sin(pa), -speed * math.cos(pa), 0.0], (3, 1))
    return ClubheadSeries(t, pos, vel, face_normal=np.tile(normal, (3, 1)))


def test_swing_state_maps_world_to_down_range_left_up():
    state = swing_state_from_series(_series(speed=49.0, loft_deg=12.0), 1, FRAME)
    # pipeline frame: x down-range, y left, z up
    assert state.clubhead_velocity == pytest.approx([49.0, 0.0, 0.0], abs=1e-9)
    assert state.clubhead_orientation[0] > 0.9
    assert state.clubhead_orientation[2] == pytest.approx(math.sin(math.radians(12)))
    assert state.clubhead_loft_deg == pytest.approx(12.0, abs=1e-6)


def test_swing_state_rejects_unobservable_face():
    s = _series()
    blind = ClubheadSeries(
        s.times_s,
        s.face_center_m,
        s.velocity_mps,
        face_unobservable_reason="axial rotation unobservable",
    )
    with pytest.raises(ValueError, match="face"):
        swing_state_from_series(blind, 1, FRAME)


def test_flight_for_impact_is_plausible_and_carry_only():
    summary = flight_for_impact(_series(speed=49.0), 1, FRAME)
    assert isinstance(summary, FlightSummary)
    assert 150.0 < summary.carry_m < 300.0
    assert 5.0 < summary.launch_deg < 25.0
    assert summary.ball_speed_mph > 100.0
    assert summary.total_m is None and "bounce" in summary.total_unavailable


def test_open_face_sends_the_ball_right_of_target():
    right = flight_for_impact(_series(face_open_deg=5.0), 1, FRAME)
    left = flight_for_impact(_series(face_open_deg=-5.0), 1, FRAME)
    assert right.lateral_m > 5.0 > -5.0 > left.lateral_m  # + = golfer's right


def test_overlay_lines_state_every_condition_and_the_gap():
    summary = flight_for_impact(_series(), 1, FRAME)
    cond = ImpactConditions(110.1, -4.2, 21.0, 6.5)
    lines = overlay_lines("Measured 110 mph swing (not the showpiece)", cond, summary)
    text = "\n".join(lines)
    for needle in (
        "Measured 110 mph",
        "110.1",
        "21.0",
        "6.5",
        "-4.2",
        "carry",
        "total",
    ):
        assert needle in text
    assert "n/a" in text


def test_square_driver_strike_is_driver_like():
    """Sanity anchor: 110 mph clubhead, 10.5 deg loft, square: tour-like numbers."""
    summary = flight_for_impact(_series(speed=49.2, loft_deg=10.5), 1, FRAME)
    assert 150.0 < summary.ball_speed_mph < 175.0
    assert 1500.0 < summary.spin_rpm < 4000.0
    assert 200.0 < summary.carry_m < 260.0
    assert abs(summary.lateral_m) < 1.0


def test_extreme_slice_outside_model_validity_is_flagged_not_presented():
    """A far-outside case (face +35 open, path +6.5 deg) is not a result."""
    bad = flight_for_impact(
        _series(speed=49.2, loft_deg=12.0, face_open_deg=35.0, path_deg=6.5), 1, FRAME
    )
    assert not bad.valid and bad.invalid_reason
    cond = ImpactConditions(110.1, -4.2, 21.0, 6.5)
    text = "\n".join(overlay_lines("t", cond, bad))
    assert "INVALID" in text and "carry n/a" in text
    good = flight_for_impact(_series(), 1, FRAME)
    assert good.valid and good.invalid_reason == ""
