"""Club head mass, inertia and COR reach the flight pipeline (#15443)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.shared.python.impact_parameters import flight_report as fr
from src.shared.python.impact_parameters.club_impact import (
    club_impact_for,
    club_impact_from_spec,
)
from src.shared.python.motion_matching.club_models import IRON_7
from src.shared.python.physics import swing_ball_flight_pipeline as sbp
from src.shared.python.physics.impact_model.types import ImpactParameters

pytestmark = pytest.mark.unit


def _state(**kw) -> sbp.SwingState:
    normal = np.array([np.cos(0.4), 0.0, np.sin(0.4)])
    return sbp.SwingState(
        clubhead_velocity=np.array([40.0, 0.0, -1.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        clubhead_loft_deg=float(np.degrees(0.4)),
        **kw,
    )


def _launch(state, params=None):
    result = sbp.SwingBallFlightPipeline(impact_params=params).run(state)
    return result.launch_conditions


def test_iron_takes_the_catalogue_head_mass() -> None:
    impact = club_impact_for("iron7")
    assert impact is not None
    assert impact.head_mass_kg == IRON_7.head_mass_kg == 0.268
    assert 0.0 < impact.head_moi_kg_m2 < 5e-3


def test_driver_keeps_the_pipeline_defaults() -> None:
    assert club_impact_for("driver") is None
    assert club_impact_from_spec({"name": "driver"}) is None


def test_unknown_club_warns_and_falls_back(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert club_impact_for("no-such-club") is None
    assert "no-such-club" in caplog.text
    assert club_impact_from_spec(None) is None


def test_heavier_head_gives_more_ball_speed() -> None:
    iron = club_impact_for("iron7")
    base = _launch(_state())
    heavy = _launch(
        _state(clubhead_mass=iron.head_mass_kg, clubhead_moi=iron.head_moi_kg_m2),
        ImpactParameters(cor=iron.cor),
    )
    assert heavy.velocity > base.velocity


def test_driver_launch_is_unchanged_to_1e_12() -> None:
    state = _state()
    pinned = _launch(state)
    again = _launch(state, ImpactParameters())  # explicit driver defaults
    assert again.velocity == pytest.approx(pinned.velocity, abs=1e-12)
    assert again.launch_angle == pytest.approx(pinned.launch_angle, abs=1e-12)
    assert again.spin_rate == pytest.approx(pinned.spin_rate, abs=1e-12)


def test_swing_state_carries_club_and_offset() -> None:
    from tests.unit.impact_parameters.test_flight_report import FRAME, _series

    iron = club_impact_for("iron7")
    state = fr.swing_state_from_series(_series(), 1, FRAME, iron, (0.004, -0.01))
    assert state.clubhead_mass == 0.268
    assert state.clubhead_moi == iron.head_moi_kg_m2
    assert list(state.impact_offset) == [0.004, -0.01]
    plain = fr.swing_state_from_series(_series(), 1, FRAME)
    assert plain.clubhead_mass == 0.200 and plain.impact_offset is None


def test_flight_for_impact_uses_the_club() -> None:
    from tests.unit.impact_parameters.test_flight_report import FRAME, _series

    plain = fr.flight_for_impact(_series(speed=40.0), 1, FRAME)
    iron = fr.flight_for_impact(
        _series(speed=40.0), 1, FRAME, club_impact=club_impact_for("iron7")
    )
    assert iron.ball_speed_mph > plain.ball_speed_mph
