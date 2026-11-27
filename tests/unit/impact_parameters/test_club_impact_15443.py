"""Club head mass, inertia and COR reach the flight pipeline (#15443)."""

from __future__ import annotations

import json
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
        ImpactParameters(cor=0.83),  # driver COR: isolate the head-mass effect
    )
    assert heavy.velocity > base.velocity


# Penner (2001) Am. J. Phys. 69, 563, Eq. (5): e = 0.86 - 0.0029 v_ci cos(theta),
# a linear fit to Lieberman and Johnson's balls fired at a stationary plate.
_PENNER_INTERCEPT, _PENNER_SLOPE_S_PER_M = 0.86, 0.0029


@pytest.mark.parametrize("normal_speed_mps", [25.0, 37.0, 50.0])
def test_iron_cor_follows_the_sourced_speed_law(normal_speed_mps: float) -> None:
    iron = club_impact_for("iron7")
    expected = _PENNER_INTERCEPT - _PENNER_SLOPE_S_PER_M * normal_speed_mps
    assert iron.cor_at(normal_speed_mps) == pytest.approx(expected, abs=1e-12)


def test_iron_cor_settings_cite_the_source() -> None:
    from src.shared.python.impact_parameters.club_impact import SETTINGS_PATH

    cfg = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    law = cfg["cor_by_club"]["iron7"]
    assert law == {"intercept": 0.86, "slope_s_per_m": 0.0029}
    source = cfg["iron_cor_source"]
    for key in ("citation", "location", "fitted_data", "validity", "applicability"):
        assert source[key].strip(), key
    assert "Am. J. Phys. 69" in source["citation"]


def test_cor_at_rejects_negative_speed() -> None:
    with pytest.raises(ValueError):
        club_impact_for("iron7").cor_at(-1.0)


def test_flight_for_impact_evaluates_the_iron_cor_at_the_normal_speed(
    monkeypatch,
) -> None:
    from tests.unit.impact_parameters.test_flight_report import FRAME, _series

    seen: list[float] = []
    real = fr.ImpactParameters

    def spy(**kw):
        seen.append(kw["cor"])
        return real(**kw)

    monkeypatch.setattr(fr, "ImpactParameters", spy)
    series = _series(speed=40.0, loft_deg=34.0)
    iron = club_impact_for("iron7")
    fr.flight_for_impact(series, 1, FRAME, club="iron7", club_impact=iron)
    state = fr.swing_state_from_series(series, 1, FRAME, iron)
    normal = state.clubhead_orientation / np.linalg.norm(state.clubhead_orientation)
    v_n = float(np.dot(state.clubhead_velocity, normal))
    assert 30.0 < v_n < 40.0
    assert seen == [pytest.approx(_PENNER_INTERCEPT - _PENNER_SLOPE_S_PER_M * v_n)]


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


def test_rigid_gear_heads_are_settings_per_club():
    """Gear-law head inertia and CG depth come from the settings file (#15443)."""
    from src.shared.python.impact_parameters.club_impact import rigid_gear_head_for

    driver = rigid_gear_head_for("driver")
    iron = rigid_gear_head_for("iron7")
    assert driver.inertia_diagonal_kg_m2 == (0.00025, 0.00040, 0.00045)
    assert driver.cg_depth_m == pytest.approx(0.035)
    assert iron.cg_depth_m == pytest.approx(0.010)
    assert rigid_gear_head_for(None) == driver
    assert rigid_gear_head_for("no-such-club") == driver
