"""Tee under a lifted ball (#15150)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.model_appearance.ball_strike import load_ball_strike_settings
from src.shared.python.model_appearance.ball_tee import (
    BallTeeSettings,
    default_ball_tee_settings,
    load_ball_tee_settings,
    tee_for_ball,
)

pytestmark = pytest.mark.unit

SETTINGS = default_ball_tee_settings()
#: Cup top may sit this far above the lowest point of the ball (the sagitta of
#: the cup rim on the ball sphere, 3.0 mm for the default cup).
TOP_TOLERANCE_M = 0.004


def _ball(lift_m: float, ground: float = 0.0) -> tuple[float, float, float]:
    return (0.3, -0.1, ground + BALL_RADIUS_M + lift_m)


def test_shipped_defaults() -> None:
    assert SETTINGS.enabled is True
    assert SETTINGS.min_gap_m == 0.01
    assert (SETTINGS.shaft_radius_m, SETTINGS.cup_radius_m) == (0.0025, 0.011)
    assert SETTINGS.cup_height_m == 0.006
    assert len(SETTINGS.rgba) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_gap_m", 0.0),
        ("min_gap_m", math.nan),
        ("shaft_radius_m", -0.001),
        ("cup_radius_m", 0.001),
        ("cup_height_m", math.inf),
        ("rgba", (1.0, 1.0, 1.0)),
        ("rgba", (1.0, 1.0, 1.0, 2.0)),
        ("enabled", 1),
    ],
)
def test_invalid_settings_raise_value_error(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SETTINGS, **{field: value})


def test_missing_key_raises_key_error(tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_ball_tee_settings(path)


def test_cup_must_be_smaller_than_ball() -> None:
    wide = dataclasses.replace(SETTINGS, cup_radius_m=BALL_RADIUS_M)
    with pytest.raises(ValueError):
        tee_for_ball(_ball(0.05), BALL_RADIUS_M, wide)
    with pytest.raises(ValueError):
        tee_for_ball((0.0, 0.0), BALL_RADIUS_M, SETTINGS)


def test_no_tee_when_ball_sits_on_ground() -> None:
    assert tee_for_ball(_ball(0.0), BALL_RADIUS_M, SETTINGS) is None
    assert tee_for_ball(_ball(0.0, ground=0.3), BALL_RADIUS_M, SETTINGS, 0.3) is None


def test_no_tee_below_threshold_but_tee_above() -> None:
    gap = SETTINGS.min_gap_m
    assert tee_for_ball(_ball(gap * 0.99), BALL_RADIUS_M, SETTINGS) is None
    assert tee_for_ball(_ball(gap * 1.01), BALL_RADIUS_M, SETTINGS) is not None


def test_disabled_tee_is_never_built() -> None:
    off = dataclasses.replace(SETTINGS, enabled=False)
    assert tee_for_ball(_ball(0.059), BALL_RADIUS_M, off) is None


@pytest.mark.parametrize("lift", [0.02, 0.059, 0.1016])
@pytest.mark.parametrize("ground", [0.0, 0.25])
def test_tee_stands_on_ground_and_touches_ball(lift: float, ground: float) -> None:
    centre = _ball(lift, ground)
    tee = tee_for_ball(centre, BALL_RADIUS_M, SETTINGS, ground)
    assert tee is not None and tee.shaft is not None
    assert tee.shaft.base_z_m == pytest.approx(ground)
    underside = centre[2] - BALL_RADIUS_M
    assert 0.0 <= tee.cup.top_z_m - underside <= TOP_TOLERANCE_M
    # cup rim lies on the ball sphere and the pieces are contiguous
    rim = math.hypot(SETTINGS.cup_radius_m, tee.cup.top_z_m - centre[2])
    assert rim == pytest.approx(BALL_RADIUS_M, abs=1e-9)
    assert tee.shaft.top_z_m == pytest.approx(tee.cup.base_z_m)
    assert tee.cup.xy_m == tee.shaft.xy_m == (centre[0], centre[1])


def test_driver_tee_limit_gets_a_tee() -> None:
    limit = load_ball_strike_settings().tee_limit_m("driver")
    assert tee_for_ball(_ball(limit), BALL_RADIUS_M, SETTINGS) is not None


def test_cup_is_cut_to_a_low_tee() -> None:
    low = dataclasses.replace(SETTINGS, min_gap_m=0.001, cup_height_m=0.05)
    tee = tee_for_ball(_ball(0.01), BALL_RADIUS_M, low)
    assert tee is not None and tee.shaft is None
    assert tee.cup.base_z_m == pytest.approx(0.0)


def test_tee_is_render_geometry_only() -> None:
    fields = {f.name for f in dataclasses.fields(BallTeeSettings)}
    tee = tee_for_ball(_ball(0.059), BALL_RADIUS_M, SETTINGS)
    assert tee is not None
    for name in ("mass", "collision", "contype", "friction"):
        assert name not in fields and not hasattr(tee, name)
