"""Ball placed where the club face strikes it (#15150): synthetic swings only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.model_appearance.ball_strike import (
    SETTINGS_PATH,
    BallStrikeSettings,
    load_ball_strike_settings,
    strike_ball_centre,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

RATE_HZ = 360.0  # one sample every 125 mm at 45 m/s: contact falls between samples
OPEN = math.radians(12.0)
NORMAL = np.array([-math.sin(OPEN), -math.cos(OPEN), 0.0])  # open face, target -y


def _pass(height_m: float = 0.060):
    """Face centre crossing x = 0.1 along -y at ``height_m`` (times, centres, normals)."""
    t = (np.arange(12) + 0.37) / RATE_HZ
    centre = np.column_stack([np.full(12, 0.1), 0.75 - 45.0 * t, np.full(12, height_m)])
    return t, centre, np.tile(NORMAL, (12, 1))


def _place(nominal, *, tee=0.1016, shift=0.15, **kw):
    t, centre, normal = _pass(**kw)
    return strike_ball_centre(
        t, centre, normal, nominal, max_tee_height_m=tee, max_shift_m=shift
    )


def test_the_face_centre_meets_the_ball_between_samples() -> None:
    nominal = np.array([0.1, 0.0, BALL_RADIUS_M])  # on the ground, 38 mm too low
    placed = _place(nominal)
    t, centre, _ = _pass()
    # the closest point of the path to the nominal ball is on the line y = 0
    contact = np.array([0.1, 0.0, 0.060])
    assert np.asarray(placed.centre_m) == pytest.approx(
        contact + BALL_RADIUS_M * NORMAL, abs=1e-9
    )
    assert t[0] < placed.contact_time_s < t[-1]
    assert not np.any(np.isclose(t, placed.contact_time_s, atol=1e-5))  # sub-sample
    assert np.interp(placed.contact_time_s, t, centre[:, 1]) == pytest.approx(0.0)
    assert placed.height_limited_m == 0.0
    assert placed.tee_height_m == pytest.approx(0.060 - BALL_RADIUS_M)
    assert placed.shift_m == pytest.approx(np.asarray(placed.centre_m) - nominal)


def test_a_ball_cannot_sit_above_the_tee_limit_or_below_the_ground() -> None:
    nominal = np.array([0.1, 0.0, BALL_RADIUS_M])
    high = _place(nominal, tee=0.020, height_m=0.090)
    assert high.centre_m[2] == pytest.approx(BALL_RADIUS_M + 0.020)
    assert high.height_limited_m == pytest.approx(BALL_RADIUS_M + 0.020 - 0.090)
    low = _place(nominal, tee=0.0, height_m=0.005)  # an iron: no tee, face low
    assert low.centre_m[2] == pytest.approx(BALL_RADIUS_M)
    assert low.tee_height_m == 0.0 and low.height_limited_m > 0.0


def test_a_swing_that_misses_by_more_than_the_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="more than the 0.050 m limit"):
        _place(np.array([0.1, 0.0, BALL_RADIUS_M]), shift=0.050, height_m=0.120)


def test_bad_limits_and_paths_are_rejected() -> None:
    t, centre, normal = _pass()
    at = [0.1, 0.0, 0.02]
    ok = {"max_tee_height_m": 0.05, "max_shift_m": 0.1}
    for bad in (
        {"max_tee_height_m": -0.01, "max_shift_m": 0.1},
        {"max_tee_height_m": 0.05, "max_shift_m": 0.0},
        {**ok, "radius_m": 0.0},
        {**ok, "ground_height_m": math.nan},
    ):
        with pytest.raises(ValueError):
            strike_ball_centre(t, centre, normal, at, **bad)
    with pytest.raises(ValueError, match="at least 2"):
        strike_ball_centre(t[:1], centre[:1], normal[:1], at, **ok)
    with pytest.raises(ValueError, match="shape"):
        strike_ball_centre(t, centre[:, :2], normal, at, **ok)
    with pytest.raises(ValueError, match="finite"):
        strike_ball_centre(t, centre, normal, [0.1, math.nan, 0.02], **ok)
    with pytest.raises(ValueError, match="nonzero"):
        strike_ball_centre(t, centre, 0.0 * normal, at, **ok)


def test_every_setting_has_a_source_note_and_loads() -> None:
    raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}
    settings = load_ball_strike_settings()
    assert settings.placement in ("address", "strike")
    assert settings.tee_limit_m("driver") > settings.tee_limit_m("iron7") == 0.0
    assert settings.tee_limit_m("driver") <= 0.1016  # no tee is longer than 4 in
    assert settings.tee_limit_m("unknown club") == settings.tee_limit_m("default")


def test_settings_are_validated() -> None:
    for match, args in (
        ("placement", ("teleport", 0.02, 0.15, {"default": 0.0})),
        ("search_window_s", ("strike", 0.0, 0.15, {"default": 0.0})),
        ("default", ("strike", 0.02, 0.15, {"driver": 0.05})),
        ("max_tee_height_m", ("strike", 0.02, 0.15, {"default": -1.0})),
    ):
        with pytest.raises(ValueError, match=match):
            BallStrikeSettings(*args)
