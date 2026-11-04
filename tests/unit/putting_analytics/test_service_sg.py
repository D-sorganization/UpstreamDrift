"""Player-specific Bellman strokes gained on the current green (#14467, epic #14452).

Golden parity: the curve is the core policy-iteration solve
(:func:`.optimize.optimal_expected_putts`) for the player's own skill roll
model on the caller's green, and the putt's strokes gained is the core
:func:`.optimize.strokes_gained` against that curve, so the desktop, web and
studio surfaces that call the service show the core's values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.putting_analytics import optimize, service_sg
from src.shared.python.putting_analytics.expected_putts import (
    PuttAction,
    PuttOutcomes,
)
from src.shared.python.putting_analytics.roll_model import FT_M
from src.shared.python.putting_analytics.sampling import sample_execution, sample_read
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.putting_analytics.sg_curve import skill_model
from src.shared.python.putting_analytics.skill_roll_model import PlaneGreen

pytestmark = pytest.mark.unit

GRID = [1.0, 3.0, 6.0, 12.0]
GREEN = PlaneGreen(2.0, -90.0, 11.0)


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "distance_ft": 9.0,
        "grade_pct": 2.0,
        "aspect_deg": -90.0,
        "stimp": 11.0,
        "face_sd_deg": 1.0,
        "speed_cv": 0.06,
        "distances_ft": GRID,
        "paces_ft": [0.5, 2.0],
        "holed": False,
        "leave_ft": 2.5,
        "n": 16,
        "seed": 5,
    }
    return {**args, **overrides}


@pytest.fixture(scope="module")
def core_curve() -> Any:
    model = service_sg.SkillPuttModel(profile_from_args(_args()), GREEN)
    actions = [PuttAction(0.0, 0.5), PuttAction(0.0, 2.0)]
    return optimize.optimal_expected_putts(model, GRID, actions, 16, 5)


@pytest.fixture(scope="module")
def miss_report() -> dict[str, Any]:
    return service_sg.sg_study(_args())


def test_adapter_plays_the_core_skill_roll_at_the_action_pace() -> None:
    profile = profile_from_args(_args())
    outcomes = service_sg.SkillPuttModel(profile, GREEN).simulate(
        6.0, PuttAction(0.0, 2.0), 16, 7
    )
    core = skill_model(profile, GREEN, 6.0)
    played = type(core)(
        core.green, core.distance_ft, core.baseline, core.lattice, pace_ft_past=2.0
    )(sample_execution(profile.stroke, 16, 7), sample_read(profile.read, 16, 7))
    assert outcomes.holed.tolist() == played.holed.tolist()
    np.testing.assert_allclose(outcomes.leave_ft, played.leave_m / FT_M)


def test_adapter_rejects_an_aim_offset_because_the_player_plans_the_line() -> None:
    model = service_sg.SkillPuttModel(profile_from_args(_args()), GREEN)
    with pytest.raises(ValueError, match="aim"):
        model.simulate(6.0, PuttAction(1.0, 2.0), 4, 0)


def test_curve_is_the_core_bellman_solve_for_the_player(
    core_curve: Any, miss_report: dict[str, Any]
) -> None:
    curve, policy = core_curve
    assert miss_report["curve"]["distances_ft"] == GRID
    np.testing.assert_allclose(miss_report["curve"]["expected_putts"], curve.values)
    assert miss_report["curve"]["pace_ft_past"] == [a.pace_ft_past for a in policy]
    assert miss_report["expected_putts"] == pytest.approx(float(curve(9.0)))


def test_missed_putt_strokes_gained_is_the_core_value(
    core_curve: Any, miss_report: dict[str, Any]
) -> None:
    curve, _ = core_curve
    putt = PuttOutcomes(np.array([False]), np.array([2.5]))
    sg = optimize.strokes_gained(9.0, putt, curve, curve)
    assert miss_report["sg_player"] == pytest.approx(sg.sg_player)
    assert miss_report["sg_player"] == pytest.approx(
        float(curve(9.0)) - 1.0 - float(curve(2.5))
    )


def test_holed_putt_gains_its_expected_putts_less_one() -> None:
    holed = service_sg.sg_study(_args(paces_ft=[2.0], holed=True, leave_ft=7.0))
    assert holed["sg_player"] == pytest.approx(holed["expected_putts"] - 1.0)
    args = _args(paces_ft=[2.0])
    del args["holed"], args["leave_ft"]
    curve_only = service_sg.sg_study(args)
    assert curve_only["sg_player"] is None
    assert curve_only["curve"] == holed["curve"]
    assert curve_only["green"] == {"grade_pct": 2.0, "aspect_deg": -90.0, "stimp": 11.0}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"distance_ft": 20.0}, "grid"),
        ({"leave_ft": -1.0}, "leave_ft"),
        ({"holed": "yes"}, "holed"),
        ({"paces_ft": []}, "paces_ft"),
        ({"paces_ft": [12.0]}, "paces_ft"),
        ({"n": 1500, "paces_ft": [0.5, 1, 2, 3, 4, 5, 6, 7]}, "work"),
    ],
)
def test_bad_inputs_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        service_sg.sg_study(_args(**overrides))
