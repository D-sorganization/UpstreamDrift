"""Aim and pace optimisation under strokes-gained and must-make objectives."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_analytics.expected_putts import distance_seed
from src.shared.python.putting_analytics.optimize import (
    best_action,
    evaluate_actions,
    optimal_expected_putts,
    strokes_gained,
)

from .conftest import ToyBreakModel

N = 4000

pytestmark = pytest.mark.unit


@pytest.fixture
def solved(toy_model, grid, actions):
    return optimal_expected_putts(toy_model, grid, actions, n=N, seed=11)


def test_curve_is_monotone_and_at_least_one(solved) -> None:
    curve, _policy = solved
    assert np.all(curve.values >= 1.0)
    assert np.all(np.diff(curve.values) > -0.02)
    assert curve(25.0) > curve(3.0) + 0.3


def test_optimal_aim_compensates_for_break(solved, grid) -> None:
    _curve, policy = solved
    assert policy[list(grid).index(10)].aim_offset_deg == 2.0


def test_must_make_hits_harder_and_trades_strokes_for_makes(
    toy_model, solved, actions
) -> None:
    curve, _policy = solved
    evals = evaluate_actions(toy_model, 20.0, actions, curve, n=N, seed=5)
    sg, must = best_action(evals, "sg"), best_action(evals, "must_make")
    assert must.action.pace_ft_past > sg.action.pace_ft_past
    assert must.make_rate >= sg.make_rate
    assert must.expected_strokes >= sg.expected_strokes
    assert 0.0 < sg.make_rate_se < 0.02 and 0.0 < sg.expected_strokes_se < 0.02


def test_strokes_gained_against_own_curve_and_tour(solved, grid, actions) -> None:
    weak = ToyBreakModel(line_sd_deg=2.0, speed_cv=0.15)
    player, policy = optimal_expected_putts(weak, grid, actions, N, 11)
    tour, _ = solved
    index = list(grid).index(10)
    seed = distance_seed(11, index)
    best = strokes_gained(
        10.0, weak.simulate(10.0, policy[index], N, seed), player, tour
    )
    assert player(15.0) > tour(15.0) + 0.05  # worse skill, more putts
    assert best.sg_player == pytest.approx(0.0, abs=1e-9)
    assert best.sg_tour < -3 * best.sg_tour_se < 0
    for action in actions:
        other = weak.simulate(10.0, action, N, seed)
        assert strokes_gained(10.0, other, player, tour).sg_player <= 1e-9


def test_unknown_objective_is_rejected(toy_model, solved, actions) -> None:
    curve, _ = solved
    evals = evaluate_actions(toy_model, 5.0, actions[:2], curve, n=50, seed=1)
    with pytest.raises(ValueError):
        best_action(evals, "fastest")
