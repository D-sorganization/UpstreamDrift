"""Plan on the perceived green, execute on the true green (#14173)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
)
from src.shared.python.putting_analytics.optimize import best_action, evaluate_actions
from src.shared.python.putting_analytics.perceived_green import (
    GreenRead,
    PerceivedGreenPlanner,
    perceive,
    read_error_cost,
)
from src.shared.python.putting_analytics.roll_model import BatchRollModel
from src.shared.python.putting_dynamics.surfaces import HeightField, SurfaceSpec

DISTANCE_FT = 10.0
FALLS_TOWARD_MINUS_Y = 270.0  # downhill to the right of a putt along +x
# Test-only miss pricing, shaped like a putting curve; not reference data.
BASELINE = ExpectedPuttsCurve(
    np.array([1.0, 3.0, 6.0, 10.0, 20.0, 40.0]),
    np.array([1.0, 1.1, 1.4, 1.6, 1.85, 2.1]),
)
CANDIDATES = tuple(
    PuttAction(float(aim), pace)
    for aim in np.arange(0.0, 12.01, 0.5)
    for pace in (1.0, 2.0)
)


def _noise(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    return rng.normal(0.0, 0.5, n), np.exp(rng.normal(0.0, 0.03, n))


def _side_hill(grade: float) -> SurfaceSpec:
    flat = SurfaceSpec.flat_uniform(10.0)
    return dataclasses.replace(
        flat, height=HeightField.planar(grade, FALLS_TOWARD_MINUS_Y)
    )


def _planner(grade: float, n: int = 24) -> PerceivedGreenPlanner:
    return PerceivedGreenPlanner(_side_hill(grade), CANDIDATES, BASELINE, n, _noise)


def test_perceive_scales_slope_and_speed() -> None:
    surface = _side_hill(2.0)
    assert perceive(surface, GreenRead()) is surface
    seen = perceive(surface, GreenRead(slope_scale=0.5, stimp_ratio=2.0))
    assert seen.height.gradient(0.3, -0.2)[1] == pytest.approx(
        0.5 * surface.height.gradient(0.3, -0.2)[1], rel=1e-12
    )
    assert seen.friction.mu_roll0 == pytest.approx(surface.friction.mu_roll0 / 2.0)


def test_roll_many_matches_rolling_each_action_alone() -> None:
    model = BatchRollModel(_side_hill(2.0), _noise)
    actions = CANDIDATES[::9]
    together = model.roll_many(DISTANCE_FT, actions, 6, seed=3)
    for k, action in enumerate(actions):
        alone = model.roll(DISTANCE_FT, action, 6, seed=3)
        np.testing.assert_array_equal(
            together.rest_y_m[k * 6 : (k + 1) * 6], alone.rest_y_m
        )


def test_zero_read_error_reproduces_the_true_surface_optimum() -> None:
    few = CANDIDATES[::4]  # the unbatched reference path is slow
    planner = PerceivedGreenPlanner(_side_hill(2.0), few, BASELINE, 24, _noise)
    direct = evaluate_actions(planner.true_model, DISTANCE_FT, few, BASELINE, 24, 0)
    assert planner.plan(DISTANCE_FT, GreenRead()) == best_action(direct)


def test_slope_under_read_misses_low_and_more_so_with_more_break() -> None:
    low_side_shift = []
    for grade in (1.0, 3.0):
        planner = _planner(grade)
        mean_miss_y = []
        for read in (GreenRead(), GreenRead(slope_scale=0.5)):
            action = planner.plan(DISTANCE_FT, read).action
            result = planner.true_model.roll(DISTANCE_FT, action, 200, seed=11)
            missed_y = result.rest_y_m[~result.holed]
            if read.slope_scale < 1.0:
                assert np.mean(missed_y < 0.0) > 0.9  # the low (downhill) side
            mean_miss_y.append(missed_y.mean())
        low_side_shift.append(mean_miss_y[0] - mean_miss_y[1])
    assert 0.0 < low_side_shift[0] < low_side_shift[1]


def test_read_error_cost_is_zero_for_perfect_reads_and_positive_for_misreads() -> None:
    planner = _planner(1.0)
    exact = read_error_cost(planner, DISTANCE_FT, [GreenRead()] * 2, seed=5)
    assert exact.cost == 0.0 and exact.cost_se == 0.0
    misread = read_error_cost(
        planner, DISTANCE_FT, [GreenRead(0.5), GreenRead(0.8)], seed=5
    )
    assert misread.cost > 3.0 * misread.cost_se
    assert misread.expected_strokes == pytest.approx(
        misread.perfect_read_strokes + misread.cost
    )


def test_plans_are_cached_per_read_grid_cell() -> None:
    planner = _planner(1.0, n=4)
    first = planner.plan(DISTANCE_FT, GreenRead(slope_scale=0.81))
    assert planner.plan(DISTANCE_FT, GreenRead(slope_scale=0.79)) is first
    assert GreenRead(0.81).snapped(20) == GreenRead(0.8)


def test_contract_errors() -> None:
    with pytest.raises(ValueError):
        GreenRead(slope_scale=0.0)
    with pytest.raises(ValueError):
        GreenRead(stimp_ratio=float("nan"))
    with pytest.raises(ValueError):
        PerceivedGreenPlanner(_side_hill(1.0), (), BASELINE, 8)
    with pytest.raises(ValueError):
        PerceivedGreenPlanner(_side_hill(1.0), CANDIDATES, BASELINE, 8, objective="x")
    with pytest.raises(ValueError):
        read_error_cost(_planner(1.0, n=4), DISTANCE_FT, [], seed=0)
