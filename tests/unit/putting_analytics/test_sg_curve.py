"""Simulated putting curve for the strokes-gained optimizer (#14177)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.putting_analytics import sg_curve
from src.shared.python.putting_analytics.skill import (
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)
from src.shared.python.sg_optimizer.course.conditions import GreenModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PuttingSkill
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    putting_provider,
)

pytestmark = pytest.mark.unit

MID = PuttingSkillProfile(
    "mid",
    StrokeDispersion(face_sd_deg=1.5, path_sd_deg=2.0, speed_cv=0.08),
    ReadDispersion(grade_rel_sd=0.25, aspect_sd_deg=5.0, stimp_rel_sd=0.08),
)
GRID = (3.0, 10.0, 30.0)


@pytest.fixture(scope="module")
def curve() -> sg_curve.SimulatedPuttingCurve:
    return sg_curve.simulate_putting_curve(MID, GRID, n=200, seed=2)


def test_make_rate_falls_and_leave_grows_with_distance(
    curve: sg_curve.SimulatedPuttingCurve,
) -> None:
    assert curve.distances_ft == GRID and curve.samples_per_distance == 200
    assert curve.make_rate[0] > curve.make_rate[1] > curve.make_rate[2]
    assert curve.leave_scale_ft[2] > curve.leave_scale_ft[1] > 0.0
    assert all(0.0 < s < 2.0 for s in curve.leave_sigma_log)


def test_curve_feeds_the_simulation_provider(
    curve: sg_curve.SimulatedPuttingCurve,
) -> None:
    provider = putting_provider("simulation", curve)
    greens = GreenModel(stimp=10.0)
    p = provider.make_probability(10.0, PuttingSkill(), greens)
    assert p == pytest.approx(curve.make_rate[1])


def test_curve_is_seeded(curve: sg_curve.SimulatedPuttingCurve) -> None:
    assert sg_curve.simulate_putting_curve(MID, GRID, n=200, seed=2) == curve


def test_distances_without_enough_misses_borrow_a_neighbour_fit() -> None:
    grid = np.array([2.0, 4.0, 8.0])
    filled = sg_curve._fill(grid, [None, (1.0, 0.5), (2.0, 0.7)])
    np.testing.assert_allclose(filled, [[1.0, 0.5], [1.0, 0.5], [2.0, 0.7]])
    with pytest.raises(ValueError, match="misses"):
        sg_curve._fill(grid, [None, None, None])


def test_leave_fit_needs_enough_misses() -> None:
    assert sg_curve._leave_fit(np.ones(sg_curve.MIN_MISSES - 1)) is None
    scale, sigma = sg_curve._leave_fit(np.full(sg_curve.MIN_MISSES, 2.0)) or (0, 1)
    assert (scale, sigma) == pytest.approx((2.0, 0.0))


def test_a_golfer_who_never_misses_has_no_leave_to_fit() -> None:
    perfect = PuttingSkillProfile("perfect", StrokeDispersion(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="misses"):
        sg_curve.simulate_putting_curve(perfect, (2.0, 4.0), n=40)
