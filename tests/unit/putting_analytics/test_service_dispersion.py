"""Dispersion batches and make-% spread on the current green (#14459, epic #14452).

Golden parity: every number the service returns is the shared skill roll
model's own answer for the same player, green and seed, so the desktop,
web and studio surfaces that call it show the core's values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.putting_analytics import service_dispersion
from src.shared.python.putting_analytics.roll_model import FT_M
from src.shared.python.putting_analytics.sampling import sample_execution, sample_read
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.putting_analytics.sg_curve import (
    simulated_make_rates,
    skill_model,
)
from src.shared.python.putting_analytics.skill_roll_model import PlaneGreen

pytestmark = pytest.mark.unit


def _args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "distance_ft": 12.0,
        "grade_pct": 2.0,
        "aspect_deg": 90.0,
        "stimp": 11.0,
        "face_sd_deg": 1.2,
        "speed_cv": 0.07,
        "n": 40,
        "seed": 3,
    }
    return {**args, **overrides}


def test_batch_matches_the_core_roll_for_the_same_player_and_green() -> None:
    report = service_dispersion.dispersion_study(_args())
    profile = profile_from_args(_args())
    model = skill_model(profile, PlaneGreen(2.0, 90.0, 11.0), 12.0)
    execution = sample_execution(profile.stroke, 40, 3)
    rolled = model.roll(execution, sample_read(profile.read, 40, 3))
    hole_x_m = 12.0 * FT_M
    np.testing.assert_allclose(report["rest_x_ft"], (rolled.rest_x_m - hole_x_m) / FT_M)
    np.testing.assert_allclose(report["rest_y_ft"], rolled.rest_y_m / FT_M)
    assert report["holed"] == rolled.holed.tolist()
    assert report["make_rate"] == pytest.approx(float(rolled.holed.mean()))
    assert report["n"] == 40


def test_slope_moves_the_pattern_off_the_line() -> None:
    flat = service_dispersion.dispersion_study(_args(grade_pct=0.0))
    sloped = service_dispersion.dispersion_study(_args(grade_pct=4.0))
    # Same draws, same planned-on-the-read line: a side slope spreads the
    # misses differently, so the patterns differ.
    assert flat["rest_y_ft"] != sloped["rest_y_ft"]


def test_batch_runs_the_studio_sizes_without_the_old_100_putt_cap() -> None:
    report = service_dispersion.dispersion_study(_args(n=1000))
    assert report["n"] == 1000
    assert len(report["rest_x_ft"]) == 1000


def test_spread_matches_the_core_make_rates_on_the_current_green() -> None:
    args = _args(distances_ft=[3.0, 10.0, 25.0], n=60)
    report = service_dispersion.make_spread_study(args)
    expected = simulated_make_rates(
        profile_from_args(args),
        [3.0, 10.0, 25.0],
        n=60,
        seed=3,
        green=PlaneGreen(2.0, 90.0, 11.0),
    )
    assert report["distances_ft"] == [3.0, 10.0, 25.0]
    assert report["make_rate"] == list(expected)
    assert report["samples_per_distance"] == 60


def test_default_spread_covers_0_to_45_ft() -> None:
    report = service_dispersion.make_spread_study({"n": 10, "seed": 1})
    assert report["distances_ft"] == list(service_dispersion.DEFAULT_SPREAD_GRID_FT)
    assert report["distances_ft"][-1] == 45.0
    assert report["make_rate"][0] >= report["make_rate"][-1]


@pytest.mark.parametrize(
    ("study", "overrides", "message"),
    [
        ("dispersion_study", {"n": 0}, "'n'"),
        ("dispersion_study", {"n": service_dispersion.MAX_DISPERSION_N + 1}, "'n'"),
        ("dispersion_study", {"distance_ft": 0.0}, "distance_ft"),
        ("dispersion_study", {"stimp": 30.0}, "stimp"),
        ("dispersion_study", {"grade_pct": 12.0}, "grade_pct"),
        ("make_spread_study", {"distances_ft": [3.0]}, "at least 2"),
        ("make_spread_study", {"n": 5000, "distances_ft": [3.0, 9.0]}, "'n'"),
    ],
)
def test_rejects_bad_arguments(
    study: str, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        getattr(service_dispersion, study)(_args(**overrides))
