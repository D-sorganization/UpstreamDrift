"""JSON service for dispersion batches on the current green (#14459, epic #14452).

Two studies for the putting surfaces (desktop simulator, web page, studio):

* :func:`dispersion_study` -- ``n`` putts at one distance by the player
  under study, on a planar green with the caller's grade, fall direction and
  stimp: where every ball stopped (relative to the hole) and the make rate.
  It replaces ``/putting-green/scatter``'s 100-putt, flat-green, seed-0 batch
  for the studio's 40/100/1000 batches.
* :func:`make_spread_study` -- the player's make rate by distance over 0-45
  ft on that same green (the studio's "0-45 ft spread" button).

Both run the shared skill roll model (:func:`.sg_curve.skill_model`): lines are
planned on the green as read and rolled on the true green, with the player's
stroke and read dispersion from :func:`.service_variance.profile_from_args`.
Arguments are untrusted tool input, checked with :func:`.service._check`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.shared.python.putting_analytics.roll_model import FT_M
from src.shared.python.putting_analytics.sampling import sample_execution, sample_read
from src.shared.python.putting_analytics.service import (
    MAX_DISTANCE_FT,
    MAX_GRADE_PCT,
    STIMP_RANGE,
    _check,
    _number,
)
from src.shared.python.putting_analytics.service_curve import (
    MAX_CURVE_ROLLOUTS,
    _seed,
    distances_from_args,
)
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.putting_analytics.sg_curve import (
    simulated_make_rates,
    skill_model,
)
from src.shared.python.putting_analytics.skill_roll_model import PlaneGreen

MAX_DISPERSION_N = 2000  # the studio's largest batch
DEFAULT_DISPERSION_N = 100
DEFAULT_SPREAD_GRID_FT = tuple(float(d) for d in range(3, 46, 3))
DEFAULT_SPREAD_N = 150
MAX_SPREAD_N = 2000


def green_from_args(args: Mapping[str, Any]) -> PlaneGreen:
    """Planar green from ``grade_pct``, ``aspect_deg`` (fall, CCW of the line), ``stimp``."""
    grade = _number(args, "grade_pct", 0.0)
    _check(0.0 <= grade <= MAX_GRADE_PCT, f"grade_pct must be in [0, {MAX_GRADE_PCT}]")
    aspect = _number(args, "aspect_deg", 0.0)
    _check(-180.0 <= aspect <= 180.0, "aspect_deg must be in [-180, 180]")
    stimp = _number(args, "stimp", 10.0)
    _check(STIMP_RANGE[0] <= stimp <= STIMP_RANGE[1], f"stimp must be in {STIMP_RANGE}")
    return PlaneGreen(grade, aspect if grade else 0.0, stimp)


def _whole(args: Mapping[str, Any], key: str, default: int, high: int) -> int:
    value = _number(args, key, default)
    _check(value == int(value), f"'{key}' must be a whole number")
    _check(1 <= value <= high, f"'{key}' must be in [1, {high}]")
    return int(value)


def dispersion_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """Rest positions and make rate of ``n`` putts at ``distance_ft``.

    Returns ``rest_x_ft`` (past the hole is positive) and ``rest_y_ft`` (left
    of the line is positive) relative to the hole centre, ``holed`` per putt,
    ``make_rate``, ``mean_leave_ft`` over the misses (``None`` if all holed),
    ``n`` and the green used.
    """
    profile = profile_from_args(args)
    green = green_from_args(args)
    distance = _number(args, "distance_ft", 10.0)
    _check(
        0.0 < distance <= MAX_DISTANCE_FT,
        f"'distance_ft' must be in (0, {MAX_DISTANCE_FT}]",
    )
    n = _whole(args, "n", DEFAULT_DISPERSION_N, MAX_DISPERSION_N)
    seed = _seed(args)
    execution = sample_execution(profile.stroke, n, seed)
    rolled = skill_model(profile, green, distance).roll(
        execution, sample_read(profile.read, n, seed)
    )
    rest_x_ft = (rolled.rest_x_m - distance * FT_M) / FT_M
    rest_y_ft = rolled.rest_y_m / FT_M
    misses = ~rolled.holed
    leave_ft = (rest_x_ft**2 + rest_y_ft**2) ** 0.5
    return {
        "schema_version": 1,
        "distance_ft": distance,
        "green": _green_dict(green),
        "n": n,
        "rest_x_ft": rest_x_ft.tolist(),
        "rest_y_ft": rest_y_ft.tolist(),
        "holed": rolled.holed.tolist(),
        "make_rate": float(rolled.holed.mean()),
        "mean_leave_ft": float(leave_ft[misses].mean()) if misses.any() else None,
    }


def make_spread_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """Make rate by distance on the caller's green (default 3-45 ft every 3 ft)."""
    profile = profile_from_args(args)
    green = green_from_args(args)
    grid = distances_from_args({"distances_ft": list(DEFAULT_SPREAD_GRID_FT), **args})
    n = _whole(args, "n", DEFAULT_SPREAD_N, MAX_SPREAD_N)
    work = n * len(grid)
    _check(
        work <= MAX_CURVE_ROLLOUTS, f"work is {work} putts; cap is {MAX_CURVE_ROLLOUTS}"
    )
    make = simulated_make_rates(profile, grid, n=n, seed=_seed(args), green=green)
    return {
        "schema_version": 1,
        "green": _green_dict(green),
        "distances_ft": list(grid),
        "make_rate": list(make),
        "samples_per_distance": n,
    }


def _green_dict(green: PlaneGreen) -> dict[str, float]:
    return {
        "grade_pct": green.grade_pct,
        "aspect_deg": green.aspect_deg,
        "stimp": green.stimp,
    }


__all__ = [
    "DEFAULT_SPREAD_GRID_FT",
    "MAX_DISPERSION_N",
    "MAX_SPREAD_N",
    "dispersion_study",
    "green_from_args",
    "make_spread_study",
]
