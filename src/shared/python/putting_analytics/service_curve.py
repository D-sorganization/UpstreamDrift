"""JSON service for the simulated strokes-gained putting curve (#14177).

:func:`curve_study` turns a player's stroke and read dispersion into the
per-distance table that the strokes-gained optimizer's ``"simulation"``
putting provider reads (:class:`.SimulatedPuttingCurve`), via
:func:`.sg_curve.simulate_putting_curve`. :func:`curve_from_report` turns that
JSON back into the curve, so a client can build the curve once and hand it to
:func:`src.shared.python.sg_optimizer.shot_model.putting_provider.putting_provider`.

Arguments are untrusted tool input: the player uses the same bounds as the
variance study (:func:`.service_variance.profile_from_args`) and every other
number is checked here with :func:`.service._check` before the
contract-checked simulation sees it (``require`` can be switched off
globally).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

from src.shared.python.putting_analytics.service import (
    MAX_DISTANCE_FT,
    MAX_LIBRARY_ROLLOUTS,
    _check,
    _number,
)
from src.shared.python.putting_analytics.service_variance import profile_from_args
from src.shared.python.putting_analytics.sg_curve import (
    DEFAULT_CURVE_GRID_FT,
    MIN_MISSES,
    simulate_putting_curve,
)
from src.shared.python.sg_optimizer.shot_model.putting_provider import (
    REFERENCE_STIMP,
    SimulatedPuttingCurve,
)

DEFAULT_CURVE_N = 2000
MAX_CURVE_N = 5000
MAX_CURVE_DISTANCES = 30
MAX_CURVE_ROLLOUTS = MAX_LIBRARY_ROLLOUTS
_CURVE_FIELDS = ("distances_ft", "make_rate", "leave_scale_ft", "leave_sigma_log")


def distances_from_args(args: Mapping[str, Any]) -> tuple[float, ...]:
    """Distance grid in feet; defaults to the optimizer's grid."""
    raw = args.get("distances_ft", list(DEFAULT_CURVE_GRID_FT))
    _check(isinstance(raw, list | tuple), "'distances_ft' must be a list")
    _check(
        2 <= len(raw) <= MAX_CURVE_DISTANCES,
        f"'distances_ft' needs at least 2 and at most {MAX_CURVE_DISTANCES} values",
    )
    grid = tuple(_number({"d": value}, "d") for value in raw)
    _check(
        all(0.0 < d <= MAX_DISTANCE_FT for d in grid),
        f"'distances_ft' values must be in (0, {MAX_DISTANCE_FT}]",
    )
    _check(
        all(b > a for a, b in itertools.pairwise(grid)), "'distances_ft' must increase"
    )
    return grid


def _putts_per_distance(args: Mapping[str, Any]) -> int:
    n = _number(args, "n", DEFAULT_CURVE_N)
    _check(n == int(n), "'n' must be a whole number")
    _check(
        MIN_MISSES <= n <= MAX_CURVE_N, f"'n' must be in [{MIN_MISSES}, {MAX_CURVE_N}]"
    )
    return int(n)


def _seed(args: Mapping[str, Any]) -> int:
    seed = _number(args, "seed", 0)
    _check(seed >= 0 and seed == int(seed), "'seed' must be a non-negative integer")
    return int(seed)


def curve_study(args: Mapping[str, Any]) -> dict[str, Any]:
    """Simulated make rate and miss-leave fit by distance for one player.

    Returns ``distances_ft``, ``make_rate``, ``leave_scale_ft`` and
    ``leave_sigma_log`` (one entry per distance), ``samples_per_distance`` and
    ``reference_stimp``, the green speed the curve is built at.
    """
    profile = profile_from_args(args)
    grid = distances_from_args(args)
    n = _putts_per_distance(args)
    seed = _seed(args)
    work = n * len(grid)
    _check(
        work <= MAX_CURVE_ROLLOUTS, f"work is {work} putts; cap is {MAX_CURVE_ROLLOUTS}"
    )
    curve = simulate_putting_curve(profile, grid, n=n, seed=seed)
    report: dict[str, Any] = {
        name: list(getattr(curve, name)) for name in _CURVE_FIELDS
    }
    report.update(
        schema_version=1,
        samples_per_distance=curve.samples_per_distance,
        reference_stimp=REFERENCE_STIMP,
    )
    return report


def _column(report: Mapping[str, Any], name: str) -> tuple[float, ...]:
    values = report.get(name)
    _check(isinstance(values, list | tuple), f"'{name}' must be a list")
    assert isinstance(values, list | tuple)  # narrowed by the check above
    floats = tuple(float(v) for v in values)
    _check(all(math.isfinite(v) for v in floats), f"'{name}' must be finite")
    return floats


def curve_from_report(report: Mapping[str, Any]) -> SimulatedPuttingCurve:
    """Rebuild the optimizer's curve from a :func:`curve_study` report."""
    distances, make, scale, sigma = (_column(report, name) for name in _CURVE_FIELDS)
    samples = int(report.get("samples_per_distance", 0))
    return SimulatedPuttingCurve(
        distances, make, scale, sigma, samples_per_distance=samples
    )


__all__ = [
    "DEFAULT_CURVE_GRID_FT",
    "MAX_CURVE_DISTANCES",
    "MAX_CURVE_N",
    "MAX_CURVE_ROLLOUTS",
    "curve_from_report",
    "curve_study",
    "distances_from_args",
]
