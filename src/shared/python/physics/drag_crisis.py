"""Calibrated golf-ball drag crisis curve.

The table in ``calibration_data/golf_ball_drag_bearman_harvey.csv`` is a
small engineering calibration dataset derived from published dimpled golf-ball
drag-crisis curves:

- Bearman, P. W. & Harvey, J. K. (1976). Golf ball aerodynamics.
  Aeronautical Quarterly, 27(2), 112-122.
- Smits, A. J. & Ogg, S. (2004). Golf ball aerodynamics. Physics Today.

The values are intended to replace the previous three-segment monotone
approximation with a source-tagged curve that preserves the drag minimum near
Re ~= 7e4 and the modest post-crisis Cd recovery.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
from typing import Final

DEFAULT_REFERENCE_REYNOLDS: Final[float] = 160_000.0


@lru_cache(maxsize=1)
def load_golf_ball_drag_calibration() -> tuple[tuple[float, float], ...]:
    """Load the committed golf-ball Cd(Re) calibration table."""
    data_path = (
        resources.files("src.shared.python.physics.calibration_data")
        / "golf_ball_drag_bearman_harvey.csv"
    )
    with data_path.open("r", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        points = tuple(
            (float(row["reynolds_number"]), float(row["drag_coefficient"]))
            for row in rows
        )

    if len(points) < 2:
        raise ValueError("drag calibration table must contain at least two points")
    if any(
        points[index][0] >= points[index + 1][0] for index in range(len(points) - 1)
    ):
        raise ValueError(
            "drag calibration Reynolds numbers must be strictly increasing"
        )
    return points


def calibrated_golf_ball_drag_coefficient(
    reynolds_number: float,
    reference_drag_coefficient: float = 0.25,
) -> float:
    """Return a natural-cubic interpolated golf-ball drag coefficient.

    Args:
        reynolds_number: Ball Reynolds number based on diameter.
        reference_drag_coefficient: Existing simulator/user post-crisis Cd
            setting. The calibration curve is scaled so its value at
            ``DEFAULT_REFERENCE_REYNOLDS`` matches this reference, preserving
            coefficient tunability while fixing the drag-crisis shape.
    """
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reference_drag_coefficient < 0:
        raise ValueError("reference_drag_coefficient must be non-negative")

    points = load_golf_ball_drag_calibration()
    re_value = float(reynolds_number)
    cd = _natural_cubic_interpolate(points, re_value)
    reference_cd = _natural_cubic_interpolate(points, DEFAULT_REFERENCE_REYNOLDS)
    if reference_cd <= 0:
        raise ValueError("drag calibration reference coefficient must be positive")
    return cd * (float(reference_drag_coefficient) / reference_cd)


def _natural_cubic_interpolate(
    points: tuple[tuple[float, float], ...], x: float
) -> float:
    """Interpolate with natural cubic splines, clamping outside the data range."""
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    second = _natural_cubic_second_derivatives(xs, ys)

    low = 0
    high = len(xs) - 1
    while high - low > 1:
        mid = (high + low) // 2
        if xs[mid] > x:
            high = mid
        else:
            low = mid

    h = xs[high] - xs[low]
    if h <= 0:
        raise ValueError("drag calibration points must be strictly increasing")

    a = (xs[high] - x) / h
    b = (x - xs[low]) / h
    return (
        a * ys[low]
        + b * ys[high]
        + ((a**3 - a) * second[low] + (b**3 - b) * second[high]) * h**2 / 6.0
    )


@lru_cache(maxsize=8)
def _natural_cubic_second_derivatives(
    xs_tuple: tuple[float, ...] | list[float],
    ys_tuple: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Compute natural spline second derivatives for the calibration table."""
    xs = tuple(xs_tuple)
    ys = tuple(ys_tuple)
    count = len(xs)
    second = [0.0] * count
    work = [0.0] * (count - 1)

    for index in range(1, count - 1):
        sig = (xs[index] - xs[index - 1]) / (xs[index + 1] - xs[index - 1])
        p = sig * second[index - 1] + 2.0
        second[index] = (sig - 1.0) / p
        slope_next = (ys[index + 1] - ys[index]) / (xs[index + 1] - xs[index])
        slope_prev = (ys[index] - ys[index - 1]) / (xs[index] - xs[index - 1])
        work[index] = (
            6.0 * (slope_next - slope_prev) / (xs[index + 1] - xs[index - 1])
            - sig * work[index - 1]
        ) / p

    for index in range(count - 2, -1, -1):
        second[index] = second[index] * second[index + 1] + work[index]

    return tuple(second)
