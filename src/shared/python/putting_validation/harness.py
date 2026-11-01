"""Band-comparison harness: simulated putting curves against a reference.

Pure functions. Residual is ``simulated - reference``; a comparison passes when
``max |residual| <= tolerance`` (edge inclusive). It never adjusts anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src.shared.python.contracts import require
from src.shared.python.putting_validation.reference_data import (
    Quantity,
    ReferenceSeries,
    is_strictly_increasing,
)

Curve = Callable[[float], float] | Sequence[float]

MIN_EXPECTED_PUTTS = 1.0  # a holed ball needs at least one putt


@dataclass(frozen=True)
class BandComparison:
    """Result of comparing a simulated curve with a reference curve."""

    distances_ft: tuple[float, ...]
    residuals: tuple[float, ...]
    max_abs_residual: float
    worst_distance_ft: float
    tolerance: float
    passed: bool


def _evaluate(curve: Curve, distances_ft: Sequence[float]) -> list[float]:
    if callable(curve):
        return [float(curve(d)) for d in distances_ft]
    require(len(curve) == len(distances_ft), "simulated length must match distances")
    return [float(v) for v in curve]


def _check_values(values: Sequence[float], quantity: Quantity, name: str) -> None:
    require(all(math.isfinite(v) for v in values), f"{name} must be finite")
    if quantity is Quantity.MAKE_FRACTION:
        require(all(0.0 <= v <= 1.0 for v in values), f"{name} must lie in [0, 1]")
    else:
        require(
            all(v >= MIN_EXPECTED_PUTTS for v in values),
            f"{name} must be >= {MIN_EXPECTED_PUTTS} putt",
        )


def compare_curves(
    simulated: Curve,
    distances_ft: Sequence[float],
    reference: Sequence[float],
    tolerance: float,
    quantity: Quantity = Quantity.MAKE_FRACTION,
) -> BandComparison:
    """Compare a simulated curve with reference values at ``distances_ft``.

    ``simulated`` is a callable of distance (ft) or a sequence aligned with
    ``distances_ft``. Preconditions: distances positive and increasing, equal
    lengths, finite tolerance > 0, values valid for ``quantity``.
    """
    require(len(distances_ft) > 0, "distances_ft must not be empty")
    require(distances_ft[0] > 0, "distances_ft must be > 0", distances_ft[0])
    require(is_strictly_increasing(distances_ft), "distances_ft not increasing")
    require(len(reference) == len(distances_ft), "reference length mismatch")
    require(math.isfinite(tolerance) and tolerance > 0, "tolerance must be > 0")
    sim = _evaluate(simulated, distances_ft)
    _check_values(reference, quantity, "reference")
    _check_values(sim, quantity, "simulated")
    residuals = tuple(s - r for s, r in zip(sim, reference, strict=True))
    worst = max(range(len(residuals)), key=lambda i: abs(residuals[i]))
    max_abs = abs(residuals[worst])
    return BandComparison(
        distances_ft=tuple(float(d) for d in distances_ft),
        residuals=residuals,
        max_abs_residual=max_abs,
        worst_distance_ft=float(distances_ft[worst]),
        tolerance=tolerance,
        passed=max_abs <= tolerance,
    )


def compare_to_series(
    simulated: Curve, series: ReferenceSeries, tolerance: float
) -> BandComparison:
    """Compare at the series' own distances (no ``validation_grade`` check)."""
    return compare_curves(
        simulated, series.distances_ft, series.values, tolerance, series.quantity
    )
