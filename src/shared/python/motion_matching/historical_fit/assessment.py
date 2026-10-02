"""Assess authored coordinate bounds at actual cubic Hermite extrema.

This module does not fit observations or certify nonlinear grip/ground motion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from .contracts import ImageFitResult

_SOURCE_SAMPLE_ATOL = 1e-10
_SOURCE_SAMPLE_RTOL = 1e-8
_BASIS_CHUNK_CELLS = 65536


@dataclass(frozen=True)
class CoordinateExtrema:
    """Scalar coordinate extrema in native units and the source clock."""

    name: str
    minimum: float
    minimum_source_time: float
    maximum: float
    maximum_source_time: float
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        values = (
            self.minimum,
            self.maximum,
            self.minimum_source_time,
            self.maximum_source_time,
        )
        if (
            not isinstance(self.name, str)
            or not self.name
            or not np.isfinite(values).all()
            or self.minimum > self.maximum
        ):
            raise ValueError(
                "Coordinate extrema require a name and finite ordered values"
            )
        if self.bounds is not None:
            object.__setattr__(self, "bounds", _validated_bounds(self.bounds))

    @property
    def lower_violation(self) -> float:
        return 0.0 if self.bounds is None else max(0.0, self.bounds[0] - self.minimum)

    @property
    def upper_violation(self) -> float:
        return 0.0 if self.bounds is None else max(0.0, self.maximum - self.bounds[1])

    @property
    def within_authored_bounds(self) -> bool | None:
        if self.bounds is None:
            return None
        return self.lower_violation == 0.0 and self.upper_violation == 0.0


@dataclass(frozen=True)
class SplineBoundsAssessment:
    """Numerical coordinate assessment; no nonlinear continuous certification."""

    source_interval: tuple[float, float]
    coordinates: tuple[CoordinateExtrema, ...]
    method: str = field(default="cubic_hermite_derivative_roots", init=False)
    continuous_certified: bool = field(default=False, init=False)
    grip_assessment: str = field(default="not_assessed", init=False)
    ground_assessment: str = field(default="not_assessed", init=False)
    source_sample_atol: float = field(default=_SOURCE_SAMPLE_ATOL, init=False)
    source_sample_rtol: float = field(default=_SOURCE_SAMPLE_RTOL, init=False)

    def __post_init__(self) -> None:
        interval = _validated_bounds(self.source_interval, strict=True)
        coordinates = tuple(self.coordinates)
        if len({item.name for item in coordinates}) != len(coordinates):
            raise ValueError("Coordinate assessment names must be unique")
        if any(
            not interval[0] <= time <= interval[1]
            for item in coordinates
            for time in (item.minimum_source_time, item.maximum_source_time)
        ):
            raise ValueError("Extrema must remain inside the source interval")
        object.__setattr__(self, "source_interval", interval)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def bounded_coordinates(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.coordinates if item.bounds is not None)

    @property
    def unbounded_coordinates(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.coordinates if item.bounds is None)

    @property
    def violating_coordinates(self) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in self.coordinates
            if item.within_authored_bounds is False
        )


def _validated_bounds(
    values: tuple[float, float], *, strict: bool = False
) -> tuple[float, float]:
    try:
        original = tuple(values)
        limits = np.asarray(original, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bounds must contain two finite ordered numbers") from exc
    if (
        limits.shape != (2,)
        or any(isinstance(value, (bool, np.bool_)) for value in original)
        or not np.isfinite(limits).all()
        or limits[0] > limits[1]
        or (strict and limits[0] == limits[1])
    ):
        raise ValueError("Bounds must contain two finite ordered numbers")
    return float(limits[0]), float(limits[1])


def _validate_fit(fit: ImageFitResult) -> None:
    order, free = fit.coordinate_order, fit.free_coordinates
    if not free:
        raise ValueError("Spline assessment requires at least one free coordinate")
    if (
        not order
        or len(set(order)) != len(order)
        or any(not isinstance(name, str) or not name for name in order)
        or len(set(free)) != len(free)
        or any(name not in order for name in free)
    ):
        raise ValueError("Fit coordinate identities must be unique and known")
    times, knots = fit.source_times, fit.knot_times
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.diff(times) > 0)
        or knots.ndim != 1
        or len(knots) < 2
        or not np.all(np.diff(knots) > 0)
        or times[0] != knots[0]
        or times[-1] != knots[-1]
        or fit.q.shape != (len(times), len(order))
    ):
        raise ValueError(
            "Fit arrays must match coordinates and the exact increasing source interval"
        )
    locked = [index for index, name in enumerate(order) if name not in free]
    if not np.array_equal(fit.q[:, locked], np.tile(fit.q[0, locked], (len(times), 1))):
        raise ValueError("Locked coordinates must retain constant source poses")
    if fit.spline_coefficients.shape != (2 * len(knots) * len(free),):
        raise ValueError(
            "Spline coefficients must match the free coordinates and knots"
        )


def _validate_source_samples(
    fit: ImageFitResult, positions: np.ndarray, velocities: np.ndarray
) -> None:
    """Compare stored samples with the canonical spline using bounded scalar batches."""
    trajectory = CubicHermiteSplineTrajectory(fit.knot_times, 1)
    batch = max(1, _BASIS_CHUNK_CELLS // trajectory.coefficient_size)
    for column, name in enumerate(fit.free_coordinates):
        coefficients = trajectory.pack(
            positions[:, column, None], velocities[:, column, None]
        )
        expected = np.empty(len(fit.source_times))
        for start in range(0, len(expected), batch):
            stop = start + batch
            expected[start:stop] = trajectory.evaluate(
                coefficients, fit.source_times[start:stop]
            ).q[:, 0]
        if not np.allclose(
            expected,
            fit.q[:, fit.coordinate_order.index(name)],
            atol=_SOURCE_SAMPLE_ATOL,
            rtol=_SOURCE_SAMPLE_RTOL,
        ):
            raise ValueError("Stored source samples disagree with the preserved spline")


def _stationary_fractions(
    q0: float, q1: float, v0: float, v1: float, h: float
) -> np.ndarray:
    """Real interior roots of the actual cubic derivative in normalized time."""
    with np.errstate(over="raise", invalid="raise"):
        try:
            a = 2 * q0 - 2 * q1 + h * (v0 + v1)
            b = -3 * q0 + 3 * q1 - h * (2 * v0 + v1)
            derivative = np.asarray([3 * a, 2 * b, h * v0])
        except FloatingPointError as exc:
            raise ValueError("Spline derivative overflowed") from exc
    if not np.isfinite(derivative).all():
        raise ValueError("Spline derivative must remain finite")
    scale = float(np.max(np.abs(derivative)))
    roots = np.roots(derivative / scale) if scale else np.empty(0)
    interior = roots.real[
        (np.abs(roots.imag) < 1e-10) & (roots.real > 0) & (roots.real < 1)
    ]
    return np.unique(np.r_[0.0, interior, 1.0])


def _free_extrema(
    knots: np.ndarray, positions: np.ndarray, velocities: np.ndarray
) -> tuple[float, float, float, float]:
    """Evaluate each scalar segment through the canonical public spline API.

    Local one-DOF segments avoid allocating whole-model coefficient Jacobians
    for every candidate extremum in a long track.
    """
    candidates: list[tuple[float, float]] = []
    for index in range(len(knots) - 1):
        h = float(knots[index + 1] - knots[index])
        fractions = _stationary_fractions(
            float(positions[index]),
            float(positions[index + 1]),
            float(velocities[index]),
            float(velocities[index + 1]),
            h,
        )
        times = knots[index] + h * fractions
        trajectory = CubicHermiteSplineTrajectory(knots[index : index + 2], 1)
        coefficients = trajectory.pack(
            positions[index : index + 2, None], velocities[index : index + 2, None]
        )
        values = trajectory.evaluate(coefficients, times).q[:, 0]
        if not np.isfinite(values).all():
            raise ValueError("Spline extrema must remain finite")
        candidates.extend(
            (float(value), float(time))
            for value, time in zip(values, times, strict=True)
        )
    minimum, min_time = min(candidates, key=lambda item: (item[0], item[1]))
    maximum, max_time = min(candidates, key=lambda item: (-item[0], item[1]))
    return minimum, min_time, maximum, max_time


def assess_spline_bounds(
    fit: ImageFitResult, bounds: Mapping[str, tuple[float, float]]
) -> SplineBoundsAssessment:
    """Assess every stored scalar spline over its exact source-clock interval.

    Bounds must be named finite authored limits in coordinate-native units.
    No image evidence is synthesized and no native engine internals are used.
    Floating-point polynomial assessment does not qualify nonlinear constraints,
    camera/anatomy, historical identity, physical timing, or dynamic feasibility.
    """
    _validate_fit(fit)
    if any(
        not isinstance(name, str) or name not in fit.coordinate_order for name in bounds
    ):
        raise ValueError("Authored bounds reference an unknown coordinate")
    authored = {name: _validated_bounds(limits) for name, limits in bounds.items()}
    free = fit.free_coordinates
    if free:
        trajectory = CubicHermiteSplineTrajectory(fit.knot_times, len(free))
        positions, velocities = trajectory.unpack(fit.spline_coefficients)
        _validate_source_samples(fit, positions, velocities)
    coordinates = []
    start = float(fit.knot_times[0])
    for index, name in enumerate(fit.coordinate_order):
        if name in free:
            column = free.index(name)
            extrema = _free_extrema(
                fit.knot_times, positions[:, column], velocities[:, column]
            )
        else:
            value = float(fit.q[0, index])
            extrema = value, start, value, start
        coordinates.append(CoordinateExtrema(name, *extrema, authored.get(name)))
    return SplineBoundsAssessment(
        (start, float(fit.knot_times[-1])), tuple(coordinates)
    )
