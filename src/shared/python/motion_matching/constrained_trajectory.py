"""Sampled collocation for a smooth holonomically constrained trajectory.

The module accepts independent position, velocity, and acceleration closure
oracles.  It deliberately does not identify efforts or integrate dynamics.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

Array = NDArray[np.float64]
PositionClosure = Callable[[Array], Array]
RateClosure = Callable[[Array, Array], Array]
AccelerationClosure = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class TrajectoryClosureReport:
    """Closure values and spline-derived derivatives at supplied sample times."""

    rates: Array
    accelerations: Array
    position_max_abs: float
    rate_max_abs: float
    acceleration_max_abs: float


@dataclass(frozen=True)
class CollocationResult:
    """A position correction and its independently recomputed closure report."""

    coordinates: Array
    closure: TrajectoryClosureReport
    optimizer_converged: bool
    message: str
    iterations: int


def _validate_samples(times: Array, coordinates: Array) -> tuple[Array, Array]:
    clock = np.asarray(times, dtype=float)
    values = np.asarray(coordinates, dtype=float)
    if (
        clock.ndim != 1
        or clock.size < 4
        or not np.isfinite(clock).all()
        or not np.all(np.diff(clock) > 0)
        or values.ndim != 2
        or values.shape[0] != clock.size
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Expected at least four finite, strictly timed coordinate samples"
        )
    return clock, values


def spline_node_derivative_maps(times: Array) -> tuple[Array, Array]:
    """Return linear maps from node values to cubic-spline qd and qdd at nodes."""
    clock = np.asarray(times, dtype=float)
    if clock.ndim != 1 or clock.size < 4 or not np.all(np.diff(clock) > 0):
        raise ValueError("Expected at least four strictly increasing finite times")
    identity = np.eye(clock.size)
    spline = CubicSpline(clock, identity, axis=0)
    first = np.asarray(spline(clock, 1), dtype=float)
    second = np.asarray(spline(clock, 2), dtype=float)
    return first, second


def _closure_values(
    positions: Array,
    rates: Array,
    accelerations: Array,
    position_closure: PositionClosure,
    rate_closure: RateClosure,
    acceleration_closure: AccelerationClosure,
) -> tuple[Array, Array, Array]:
    position_values: list[Array] = [position_closure(cast(Array, q)) for q in positions]
    rate_values: list[Array] = [
        rate_closure(cast(Array, q), cast(Array, v))
        for q, v in zip(positions, rates, strict=True)
    ]
    acceleration_values: list[Array] = [
        acceleration_closure(cast(Array, q), cast(Array, v), cast(Array, a))
        for q, v, a in zip(positions, rates, accelerations, strict=True)
    ]
    position = np.asarray(position_values, dtype=np.float64)
    rate = np.asarray(rate_values, dtype=np.float64)
    acceleration = np.asarray(acceleration_values, dtype=np.float64)
    if (
        position.ndim != 2
        or rate.shape != position.shape
        or acceleration.shape != position.shape
        or position.shape[1] == 0
        or not np.isfinite(position).all()
        or not np.isfinite(rate).all()
        or not np.isfinite(acceleration).all()
    ):
        raise ValueError("Closure oracles must return same-shaped finite vectors")
    return position, rate, acceleration


def evaluate_trajectory_closure(
    times: Array,
    coordinates: Array,
    position_closure: PositionClosure,
    rate_closure: RateClosure,
    acceleration_closure: AccelerationClosure,
) -> TrajectoryClosureReport:
    """Evaluate C2 spline derivatives and all three closure levels at nodes."""
    clock, positions = _validate_samples(times, coordinates)
    spline = CubicSpline(clock, positions, axis=0)
    rates = np.asarray(spline(clock, 1), dtype=float)
    accelerations = np.asarray(spline(clock, 2), dtype=float)
    position, rate, acceleration = _closure_values(
        positions,
        rates,
        accelerations,
        position_closure,
        rate_closure,
        acceleration_closure,
    )
    for value in (rates, accelerations):
        value.setflags(write=False)
    return TrajectoryClosureReport(
        rates=rates,
        accelerations=accelerations,
        position_max_abs=float(np.max(np.abs(position))),
        rate_max_abs=float(np.max(np.abs(rate))),
        acceleration_max_abs=float(np.max(np.abs(acceleration))),
    )


def collocate_positions(
    times: Array,
    seed_coordinates: Array,
    position_closure: PositionClosure,
    rate_closure: RateClosure,
    acceleration_closure: AccelerationClosure,
    *,
    max_iterations: int = 100,
    closure_tolerance: float = 1e-7,
) -> CollocationResult:
    """Minimize node displacement while enforcing spline closure through qdd.

    The returned qd/qdd are always recomputed from the returned position spline.
    This prevents combining separately projected state derivatives as though they
    formed a single trajectory.
    """
    clock, seed = _validate_samples(times, seed_coordinates)
    if (
        type(max_iterations) is not int
        or max_iterations < 1
        or not np.isfinite(closure_tolerance)
        or closure_tolerance <= 0
    ):
        raise ValueError(
            "Expected positive finite closure tolerance and iteration budget"
        )
    shape = seed.shape

    def closure_vector(flat: Array) -> Array:
        report = evaluate_trajectory_closure(
            clock,
            flat.reshape(shape),
            position_closure,
            rate_closure,
            acceleration_closure,
        )
        values = _closure_values(
            flat.reshape(shape),
            report.rates,
            report.accelerations,
            position_closure,
            rate_closure,
            acceleration_closure,
        )
        return np.concatenate(values).reshape(-1)

    start = np.array(seed, copy=True).reshape(-1)
    result = minimize(
        lambda flat: float(np.sum((flat - start) ** 2)),
        start,
        method="trust-constr",
        constraints={"type": "eq", "fun": closure_vector},
        options={"maxiter": max_iterations, "gtol": 1e-12},
    )
    coordinates = np.asarray(result.x, dtype=float).reshape(shape)
    report = evaluate_trajectory_closure(
        clock,
        coordinates,
        position_closure,
        rate_closure,
        acceleration_closure,
    )
    coordinates.setflags(write=False)
    return CollocationResult(
        coordinates=coordinates,
        closure=report,
        optimizer_converged=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
    )
