"""Local static marker feasibility with explicit holonomic closure constraints.

A feasible pose is not a forward trajectory; a failed local solve does not
prove a global kinematic error floor. No dynamics or actuator fitting occurs.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .marker_replay_report import marker_errors

Array = NDArray[np.float64]


@dataclass(frozen=True)
class MarkerPoseResult:
    """Independently evaluated local pose and distinct constraint/solver status."""

    coordinates: Array
    marker_rms_m: float
    closure_max_abs: float
    closure_satisfied: bool
    optimizer_converged: bool
    message: str
    iterations: int


def fit_marker_pose(
    initial: Array,
    lower: Array,
    upper: Array,
    target_m: Array,
    valid: NDArray[np.bool_],
    forward: Callable[[Array], Array],
    closure: Callable[[Array], Array],
    *,
    forward_jacobian: Callable[[Array], Array] | None = None,
    closure_jacobian: Callable[[Array], Array] | None = None,
    max_iterations: int = 100,
    closure_tolerance: float = 1e-7,
) -> MarkerPoseResult:
    """Minimize observed marker distance subject to exact local pose closure."""
    start, lo, hi = (
        np.array(v, dtype=float, copy=True) for v in (initial, lower, upper)
    )
    target, observed = np.asarray(target_m), np.asarray(valid)
    if (
        start.ndim != 1
        or not start.size
        or lo.shape != start.shape
        or hi.shape != start.shape
        or not np.isfinite([start, lo, hi]).all()
        or np.any(lo >= hi)
        or np.any(start < lo)
        or np.any(start > hi)
    ):
        raise ValueError("Invalid initial pose or local coordinate bounds")
    if (
        type(max_iterations) is not int
        or max_iterations < 1
        or not np.isfinite(closure_tolerance)
        or closure_tolerance <= 0
    ):
        raise ValueError("Invalid solver budget or closure tolerance")
    if (
        target.ndim != 2
        or target.shape[1] != 3
        or observed.shape != target.shape[:1]
        or not observed.any()
    ):
        raise ValueError("Expected observed named marker coordinates")

    def distances(q: Array) -> Array:
        prediction = np.asarray(forward(q), dtype=float)
        return marker_errors(target[None], prediction[None], observed[None])[
            0, observed
        ]

    initial_closure = np.asarray(closure(start), dtype=float)
    if (
        initial_closure.ndim != 1
        or not initial_closure.size
        or not np.isfinite(initial_closure).all()
    ):
        raise ValueError("Expected finite closure vector")

    def constraint(q: Array) -> Array:
        value = np.asarray(closure(q), dtype=float)
        if value.shape != initial_closure.shape or not np.isfinite(value).all():
            raise ValueError("Closure oracle changed shape or became non-finite")
        return value

    def cost(q: Array) -> float:
        error = distances(q)
        return float(error @ error)

    def cost_jacobian(q: Array) -> Array:
        assert forward_jacobian is not None
        derivative = np.asarray(forward_jacobian(q), dtype=float)
        if (
            derivative.shape != (*target.shape, start.size)
            or not np.isfinite(derivative).all()
        ):
            raise ValueError(
                "Marker Jacobian must have finite marker/xyz/coordinate shape"
            )
        prediction = np.asarray(forward(q), dtype=float)
        distances(q)  # Preserve the forward oracle's validation and mask contract.
        delta = prediction[observed] - target[observed]
        return 2 * np.einsum("mi,mij->j", delta, derivative[observed])

    def constraint_jacobian(q: Array) -> Array:
        assert closure_jacobian is not None
        derivative = np.asarray(closure_jacobian(q), dtype=float)
        if (
            derivative.shape != (initial_closure.size, start.size)
            or not np.isfinite(derivative).all()
        ):
            raise ValueError(
                "Closure Jacobian must have finite constraint/coordinate shape"
            )
        return derivative

    constraints = {"type": "eq", "fun": constraint}
    if closure_jacobian is not None:
        constraint_jacobian(start)
        constraints["jac"] = constraint_jacobian
    if forward_jacobian is not None:
        cost_jacobian(start)
    cost(start)
    optimum = minimize(
        cost,
        start,
        method="SLSQP",
        bounds=list(zip(lo, hi, strict=True)),
        constraints=constraints,
        jac=cost_jacobian if forward_jacobian is not None else None,
        options={"maxiter": max_iterations, "ftol": 1e-12},
    )
    coordinates = np.array(optimum.x, copy=True)
    errors = distances(coordinates)
    closure_max = float(np.max(np.abs(constraint(coordinates))))
    coordinates.setflags(write=False)
    return MarkerPoseResult(
        coordinates,
        float(np.sqrt(np.mean(errors**2))),
        closure_max,
        closure_max <= closure_tolerance,
        bool(optimum.success),
        str(optimum.message),
        int(optimum.nit),
    )
