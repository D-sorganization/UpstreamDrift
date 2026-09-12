"""Bounded least-squares objectives with explicit projected equality rows."""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

Array: TypeAlias = NDArray[np.float64]


class _EvaluationBudgetExceeded(Exception):
    """Stop before evaluating an additional physical residual."""


def solve_equality_least_squares(
    residual: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    initial: Array,
    lower: Array,
    upper: Array,
    *,
    equality_start: int,
    projection: Array,
    max_iterations: int,
    max_evaluations: int,
    constraint_tolerance: float = 1e-8,
) -> OptimizeResult:
    """Minimize non-equality squared residuals subject to P @ equality_rows=0.

    The projection is fixed, finite and full row rank. It does not certify that
    omitted physical rows vanish: callers must check full physical continuity.
    Objective/constraint calls share each residual evaluation. Iterations and
    evaluations have distinct limits. Budget exhaustion returns a previously
    evaluated point and NEVER reports success. Selection prefers a feasible
    lower-cost point, otherwise smaller projected violation. This is not fit
    acceptance or an independent continuous replay.
    """
    x0, lo, hi, p = (
        np.array(v, dtype=float, copy=True) for v in (initial, lower, upper, projection)
    )
    if (
        x0.ndim != 1
        or not x0.size
        or lo.shape != x0.shape
        or hi.shape != x0.shape
        or not all(np.isfinite(v).all() for v in (x0, lo, hi))
        or np.any(lo >= hi)
        or np.any(x0 < lo)
        or np.any(x0 > hi)
    ):
        raise ValueError("Invalid bounded initial variables")
    if (
        p.ndim != 2
        or not 0 < p.shape[0] <= p.shape[1]
        or not np.isfinite(p).all()
        or np.linalg.matrix_rank(p) != p.shape[0]
    ):
        raise ValueError("Invalid or rank-deficient equality projection")
    if (
        isinstance(equality_start, bool)
        or not isinstance(equality_start, int)
        or equality_start < 0
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (max_iterations, max_evaluations)
        )
        or not np.isfinite(constraint_tolerance)
        or constraint_tolerance <= 0
    ):
        raise ValueError("Invalid equality location, budget or tolerance")
    end = equality_start + p.shape[1]
    cache_key: bytes | None = None
    cache_r: Array = np.empty(0)
    cache_j: Array | None = None
    count = 0
    best_x = x0.copy()
    best_cost = float("inf")
    best_score = (True, float("inf"))

    def objective_rows(value: Array) -> Array:
        return np.concatenate((value[:equality_start], value[end:]), axis=0)

    def evaluate(x: Array) -> Array:
        nonlocal cache_key, cache_r, cache_j, count, best_x, best_cost, best_score
        key = x.tobytes()
        if key != cache_key:
            if count >= max_evaluations:
                raise _EvaluationBudgetExceeded
            value = np.array(residual(x.copy()), dtype=float, copy=True)
            count += 1
            if (
                value.ndim != 1
                or value.size <= p.shape[1]
                or end > value.size
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid finite residual or equality row range")
            cost = float(objective_rows(value) @ objective_rows(value))
            violation = float(np.linalg.norm(p @ value[equality_start:end], ord=np.inf))
            score = (
                violation > constraint_tolerance,
                violation if violation > constraint_tolerance else cost,
            )
            if score < best_score:
                best_x, best_cost, best_score = x.copy(), cost, score
            cache_key, cache_r, cache_j = key, value, None
        return cache_r

    def derivative(x: Array) -> Array:
        nonlocal cache_j
        value = evaluate(x)
        if cache_j is None:
            cache_j = np.array(jacobian(x.copy()), dtype=float, copy=True)
            if cache_j.shape != (value.size, x0.size) or not np.isfinite(cache_j).all():
                raise ValueError("Invalid finite residual Jacobian")
        return cache_j

    def objective(x: Array) -> float:
        value = objective_rows(evaluate(x))
        return float(value @ value)

    def gradient(x: Array) -> Array:
        value = objective_rows(evaluate(x))
        return 2 * objective_rows(derivative(x)).T @ value

    def constraint(x: Array) -> Array:
        return p @ evaluate(x)[equality_start:end]

    def constraint_jacobian(x: Array) -> Array:
        return p @ derivative(x)[equality_start:end]

    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=list(zip(lo, hi, strict=True)),
            constraints={"type": "eq", "fun": constraint, "jac": constraint_jacobian},
            options={"maxiter": max_iterations, "ftol": constraint_tolerance},
        )
    except _EvaluationBudgetExceeded:
        result = OptimizeResult(
            x=best_x,
            fun=best_cost,
            success=False,
            status=9,
            message="Physical residual evaluation budget exhausted; returned evaluated fallback",
        )
    result.nfev = count
    # Comparable least_squares scaled optimality is unavailable from SLSQP.
    result.active_mask = np.where(
        np.isclose(result.x, lo, rtol=0, atol=1e-8),
        -1,
        np.where(np.isclose(result.x, hi, rtol=0, atol=1e-8), 1, 0),
    )
    return result
