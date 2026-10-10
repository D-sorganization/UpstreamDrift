"""One bounded SLSQP search loop over caller-owned prediction and hard margins.

The caller owns physical units, native prediction, source identity, and the
independently admitted fallback. A solver result is only a proposal: this loop
re-evaluates its signed hard margins before returning a candidate plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize

if TYPE_CHECKING:
    from scipy.optimize._minimize import _MinimizeOptions

Array: TypeAlias = NDArray[np.float64]
CandidateEvaluator: TypeAlias = Callable[[Array], tuple[float, Array]]
Solver: TypeAlias = Callable[..., OptimizeResult]


def _slsqp_options(max_iterations: int) -> _MinimizeOptions:
    return {"maxiter": max_iterations, "ftol": 1e-7}


def _frozen(values: Array) -> Array:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CandidateSearchProblem:
    """Predeclared plan bounds and signed-margin numerical evaluator."""

    initial_plan: Array
    fallback_plan: Array
    lower: Array
    upper: Array
    evaluate: CandidateEvaluator

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_plan, dtype=np.float64)
        fallback = np.asarray(self.fallback_plan, dtype=np.float64)
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if (
            initial.ndim != 2
            or initial.size == 0
            or fallback.shape != initial.shape
            or lower.shape != (initial.shape[1],)
            or upper.shape != lower.shape
            or not all(np.isfinite(a).all() for a in (initial, fallback, lower, upper))
            or np.any(lower >= upper)
            or np.any(initial < lower)
            or np.any(initial > upper)
            or np.any(fallback < lower)
            or np.any(fallback > upper)
            or not callable(self.evaluate)
        ):
            raise ValueError(
                "candidate search needs finite bounded plans and evaluator"
            )
        for name, values in (
            ("initial_plan", initial),
            ("fallback_plan", fallback),
            ("lower", lower),
            ("upper", upper),
        ):
            object.__setattr__(self, name, _frozen(values))


@dataclass(frozen=True)
class CandidateSearchPolicy:
    """Cooperative numerical budgets; elapsed wall time remains measured evidence."""

    max_evaluations: int
    max_wall_s: float
    solver_max_iterations: int
    feasibility_tolerance: float

    def __post_init__(self) -> None:
        if (
            self.max_evaluations < 1
            or self.solver_max_iterations < 1
            or not np.isfinite(self.max_wall_s)
            or self.max_wall_s <= 0.0
            or not np.isfinite(self.feasibility_tolerance)
            or self.feasibility_tolerance < 0.0
        ):
            raise ValueError("candidate search budgets and tolerance are invalid")


@dataclass(frozen=True)
class CandidateSearchReceipt:
    """Numerical result only; caller must independently execute/promote it."""

    status: str
    candidate_plan: Array | None
    objective: float | None
    fallback_objective: float
    fallback_horizon_feasible: bool
    evaluations: int
    elapsed_s: float

    def __post_init__(self) -> None:
        if self.candidate_plan is not None:
            object.__setattr__(self, "candidate_plan", _frozen(self.candidate_plan))


class _BudgetExpired(Exception):
    pass


class _SolveCancelled(Exception):
    pass


class _PredictionFailure(Exception):
    pass


def _evaluate(problem: CandidateSearchProblem, plan: Array) -> tuple[float, Array]:
    try:
        cost, supplied = problem.evaluate(_frozen(plan))
        margins = np.asarray(supplied, dtype=np.float64)
        numeric_cost = float(cost)
    except Exception as exc:
        raise _PredictionFailure from exc
    if (
        not np.isfinite(numeric_cost)
        or margins.ndim != 1
        or margins.size == 0
        or not np.isfinite(margins).all()
    ):
        raise _PredictionFailure
    return numeric_cost, _frozen(margins)


def _admit_result(
    problem: CandidateSearchProblem,
    policy: CandidateSearchPolicy,
    result: OptimizeResult,
    fallback_cost: float,
) -> tuple[str, float | None, Array | None]:
    try:
        candidate = np.asarray(result.x, dtype=np.float64).reshape(
            problem.initial_plan.shape
        )
    except (AttributeError, TypeError, ValueError):
        return "fallback_solver_failure", None, None
    if not np.isfinite(candidate).all():
        return "fallback_solver_failure", None, None
    try:
        objective, margins = _evaluate(problem, candidate)
    except _PredictionFailure:
        return "fallback_prediction_failure", None, None
    bound_margin = min(
        float(np.min(candidate - problem.lower)),
        float(np.min(problem.upper - candidate)),
    )
    if min(float(np.min(margins)), bound_margin) < -policy.feasibility_tolerance:
        return "fallback_infeasible", objective, None
    if not result.success:
        return "fallback_solver_failure", objective, None
    if objective >= fallback_cost - 1e-9:
        return "fallback_no_benefit", objective, None
    return "optimized", objective, _frozen(candidate)


def search_bounded_candidates(
    problem: CandidateSearchProblem,
    policy: CandidateSearchPolicy,
    *,
    clock: Callable[[], float] = time.perf_counter,
    started: float | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    solver: Solver = minimize,
) -> CandidateSearchReceipt:
    """Run existing SLSQP search; never qualify or execute a physical input."""
    start = float(clock()) if started is None else started
    try:
        fallback_cost, fallback_margins = _evaluate(problem, problem.fallback_plan)
    except _PredictionFailure as exc:
        raise ValueError(
            "fallback evaluation requires finite cost and margins"
        ) from exc
    fallback_feasible = bool(np.min(fallback_margins) >= -policy.feasibility_tolerance)
    evaluations = 0
    objective: float | None = None
    candidate: Array | None = None

    def check_budget() -> None:
        if cancel_requested is not None and cancel_requested():
            raise _SolveCancelled
        if evaluations >= policy.max_evaluations or clock() - start > policy.max_wall_s:
            raise _BudgetExpired

    def evaluate(flat: Array) -> tuple[float, Array]:
        nonlocal evaluations
        check_budget()
        evaluations += 1
        try:
            return _evaluate(
                problem, np.asarray(flat).reshape(problem.initial_plan.shape)
            )
        except (TypeError, ValueError) as exc:
            raise _PredictionFailure from exc

    def constraint_margin(flat: Array) -> Array:
        return evaluate(flat)[1]

    try:
        if cancel_requested is not None and cancel_requested():
            raise _SolveCancelled
        lower = np.tile(problem.lower, problem.initial_plan.shape[0])
        upper = np.tile(problem.upper, problem.initial_plan.shape[0])
        constraint = NonlinearConstraint(constraint_margin, lb=0.0, ub=np.inf)
        result = solver(
            lambda flat: evaluate(flat)[0],
            problem.initial_plan.ravel(),
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=constraint,
            options=_slsqp_options(policy.solver_max_iterations),
        )
        check_budget()
        status, objective, candidate = _admit_result(
            problem, policy, result, fallback_cost
        )
    except _BudgetExpired:
        status = "fallback_timeout"
    except _SolveCancelled:
        status = "fallback_cancelled"
    except _PredictionFailure:
        status = "fallback_prediction_failure"
    return CandidateSearchReceipt(
        status,
        candidate,
        objective,
        fallback_cost,
        fallback_feasible,
        evaluations,
        max(0.0, float(clock()) - start),
    )
