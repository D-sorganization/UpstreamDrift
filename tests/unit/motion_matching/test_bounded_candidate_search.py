"""Contract tests for one bounded shooting loop shared by control adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.bounded_candidate_search import (
    CandidateSearchProblem,
    CandidateSearchPolicy,
    search_bounded_candidates,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _problem() -> CandidateSearchProblem:
    def evaluate(plan: np.ndarray) -> tuple[float, np.ndarray]:
        effort = float(plan[0, 0])
        return (effort - 0.3) ** 2, np.array([effort - 0.1])

    return CandidateSearchProblem(
        initial_plan=np.array([[0.1]]),
        fallback_plan=np.array([[0.1]]),
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        evaluate=evaluate,
    )


def _policy() -> CandidateSearchPolicy:
    return CandidateSearchPolicy(100, 1.0, 50, 1e-8)


def test_solver_accepts_only_fresh_feasible_improvement() -> None:
    receipt = search_bounded_candidates(_problem(), _policy())
    assert receipt.status == "optimized"
    assert receipt.candidate_plan is not None
    assert receipt.candidate_plan[0, 0] == pytest.approx(0.3, abs=1e-5)
    assert receipt.objective is not None
    assert receipt.objective < receipt.fallback_objective
    assert receipt.evaluations <= _policy().max_evaluations
    with pytest.raises(ValueError):
        receipt.candidate_plan[0, 0] = 0.0


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (-0.2, "fallback_infeasible"),
        (1.2, "fallback_infeasible"),
        (float("nan"), "fallback_solver_failure"),
    ],
)
def test_solver_result_cannot_bypass_independent_admission(
    candidate: float, expected: str
) -> None:
    def solver(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=True, x=np.array([candidate]))

    receipt = search_bounded_candidates(_problem(), _policy(), solver=solver)
    assert receipt.status == expected
    assert receipt.candidate_plan is None


def test_solver_success_flag_and_numerical_budget_cannot_be_bypassed() -> None:
    def unsuccessful(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(success=False, x=np.array([0.3]))

    failed = search_bounded_candidates(_problem(), _policy(), solver=unsuccessful)
    assert failed.status == "fallback_solver_failure"
    assert failed.candidate_plan is None

    def excessive(objective: object, x0: object, **kwargs: object) -> SimpleNamespace:
        assert callable(objective)
        objective(np.array([0.2]))
        objective(np.array([0.3]))
        return SimpleNamespace(success=True, x=np.array([0.3]))

    budget = CandidateSearchPolicy(1, 1.0, 50, 1e-8)
    expired = search_bounded_candidates(_problem(), budget, solver=excessive)
    assert expired.status == "fallback_timeout"
    assert expired.evaluations == 1
    assert expired.candidate_plan is None


def test_unavailable_prediction_is_an_explicit_closed_failure() -> None:
    def evaluate(plan: np.ndarray) -> tuple[float, np.ndarray]:
        if plan[0, 0] > 0.1:
            raise ValueError("native prediction unavailable")
        return 1.0, np.array([1.0])

    problem = CandidateSearchProblem(
        initial_plan=np.array([[0.1]]),
        fallback_plan=np.array([[0.1]]),
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        evaluate=evaluate,
    )

    def solver(objective: object, x0: object, **kwargs: object) -> SimpleNamespace:
        assert callable(objective)
        objective(np.array([0.3]))
        return SimpleNamespace(success=True, x=np.array([0.3]))

    receipt = search_bounded_candidates(problem, _policy(), solver=solver)
    assert receipt.status == "fallback_prediction_failure"
    assert receipt.candidate_plan is None
    assert receipt.evaluations == 1


def test_fallback_evaluation_itself_must_be_finite() -> None:
    problem = CandidateSearchProblem(
        initial_plan=np.array([[0.1]]),
        fallback_plan=np.array([[0.1]]),
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        evaluate=lambda plan: (float("nan"), np.array([1.0])),
    )
    with pytest.raises(ValueError, match="fallback.*finite"):
        search_bounded_candidates(problem, _policy())


def test_warm_start_and_cancel_respect_predeclared_budget() -> None:
    problem = _problem()
    warm = CandidateSearchProblem(
        initial_plan=np.array([[0.4]]),
        fallback_plan=problem.fallback_plan,
        lower=problem.lower,
        upper=problem.upper,
        evaluate=problem.evaluate,
    )
    seen: list[float] = []

    def solver(objective: object, x0: np.ndarray, **kwargs: object) -> SimpleNamespace:
        seen.extend(x0.tolist())
        return SimpleNamespace(success=True, x=np.array([0.3]))

    receipt = search_bounded_candidates(warm, _policy(), solver=solver)
    assert receipt.status == "optimized"
    assert seen == [0.4]
    cancelled = search_bounded_candidates(
        warm, _policy(), cancel_requested=lambda: True, solver=solver
    )
    assert cancelled.status == "fallback_cancelled"
    assert cancelled.candidate_plan is None
    assert seen == [0.4]


@pytest.mark.parametrize(
    ("last_command", "expected"),
    [(0.8, "fallback_infeasible"), (0.1, "optimized")],
)
def test_callback_enforces_every_step_and_postmapping_channel_slew(
    last_command: float, expected: str
) -> None:
    previous = np.array([0.0, 0.0])
    max_slew = np.array([0.2, 0.2])
    fallback = np.zeros((3, 2))

    def evaluate(commands: np.ndarray) -> tuple[float, np.ndarray]:
        changes = np.diff(np.vstack((previous, commands)), axis=0)
        margins = (max_slew - np.abs(changes)).ravel()
        target = np.zeros_like(commands)
        target[-1, 1] = 0.8
        return float(np.sum((commands - target) ** 2)), margins

    problem = CandidateSearchProblem(
        initial_plan=fallback,
        fallback_plan=fallback,
        lower=np.zeros(2),
        upper=np.ones(2),
        evaluate=evaluate,
    )

    def solver(*args: object, **kwargs: object) -> SimpleNamespace:
        candidate = fallback.copy()
        candidate[-1, 1] = last_command
        return SimpleNamespace(success=True, x=candidate.ravel())

    receipt = search_bounded_candidates(problem, _policy(), solver=solver)
    assert receipt.status == expected
    assert receipt.fallback_horizon_feasible
    if expected == "optimized":
        assert receipt.candidate_plan is not None
        assert receipt.candidate_plan[-1, 1] == last_command
    else:
        assert receipt.candidate_plan is None
