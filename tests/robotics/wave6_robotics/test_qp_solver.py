"""Tests for whole_body/qp_solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.control.whole_body.qp_solver import (
    NullspaceQPSolver,
    QPProblem,
    QPSolution,
    ScipyQPSolver,
    create_default_solver,
)


def test_qp_problem_basic_validation() -> None:
    p = QPProblem(H=np.eye(3), g=np.zeros(3))
    assert p.n_vars == 3
    assert p.n_eq == 0
    assert p.n_ineq == 0


def test_qp_problem_bad_h_shape() -> None:
    with pytest.raises(ValueError, match="square"):
        QPProblem(H=np.zeros((3, 4)), g=np.zeros(3))


def test_qp_problem_bad_g() -> None:
    with pytest.raises(ValueError, match="g shape"):
        QPProblem(H=np.eye(3), g=np.zeros(5))


def test_qp_problem_aeq_without_beq() -> None:
    with pytest.raises(ValueError, match="b_eq"):
        QPProblem(H=np.eye(3), g=np.zeros(3), A_eq=np.zeros((1, 3)))


def test_qp_problem_aeq_col_mismatch() -> None:
    with pytest.raises(ValueError, match="A_eq"):
        QPProblem(
            H=np.eye(3),
            g=np.zeros(3),
            A_eq=np.zeros((1, 5)),
            b_eq=np.zeros(1),
        )


def test_qp_problem_aineq_col_mismatch() -> None:
    with pytest.raises(ValueError, match="A_ineq"):
        QPProblem(H=np.eye(3), g=np.zeros(3), A_ineq=np.zeros((1, 5)))


def test_qp_problem_with_full_constraints() -> None:
    p = QPProblem(
        H=np.eye(3),
        g=np.zeros(3),
        A_eq=np.zeros((2, 3)),
        b_eq=np.zeros(2),
        A_ineq=np.zeros((1, 3)),
        lb_ineq=np.zeros(1),
        ub_ineq=np.ones(1),
    )
    assert p.n_eq == 2
    assert p.n_ineq == 1


def test_nullspace_solver_unconstrained() -> None:
    s = NullspaceQPSolver()
    # min 0.5*||x||^2 + g^T x; optimum at x = -g
    p = QPProblem(H=np.eye(3), g=np.array([1.0, 2.0, 3.0]))
    sol = s.solve(p)
    assert sol.success
    assert np.allclose(sol.x, [-1.0, -2.0, -3.0], atol=1e-3)
    assert s.is_available()


def test_nullspace_solver_equality() -> None:
    s = NullspaceQPSolver()
    # Minimize ||x||^2 s.t. x[0] = 1
    A = np.array([[1.0, 0.0, 0.0]])
    b = np.array([1.0])
    p = QPProblem(H=np.eye(3), g=np.zeros(3), A_eq=A, b_eq=b)
    sol = s.solve(p)
    assert sol.success
    assert sol.x is not None
    assert sol.x[0] == pytest.approx(1.0, abs=1e-6)
    assert sol.x[1] == pytest.approx(0.0, abs=1e-6)


def test_nullspace_solver_with_bounds_clamps() -> None:
    s = NullspaceQPSolver()
    p = QPProblem(
        H=np.eye(2),
        g=np.array([10.0, -10.0]),  # unconstrained optimum: [-10, 10]
        x_lb=np.array([-1.0, -1.0]),
        x_ub=np.array([1.0, 1.0]),
    )
    sol = s.solve(p)
    assert sol.success
    assert np.all(sol.x >= -1.0 - 1e-6)
    assert np.all(sol.x <= 1.0 + 1e-6)


def test_nullspace_solver_singular_kkt() -> None:
    s = NullspaceQPSolver(regularization=0.0)
    # Rank-deficient H + rank-deficient A => possibly singular KKT
    H = np.zeros((2, 2))
    A = np.zeros((1, 2))
    b = np.array([1.0])
    p = QPProblem(H=H, g=np.zeros(2), A_eq=A, b_eq=b)
    sol = s.solve(p)
    assert isinstance(sol, QPSolution)


def test_scipy_solver_constructor() -> None:
    s = ScipyQPSolver(method="SLSQP", max_iter=50)
    # is_available may be True or False; the API exists either way
    assert isinstance(s.is_available(), bool)


def test_scipy_solver_solves_simple() -> None:
    s = ScipyQPSolver(method="SLSQP", max_iter=50)
    if not s.is_available():
        pytest.skip("scipy not available")
    p = QPProblem(H=np.eye(2), g=np.array([1.0, 2.0]))
    sol = s.solve(p)
    assert isinstance(sol, QPSolution)


def test_scipy_solver_with_bounds_and_ineq() -> None:
    s = ScipyQPSolver(method="SLSQP", max_iter=50)
    if not s.is_available():
        pytest.skip("scipy not available")
    p = QPProblem(
        H=np.eye(2),
        g=np.zeros(2),
        A_ineq=np.eye(2),
        lb_ineq=np.array([-1.0, -1.0]),
        ub_ineq=np.array([1.0, 1.0]),
        x_lb=np.array([-2.0, -2.0]),
        x_ub=np.array([2.0, 2.0]),
    )
    sol = s.solve(p)
    assert isinstance(sol, QPSolution)


def test_create_default_solver_returns_solver() -> None:
    s = create_default_solver()
    assert s.is_available()
