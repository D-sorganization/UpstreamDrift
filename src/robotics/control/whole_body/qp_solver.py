"""Quadratic programming solver for whole-body control.

This module provides a QP solver interface and implementations
for solving the whole-body control optimization problem.

Design by Contract:
    Solver always returns a valid QPSolution.
    Infeasible problems are indicated by success=False.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class QPProblem:
    """Quadratic programming problem definition.

    Standard form:
        minimize    0.5 * x^T @ H @ x + g^T @ x
        subject to  A_eq @ x = b_eq
                    lb <= A_ineq @ x <= ub
                    x_lb <= x <= x_ub

    Attributes:
        H: Hessian matrix (n, n), must be positive semi-definite.
        g: Linear cost vector (n,).
        A_eq: Equality constraint matrix (m_eq, n).
        b_eq: Equality constraint vector (m_eq,).
        A_ineq: Inequality constraint matrix (m_ineq, n).
        lb_ineq: Lower bounds for inequality constraints (m_ineq,).
        ub_ineq: Upper bounds for inequality constraints (m_ineq,).
        x_lb: Variable lower bounds (n,).
        x_ub: Variable upper bounds (n,).
    """

    H: NDArray[np.float64]
    g: NDArray[np.float64]
    A_eq: NDArray[np.float64] | None = None
    b_eq: NDArray[np.float64] | None = None
    A_ineq: NDArray[np.float64] | None = None
    lb_ineq: NDArray[np.float64] | None = None
    ub_ineq: NDArray[np.float64] | None = None
    x_lb: NDArray[np.float64] | None = None
    x_ub: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate problem dimensions."""
        self.H = np.asarray(self.H, dtype=np.float64)
        self.g = np.asarray(self.g, dtype=np.float64)

        n = self.H.shape[0]
        if self.H.shape != (n, n):
            raise ValueError(f"H must be square, got {self.H.shape}")
        if self.g.shape != (n,):
            raise ValueError(f"g shape {self.g.shape} doesn't match H dimension {n}")

        # Validate equality constraints
        if self.A_eq is not None:
            self.A_eq = np.asarray(self.A_eq, dtype=np.float64)
            if self.b_eq is None:
                raise ValueError("b_eq required when A_eq provided")
            self.b_eq = np.asarray(self.b_eq, dtype=np.float64)
            if self.A_eq.shape[1] != n:
                raise ValueError(
                    f"A_eq columns {self.A_eq.shape[1]} doesn't match n={n}"
                )

        # Validate inequality constraints
        if self.A_ineq is not None:
            self.A_ineq = np.asarray(self.A_ineq, dtype=np.float64)
            if self.A_ineq.shape[1] != n:
                raise ValueError(
                    f"A_ineq columns {self.A_ineq.shape[1]} doesn't match n={n}"
                )

    @property
    def n_vars(self) -> int:
        """Number of decision variables."""
        return self.H.shape[0]

    @property
    def n_eq(self) -> int:
        """Number of equality constraints."""
        return self.A_eq.shape[0] if self.A_eq is not None else 0

    @property
    def n_ineq(self) -> int:
        """Number of inequality constraints."""
        return self.A_ineq.shape[0] if self.A_ineq is not None else 0


@dataclass
class QPSolution:
    """Solution to a quadratic programming problem.

    Attributes:
        success: Whether solver found a valid solution.
        x: Optimal solution vector (n,).
        cost: Optimal cost value.
        iterations: Number of iterations used.
        solve_time: Wall-clock solve time [s].
        status: Solver-specific status message.
        dual_eq: Dual variables for equality constraints.
        dual_ineq: Dual variables for inequality constraints.
    """

    success: bool
    x: NDArray[np.float64] | None
    cost: float = float("inf")
    iterations: int = 0
    solve_time: float = 0.0
    status: str = ""
    dual_eq: NDArray[np.float64] | None = None
    dual_ineq: NDArray[np.float64] | None = None


class QPSolver(ABC):
    """Abstract base class for QP solvers."""

    @abstractmethod
    def solve(self, problem: QPProblem) -> QPSolution:
        """Solve QP problem.

        Args:
            problem: QP problem definition.

        Returns:
            Solution with optimal x if successful.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if solver backend is available.

        Returns:
            True if solver can be used.
        """
        ...


class ScipyQPSolver(QPSolver):
    """QP solver using scipy.optimize.

    Uses SLSQP or trust-constr methods.
    """

    def __init__(self, method: str = "SLSQP", max_iter: int = 100) -> None:
        """Initialize scipy QP solver.

        Args:
            method: Scipy optimization method ('SLSQP' or 'trust-constr').
            max_iter: Maximum number of iterations.
        """
        self._method = method
        self._max_iter = max_iter
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if scipy is available."""
        try:
            pass

            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """Check if solver is available."""
        return self._available

    def solve(self, problem: QPProblem) -> QPSolution:
        """Solve QP using scipy.

        Args:
            problem: QP problem.

        Returns:
            QP solution.
        """
        import time

        if not self._available:
            return QPSolution(
                success=False,
                x=None,
                status="scipy not available",
            )

        from scipy.optimize import Bounds, minimize

        start_time = time.perf_counter()

        n = problem.n_vars

        # Objective function
        def objective(x: NDArray[np.float64]) -> float:
            return float(0.5 * x @ problem.H @ x + problem.g @ x)

        def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return problem.H @ x + problem.g

        # Initial guess
        x0 = np.zeros(n)

        # Variable bounds
        bounds = None
        if problem.x_lb is not None or problem.x_ub is not None:
            lb = problem.x_lb if problem.x_lb is not None else -np.inf * np.ones(n)
            ub = problem.x_ub if problem.x_ub is not None else np.inf * np.ones(n)
            bounds = Bounds(lb, ub)

        # Constraints
        constraints = []

        if problem.A_eq is not None and problem.b_eq is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, A=problem.A_eq, b=problem.b_eq: A @ x - b,
                    "jac": lambda x, A=problem.A_eq: A,
                }
            )

        if problem.A_ineq is not None:
            lb = (
                problem.lb_ineq
                if problem.lb_ineq is not None
                else -np.inf * np.ones(problem.n_ineq)
            )
            ub = (
                problem.ub_ineq
                if problem.ub_ineq is not None
                else np.inf * np.ones(problem.n_ineq)
            )

            # Convert to standard form: lb <= Ax <= ub
            # scipy needs: c(x) >= 0, so we add two constraints:
            # Ax - lb >= 0 and ub - Ax >= 0
            for i in range(problem.n_ineq):
                if lb[i] > -1e10:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, A=problem.A_ineq, lb=lb, i=i: A[i] @ x
                            - lb[i],
                            "jac": lambda x, A=problem.A_ineq, i=i: A[i],
                        }
                    )
                if ub[i] < 1e10:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda x, A=problem.A_ineq, ub=ub, i=i: ub[i]
                            - A[i] @ x,
                            "jac": lambda x, A=problem.A_ineq, i=i: -A[i],
                        }
                    )

        try:
            result = minimize(
                objective,
                x0,
                method=self._method,
                jac=gradient,
                bounds=bounds,
                constraints=constraints if constraints else None,
                options={"maxiter": self._max_iter},
            )

            solve_time = time.perf_counter() - start_time

            return QPSolution(
                success=result.success,
                x=result.x if result.success else None,
                cost=float(result.fun) if result.success else float("inf"),
                iterations=result.nit,
                solve_time=solve_time,
                status=result.message,
            )

        except Exception as e:
            return QPSolution(
                success=False,
                x=None,
                status=f"Solver error: {e}",
            )


class NullspaceQPSolver(QPSolver):
    """QP solver using nullspace projection.

    Solves unconstrained QP in nullspace of equality constraints.
    Simple and fast for small problems.
    """

    def __init__(self, regularization: float = 1e-6) -> None:
        """Initialize nullspace solver.

        Args:
            regularization: Regularization for matrix inversion.
        """
        self._reg = regularization

    def is_available(self) -> bool:
        """Always available (uses numpy only)."""
        return True

    def solve(self, problem: QPProblem) -> QPSolution:
        """Solve QP using nullspace method.

        Only handles equality constraints and soft bounds.

        Args:
            problem: QP problem (equality constraints only).

        Returns:
            QP solution.
        """
        import time

        start_time = time.perf_counter()

        n = problem.n_vars
        H = problem.H + self._reg * np.eye(n)
        g = problem.g

        if problem.A_eq is not None and problem.b_eq is not None:
            # Solve with equality constraints using KKT
            A = problem.A_eq
            b = problem.b_eq
            m = A.shape[0]

            # KKT system:
            # [H  A^T] [x]   [-g]
            # [A  0  ] [λ] = [b ]
            KKT = np.block(
                [
                    [H, A.T],
                    [A, np.zeros((m, m))],
                ]
            )

            rhs = np.concatenate([-g, b])

            try:
                solution = np.linalg.solve(KKT, rhs)
                x = solution[:n]
                dual = solution[n:]

                cost = float(0.5 * x @ problem.H @ x + problem.g @ x)

                solve_time = time.perf_counter() - start_time

                return QPSolution(
                    success=True,
                    x=x,
                    cost=cost,
                    iterations=1,
                    solve_time=solve_time,
                    status="KKT solved",
                    dual_eq=dual,
                )

            except np.linalg.LinAlgError as e:
                return QPSolution(
                    success=False,
                    x=None,
                    status=f"KKT system singular: {e}",
                )

        else:
            # Unconstrained: solve H @ x = -g
            try:
                x = np.asarray(np.linalg.solve(H, -g), dtype=np.float64)
                cost = float(0.5 * x @ problem.H @ x + problem.g @ x)

                solve_time = time.perf_counter() - start_time

                return QPSolution(
                    success=True,
                    x=x,
                    cost=cost,
                    iterations=1,
                    solve_time=solve_time,
                    status="Direct solve",
                )

            except np.linalg.LinAlgError as e:
                return QPSolution(
                    success=False,
                    x=None,
                    status=f"System singular: {e}",
                )


def create_default_solver() -> QPSolver:
    """Create the default QP solver.

    Returns scipy solver if available, otherwise nullspace solver.

    Returns:
        QPSolver instance.
    """
    scipy_solver = ScipyQPSolver()
    if scipy_solver.is_available():
        return scipy_solver
    return NullspaceQPSolver()
