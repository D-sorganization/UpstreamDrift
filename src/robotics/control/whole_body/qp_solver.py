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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import invariant

if TYPE_CHECKING:
    from scipy.optimize import Bounds


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
        self._require_finite("H", self.H)
        self._require_finite("g", self.g)

        # Validate equality constraints
        if self.A_eq is not None:
            self.A_eq = np.asarray(self.A_eq, dtype=np.float64)
            if self.b_eq is None:
                raise ValueError("b_eq required when A_eq provided")
            self.b_eq = np.asarray(self.b_eq, dtype=np.float64)
            if self.A_eq.shape[1] != n:
                raise ValueError(
                    f"A_eq columns {self.A_eq.shape[1]} doesn't match n={n}",
                )
            if self.b_eq.shape != (self.A_eq.shape[0],):
                raise ValueError(
                    f"b_eq shape {self.b_eq.shape} doesn't match A_eq rows "
                    f"{self.A_eq.shape[0]}",
                )
            self._require_finite("A_eq", self.A_eq)
            self._require_finite("b_eq", self.b_eq)
        elif self.b_eq is not None:
            raise ValueError("A_eq required when b_eq provided")

        # Validate inequality constraints
        if self.A_ineq is not None:
            self.A_ineq = np.asarray(self.A_ineq, dtype=np.float64)
            if self.A_ineq.shape[1] != n:
                raise ValueError(
                    f"A_ineq columns {self.A_ineq.shape[1]} doesn't match n={n}",
                )
            self._require_finite("A_ineq", self.A_ineq)
            self._validate_inequality_bounds()
        elif self.lb_ineq is not None or self.ub_ineq is not None:
            raise ValueError("A_ineq required when inequality bounds provided")

        self._validate_variable_bounds(n)

    @staticmethod
    def _require_finite(name: str, values: NDArray[np.float64]) -> None:
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")

    @staticmethod
    def _require_no_nan(name: str, values: NDArray[np.float64]) -> None:
        if np.any(np.isnan(values)):
            raise ValueError(f"{name} must not contain NaN")

    def _validate_inequality_bounds(self) -> None:
        if self.A_ineq is None:
            raise ValueError("A_ineq required when validating inequality bounds")
        n_ineq = self.A_ineq.shape[0]
        if self.lb_ineq is not None:
            self.lb_ineq = np.asarray(self.lb_ineq, dtype=np.float64)
            if self.lb_ineq.shape != (n_ineq,):
                raise ValueError(
                    f"lb_ineq shape {self.lb_ineq.shape} doesn't match A_ineq rows "
                    f"{n_ineq}",
                )
            self._require_no_nan("lb_ineq", self.lb_ineq)
        if self.ub_ineq is not None:
            self.ub_ineq = np.asarray(self.ub_ineq, dtype=np.float64)
            if self.ub_ineq.shape != (n_ineq,):
                raise ValueError(
                    f"ub_ineq shape {self.ub_ineq.shape} doesn't match A_ineq rows "
                    f"{n_ineq}",
                )
            self._require_no_nan("ub_ineq", self.ub_ineq)
        if (
            self.lb_ineq is not None
            and self.ub_ineq is not None
            and np.any(self.lb_ineq > self.ub_ineq)
        ):
            raise ValueError("lb_ineq must be less than or equal to ub_ineq")

    def _validate_variable_bounds(self, n_vars: int) -> None:
        if self.x_lb is not None:
            self.x_lb = np.asarray(self.x_lb, dtype=np.float64)
            if self.x_lb.shape != (n_vars,):
                raise ValueError(
                    f"x_lb shape {self.x_lb.shape} doesn't match H dimension {n_vars}",
                )
            self._require_no_nan("x_lb", self.x_lb)
        if self.x_ub is not None:
            self.x_ub = np.asarray(self.x_ub, dtype=np.float64)
            if self.x_ub.shape != (n_vars,):
                raise ValueError(
                    f"x_ub shape {self.x_ub.shape} doesn't match H dimension {n_vars}",
                )
            self._require_no_nan("x_ub", self.x_ub)
        if (
            self.x_lb is not None
            and self.x_ub is not None
            and np.any(self.x_lb > self.x_ub)
        ):
            raise ValueError("x_lb must be less than or equal to x_ub")

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

    @property
    def solver_status(self) -> str:
        """Backward-compatible canonical solver status."""
        return "success" if self.success else "failure"


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


@invariant(lambda self: self._max_iter > 0, "max_iter must be positive")
@invariant(
    lambda self: self._method in ("SLSQP", "trust-constr"),
    "Solver method must be SLSQP or trust-constr",
)
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
        if method is None:
            raise ValueError("method must be provided")
        self._method = method
        self._max_iter = max_iter
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Check if scipy is available."""
        try:
            from scipy import optimize  # noqa: F401

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
        if problem is None:
            raise ValueError("problem must be provided")
        import time

        if not self._available:
            return QPSolution(
                success=False,
                x=None,
                status="scipy not available",
            )

        from scipy.optimize import minimize

        minimize_qp = cast(Any, minimize)

        start_time = time.perf_counter()

        n = problem.n_vars

        def objective(x: NDArray[np.float64]) -> float:
            """Evaluate the quadratic objective value."""
            return float(0.5 * x @ problem.H @ x + problem.g @ x)

        def gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
            """Compute the gradient of the quadratic objective."""
            return problem.H @ x + problem.g

        x0 = np.zeros(n)
        bounds = self._build_variable_bounds(problem)
        constraints = self._build_constraints(problem)

        try:
            result = minimize_qp(
                objective,
                x0,
                method=self._method,
                jac=gradient,
                bounds=bounds,
                constraints=constraints or None,
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

        except (RuntimeError, ValueError, OSError) as e:
            return QPSolution(
                success=False,
                x=None,
                status=f"Solver error: {e}",
            )

    def _build_variable_bounds(self, problem: QPProblem) -> Bounds | None:  # type: ignore[return]
        if problem is None:
            raise ValueError("problem must be provided")
        from scipy.optimize import Bounds

        if problem.x_lb is None and problem.x_ub is None:
            return None

        n = problem.n_vars
        lb = problem.x_lb if problem.x_lb is not None else -np.inf * np.ones(n)
        ub = problem.x_ub if problem.x_ub is not None else np.inf * np.ones(n)
        return Bounds(lb, ub)

    def _build_constraints(self, problem: QPProblem) -> list[dict[str, Any]]:
        if problem is None:
            raise ValueError("problem must be provided")
        constraints: list[dict[str, Any]] = []

        if problem.A_eq is not None and problem.b_eq is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, A=problem.A_eq, b=problem.b_eq: A @ x - b,
                    "jac": lambda x, A=problem.A_eq: A,
                },
            )

        if problem.A_ineq is not None:
            self._add_inequality_constraints(constraints, problem)

        return constraints

    def _add_inequality_constraints(
        self,
        constraints: list[dict[str, Any]],
        problem: QPProblem,
    ) -> None:
        if constraints is None:
            raise ValueError("constraints must be provided")
        if problem.A_ineq is None:
            raise ValueError("A_ineq required when inequality constraints are added")

        lb = self._bound_vector(problem.lb_ineq, problem.n_ineq, -np.inf, "lb_ineq")
        ub = self._bound_vector(problem.ub_ineq, problem.n_ineq, np.inf, "ub_ineq")

        lower_mask = np.isfinite(lb)
        if np.any(lower_mask):
            self._append_inequality_lower_bound(
                constraints,
                problem.A_ineq[lower_mask],
                lb[lower_mask],
            )

        upper_mask = np.isfinite(ub)
        if np.any(upper_mask):
            self._append_inequality_upper_bound(
                constraints,
                problem.A_ineq[upper_mask],
                ub[upper_mask],
            )

    @staticmethod
    def _bound_vector(
        values: NDArray[np.float64] | None,
        size: int,
        default: float,
        name: str,
    ) -> NDArray[np.float64]:
        if size < 0:
            raise ValueError("size must be non-negative")
        if values is None:
            return np.full(size, default, dtype=np.float64)
        bounds = np.asarray(values, dtype=np.float64)
        if bounds.shape != (size,):
            raise ValueError(f"{name} shape {bounds.shape} doesn't match size {size}")
        if np.any(np.isnan(bounds)):
            raise ValueError(f"{name} must not contain NaN")
        return bounds

    @staticmethod
    def _append_inequality_lower_bound(
        constraints: list[dict[str, Any]],
        A: NDArray[np.float64],
        lb: NDArray[np.float64],
    ) -> None:
        if A.ndim != 2:
            raise ValueError("A must be two-dimensional")
        if lb.shape != (A.shape[0],):
            raise ValueError("lb must match A rows")
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, A=A, lb=lb: A @ x - lb,
                "jac": lambda x, A=A: A,
            },
        )

    @staticmethod
    def _append_inequality_upper_bound(
        constraints: list[dict[str, Any]],
        A: NDArray[np.float64],
        ub: NDArray[np.float64],
    ) -> None:
        if A.ndim != 2:
            raise ValueError("A must be two-dimensional")
        if ub.shape != (A.shape[0],):
            raise ValueError("ub must match A rows")
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, A=A, ub=ub: ub - A @ x,
                "jac": lambda x, A=A: -A,
            },
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
        if regularization is None:
            raise ValueError("regularization must be provided")
        self._reg = regularization

    def is_available(self) -> bool:
        """Always available (uses numpy only)."""
        return True

    def _apply_variable_bounds(self, x: np.ndarray, problem: QPProblem) -> np.ndarray:
        """Clamp x to variable bounds [x_lb, x_ub] if set."""
        if problem.x_lb is not None:
            x = np.maximum(x, problem.x_lb)
        if problem.x_ub is not None:
            x = np.minimum(x, problem.x_ub)
        return x

    def solve(self, problem: QPProblem) -> QPSolution:
        """Solve QP using nullspace method.

        Handles equality constraints via KKT system. Variable bounds
        are enforced by post-solve clamping (best-effort).

        Args:
            problem: QP problem.

        Returns:
            QP solution.
        """
        if problem is None:
            raise ValueError("problem must be provided")
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
                ],
            )

            rhs = np.concatenate([-g, b])

            try:
                solution = np.linalg.solve(KKT, rhs)
                x = self._apply_variable_bounds(solution[:n], problem)
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
                x = self._apply_variable_bounds(
                    np.asarray(np.linalg.solve(H, -g), dtype=np.float64), problem
                )
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
