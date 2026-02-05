"""Model Predictive Control implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


@dataclass
class CostFunction:
    """MPC cost function specification.

    Defines quadratic cost:
        Running: l(x,u) = (x-x_ref)^T Q (x-x_ref) + (u-u_ref)^T R (u-u_ref)
        Terminal: V(x) = (x-x_ref)^T P (x-x_ref)

    Attributes:
        Q: State cost matrix (n_x, n_x).
        R: Control cost matrix (n_u, n_u).
        P: Terminal cost matrix (optional).
        q: Linear state cost (optional).
        r: Linear control cost (optional).
        p: Linear terminal cost (optional).
        x_ref: Reference state trajectory (optional).
        u_ref: Reference control trajectory (optional).
    """

    Q: NDArray[np.floating]
    R: NDArray[np.floating]
    P: NDArray[np.floating] | None = None
    q: NDArray[np.floating] | None = None
    r: NDArray[np.floating] | None = None
    p: NDArray[np.floating] | None = None
    x_ref: NDArray[np.floating] | None = None
    u_ref: NDArray[np.floating] | None = None

    def evaluate_running_cost(
        self,
        x: NDArray[np.floating],
        u: NDArray[np.floating],
        k: int = 0,
    ) -> float:
        """Evaluate running cost at timestep k.

        Args:
            x: Current state.
            u: Current control.
            k: Timestep index.

        Returns:
            Running cost value.
        """
        # State cost
        x_err = x
        if self.x_ref is not None:
            if self.x_ref.ndim == 1:
                x_err = x - self.x_ref
            else:
                x_err = x - self.x_ref[k] if k < len(self.x_ref) else x - self.x_ref[-1]

        state_cost = float(x_err @ self.Q @ x_err)

        if self.q is not None:
            state_cost += float(self.q @ x_err)

        # Control cost
        u_err = u
        if self.u_ref is not None:
            if self.u_ref.ndim == 1:
                u_err = u - self.u_ref
            else:
                u_err = u - self.u_ref[k] if k < len(self.u_ref) else u - self.u_ref[-1]

        control_cost = float(u_err @ self.R @ u_err)

        if self.r is not None:
            control_cost += float(self.r @ u_err)

        return state_cost + control_cost

    def evaluate_terminal_cost(self, x: NDArray[np.floating]) -> float:
        """Evaluate terminal cost.

        Args:
            x: Terminal state.

        Returns:
            Terminal cost value.
        """
        if self.P is None:
            return 0.0

        x_err = x
        if self.x_ref is not None:
            if self.x_ref.ndim == 1:
                x_err = x - self.x_ref
            else:
                x_err = x - self.x_ref[-1]

        cost = float(x_err @ self.P @ x_err)

        if self.p is not None:
            cost += float(self.p @ x_err)

        return cost


@dataclass
class Constraint:
    """MPC constraint specification.

    Represents constraint: lb <= A @ x + B @ u <= ub

    Attributes:
        A: State coefficient matrix.
        B: Control coefficient matrix.
        lb: Lower bound.
        ub: Upper bound.
        constraint_type: "state", "control", or "mixed".
    """

    A: NDArray[np.floating] | None = None
    B: NDArray[np.floating] | None = None
    lb: NDArray[np.floating] | None = None
    ub: NDArray[np.floating] | None = None
    constraint_type: str = "mixed"


@dataclass
class MPCResult:
    """Result of MPC optimization.

    Attributes:
        success: Whether optimization succeeded.
        optimal_states: Optimal state trajectory (N+1, n_x).
        optimal_controls: Optimal control sequence (N, n_u).
        cost: Total optimal cost.
        solve_time: Solver computation time.
        iterations: Number of solver iterations.
        constraint_violations: Maximum constraint violation.
    """

    success: bool
    optimal_states: NDArray[np.floating] | None
    optimal_controls: NDArray[np.floating] | None
    cost: float
    solve_time: float = 0.0
    iterations: int = 0
    constraint_violations: float = 0.0


class ModelPredictiveController:
    """Nonlinear Model Predictive Control.

    Solves optimal control problems over a receding horizon:
        min sum_{k=0}^{N-1} l(x_k, u_k) + V(x_N)
        s.t. x_{k+1} = f(x_k, u_k)
             constraints

    Attributes:
        model: Physics engine for dynamics.
        horizon: Prediction horizon (number of steps).
        dt: Timestep for discretization.
    """

    def __init__(
        self,
        model: PhysicsEngineProtocol,
        horizon: int = 20,
        dt: float = 0.01,
    ) -> None:
        """Initialize MPC controller.

        Args:
            model: Physics engine for dynamics simulation.
            horizon: Number of prediction steps.
            dt: Timestep in seconds.
        """
        self.model = model
        self.horizon = horizon
        self.dt = dt

        self._cost: CostFunction | None = None
        self._constraints: list[Constraint] = []

        # Get dimensions from model
        if hasattr(model, "n_q"):
            self._n_x = model.n_q * 2  # pos + vel
        else:
            self._n_x = 14  # Default 7-DOF arm

        if hasattr(model, "n_v"):
            self._n_u = model.n_v
        else:
            self._n_u = 7

        # Solver settings
        self._max_iterations = 100
        self._tolerance = 1e-6
        self._line_search_alpha = 0.5
        self._line_search_beta = 0.5

    @property
    def n_states(self) -> int:
        """Number of state variables."""
        return self._n_x

    @property
    def n_controls(self) -> int:
        """Number of control variables."""
        return self._n_u

    def set_cost_function(self, cost: CostFunction) -> None:
        """Set the running and terminal cost.

        Args:
            cost: Cost function specification.
        """
        self._cost = cost

    def set_constraints(self, constraints: list[Constraint]) -> None:
        """Set state and control constraints.

        Args:
            constraints: List of constraint specifications.
        """
        self._constraints = constraints

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a single constraint.

        Args:
            constraint: Constraint to add.
        """
        self._constraints.append(constraint)

    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self._constraints.clear()

    def _dynamics(
        self,
        x: NDArray[np.floating],
        u: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate dynamics x_{k+1} = f(x_k, u_k).

        Args:
            x: Current state.
            u: Current control.

        Returns:
            Next state.
        """
        n_q = self._n_x // 2
        q = x[:n_q]
        v = x[n_q:]

        # Set state in model
        if hasattr(self.model, "set_joint_positions"):
            self.model.set_joint_positions(q)
        if hasattr(self.model, "set_joint_velocities"):
            self.model.set_joint_velocities(v)
        if hasattr(self.model, "set_joint_torques"):
            self.model.set_joint_torques(u)

        # Step simulation
        if hasattr(self.model, "step"):
            self.model.step(self.dt)

        # Get new state
        if hasattr(self.model, "get_joint_positions"):
            q_next = self.model.get_joint_positions()
        else:
            q_next = q + v * self.dt

        if hasattr(self.model, "get_joint_velocities"):
            v_next = self.model.get_joint_velocities()
        else:
            v_next = v

        return np.concatenate([q_next, v_next])

    def _dynamics_linearize(
        self,
        x: NDArray[np.floating],
        u: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Linearize dynamics around (x, u).

        Args:
            x: State linearization point.
            u: Control linearization point.

        Returns:
            Tuple of (A, B) matrices.
        """
        eps = 1e-5
        A = np.zeros((self._n_x, self._n_x))
        B = np.zeros((self._n_x, self._n_u))

        x_next = self._dynamics(x, u)

        # Numerical differentiation for A
        for i in range(self._n_x):
            x_plus = x.copy()
            x_plus[i] += eps
            A[:, i] = (self._dynamics(x_plus, u) - x_next) / eps

        # Numerical differentiation for B
        for i in range(self._n_u):
            u_plus = u.copy()
            u_plus[i] += eps
            B[:, i] = (self._dynamics(x, u_plus) - x_next) / eps

        return A, B

    def solve(
        self,
        initial_state: NDArray[np.floating],
        reference_trajectory: NDArray[np.floating] | None = None,
    ) -> MPCResult:
        """Solve MPC problem.

        Uses iterative Linear Quadratic Regulator (iLQR) algorithm.

        Args:
            initial_state: Initial state x_0.
            reference_trajectory: Optional reference trajectory.

        Returns:
            MPC result with optimal trajectory and controls.
        """
        import time

        start_time = time.perf_counter()

        if self._cost is None:
            raise ValueError("Cost function not set")

        # Update reference if provided
        if reference_trajectory is not None:
            self._cost.x_ref = reference_trajectory

        # Initialize trajectory with zero controls
        X = np.zeros((self.horizon + 1, self._n_x))
        U = np.zeros((self.horizon, self._n_u))
        X[0] = initial_state

        # Forward rollout with initial controls
        for k in range(self.horizon):
            X[k + 1] = self._dynamics(X[k], U[k])

        # iLQR iterations
        converged = False
        prev_cost = float("inf")

        for iteration in range(self._max_iterations):  # noqa: B007
            # Backward pass - compute gains
            K, d = self._backward_pass(X, U)

            # Forward pass - apply controls with line search
            X_new, U_new, cost = self._forward_pass(X, U, K, d)

            # Check convergence
            cost_reduction = prev_cost - cost
            if abs(cost_reduction) < self._tolerance:
                converged = True
                X, U = X_new, U_new
                break

            # Accept step if cost decreased
            if cost < prev_cost:
                X, U = X_new, U_new
                prev_cost = cost

        solve_time = time.perf_counter() - start_time

        # Check constraints
        max_violation = self._compute_constraint_violations(X, U)

        return MPCResult(
            success=converged,
            optimal_states=X,
            optimal_controls=U,
            cost=prev_cost,
            solve_time=solve_time,
            iterations=iteration + 1,
            constraint_violations=max_violation,
        )

    def _backward_pass(
        self,
        X: NDArray[np.floating],
        U: NDArray[np.floating],
    ) -> tuple[list[NDArray[np.floating]], list[NDArray[np.floating]]]:
        """Backward pass of iLQR to compute feedback gains.

        Args:
            X: State trajectory.
            U: Control trajectory.

        Returns:
            Tuple of (gains K, feedforward d).
        """
        K = []
        d = []

        # Terminal cost gradient and Hessian
        Vx = (
            2 * self._cost.P @ X[-1]
            if self._cost.P is not None
            else np.zeros(self._n_x)
        )
        Vxx = 2 * self._cost.P if self._cost.P is not None else np.eye(self._n_x) * 0.01

        for k in range(self.horizon - 1, -1, -1):
            # Linearize dynamics
            A, B = self._dynamics_linearize(X[k], U[k])

            # Cost gradients
            x_err = X[k] - (
                self._cost.x_ref[k]
                if self._cost.x_ref is not None and self._cost.x_ref.ndim > 1
                else np.zeros(self._n_x)
            )
            u_err = U[k] - (
                self._cost.u_ref[k]
                if self._cost.u_ref is not None and self._cost.u_ref.ndim > 1
                else np.zeros(self._n_u)
            )

            lx = 2 * self._cost.Q @ x_err
            lu = 2 * self._cost.R @ u_err
            lxx = 2 * self._cost.Q
            luu = 2 * self._cost.R
            lux = np.zeros((self._n_u, self._n_x))

            # Q-function approximation
            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            # Regularization for numerical stability
            Quu_reg = Quu + np.eye(self._n_u) * 1e-6

            # Compute gains
            Quu_inv = np.linalg.inv(Quu_reg)
            K_k = -Quu_inv @ Qux
            d_k = -Quu_inv @ Qu

            K.insert(0, K_k)
            d.insert(0, d_k)

            # Update value function
            Vx = Qx + K_k.T @ Quu @ d_k + K_k.T @ Qu + Qux.T @ d_k
            Vxx = Qxx + K_k.T @ Quu @ K_k + K_k.T @ Qux + Qux.T @ K_k

        return K, d

    def _forward_pass(
        self,
        X: NDArray[np.floating],
        U: NDArray[np.floating],
        K: list[NDArray[np.floating]],
        d: list[NDArray[np.floating]],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
        """Forward pass with line search.

        Args:
            X: Current state trajectory.
            U: Current control trajectory.
            K: Feedback gains.
            d: Feedforward terms.

        Returns:
            Tuple of (new states, new controls, cost).
        """
        alpha = 1.0
        best_cost = float("inf")
        best_X = X.copy()
        best_U = U.copy()

        for _ in range(10):  # Line search iterations
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = X[0]

            for k in range(self.horizon):
                dx = X_new[k] - X[k]
                U_new[k] = U[k] + alpha * d[k] + K[k] @ dx
                X_new[k + 1] = self._dynamics(X_new[k], U_new[k])

            # Compute cost
            cost = 0.0
            for k in range(self.horizon):
                cost += self._cost.evaluate_running_cost(X_new[k], U_new[k], k)
            cost += self._cost.evaluate_terminal_cost(X_new[-1])

            if cost < best_cost:
                best_cost = cost
                best_X = X_new
                best_U = U_new

            alpha *= self._line_search_beta

        return best_X, best_U, best_cost

    def _compute_constraint_violations(
        self,
        X: NDArray[np.floating],
        U: NDArray[np.floating],
    ) -> float:
        """Compute maximum constraint violation.

        Args:
            X: State trajectory.
            U: Control trajectory.

        Returns:
            Maximum violation.
        """
        max_violation = 0.0

        for constraint in self._constraints:
            for k in range(self.horizon):
                value = np.zeros(1)

                if constraint.A is not None:
                    value += constraint.A @ X[k]
                if constraint.B is not None:
                    value += constraint.B @ U[k]

                if constraint.lb is not None:
                    lb_violation = np.maximum(0, constraint.lb - value)
                    max_violation = max(max_violation, float(np.max(lb_violation)))

                if constraint.ub is not None:
                    ub_violation = np.maximum(0, value - constraint.ub)
                    max_violation = max(max_violation, float(np.max(ub_violation)))

        return max_violation

    def get_first_control(self, result: MPCResult) -> NDArray[np.floating]:
        """Get first control input for receding horizon.

        Args:
            result: MPC solution result.

        Returns:
            First control input u_0.
        """
        if result.optimal_controls is None:
            return np.zeros(self._n_u)
        return result.optimal_controls[0]
