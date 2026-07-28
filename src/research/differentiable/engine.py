# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Differentiable physics simulation engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


_DEFAULT_FD_REL_STEP = 1e-6


def _as_finite_array(
    name: str,
    values: NDArray[np.floating] | None,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.floating]:
    if values is None:
        raise ValueError(f"{name} must be provided")
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; got shape {array.shape}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_positive_dt(dt: float) -> float:
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be finite and positive")
    return dt_value


def _scaled_central_step(
    value: float,
    rel_step: float = _DEFAULT_FD_REL_STEP,
) -> float:
    rel_step = float(rel_step)
    if not np.isfinite(rel_step) or rel_step <= 0.0:
        raise ValueError("finite-difference relative step must be finite and positive")
    return rel_step * max(1.0, abs(float(value)))


def _finite_loss(
    loss_fn: Callable[[NDArray[np.floating]], float],
    trajectory: NDArray[np.floating],
) -> float:
    loss = float(loss_fn(trajectory))
    if not np.isfinite(loss):
        raise ValueError("loss_fn must return a finite scalar")
    return loss


def _central_difference_vector(
    evaluate: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    point: NDArray[np.floating],
    index: int,
) -> NDArray[np.floating]:
    step = _scaled_central_step(float(point[index]))
    point_plus = point.copy()
    point_minus = point.copy()
    point_plus[index] += step
    point_minus[index] -= step

    value_plus = _as_finite_array(
        "finite-difference plus evaluation",
        evaluate(point_plus),
        ndim=1,
    )
    value_minus = _as_finite_array(
        "finite-difference minus evaluation",
        evaluate(point_minus),
        shape=value_plus.shape,
    )
    return (value_plus - value_minus) / (2.0 * step)


class AutodiffBackend(Enum):
    """Automatic differentiation backend."""

    JAX = "jax"
    TORCH = "torch"
    NUMPY = "numpy"


@dataclass
class OptimizationResult:
    """Result of trajectory optimization.

    Attributes:
        success: Whether optimization converged.
        optimal_states: Optimized state trajectory.
        optimal_controls: Optimized control sequence.
        final_cost: Final cost value.
        iterations: Number of iterations.
        gradient_norm: Final gradient norm.
    """

    success: bool
    optimal_states: NDArray[np.floating]
    optimal_controls: NDArray[np.floating]
    final_cost: float
    iterations: int
    gradient_norm: float

    @property
    def solver_status(self) -> str:
        """Return a canonical solver status derived from the success flag."""
        if self.success:
            return "success"
        return "failed"


class DifferentiableEngine:
    """Differentiable physics simulation.

    Enables gradient-based optimization through physics simulation
    by computing gradients of simulation output with respect to
    initial conditions and control inputs.

    Attributes:
        engine: Underlying physics engine.
        backend: Autodiff backend used.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        backend: str = "numpy",
    ) -> None:
        """Initialize differentiable engine.

        Args:
            engine: Physics engine to wrap.
            backend: Autodiff backend ("jax", "torch", "numpy").
        """
        if engine is None:
            raise ValueError("engine must be provided")
        self.engine = engine
        self._backend = AutodiffBackend(backend)

        # Get state dimensions
        if hasattr(engine, "n_q"):
            self._n_q = engine.n_q
        else:
            self._n_q = 7

        if hasattr(engine, "n_v"):
            self._n_v = engine.n_v
        else:
            self._n_v = self._n_q

        self._n_x = self._n_q + self._n_v
        self._n_u = self._n_v

    def simulate_trajectory(
        self,
        initial_state: NDArray[np.floating],
        controls: NDArray[np.floating],
        dt: float = 0.01,
    ) -> NDArray[np.floating]:
        """Forward simulation returning state trajectory.

        Args:
            initial_state: Initial state [q, v].
            controls: Control sequence (T, n_u).
            dt: Simulation timestep.

        Returns:
            State trajectory (T+1, n_x).
        """
        initial_state = _as_finite_array(
            "initial_state",
            initial_state,
            shape=(self._n_x,),
        )
        controls = _as_finite_array("controls", controls, ndim=2)
        if controls.shape[1] != self._n_u:
            raise ValueError(
                f"controls must have shape (T, {self._n_u}); got {controls.shape}"
            )
        dt = _require_positive_dt(dt)
        T = controls.shape[0]
        trajectory = np.zeros((T + 1, self._n_x), dtype=float)
        trajectory[0] = initial_state

        # Set initial state
        q0 = initial_state[: self._n_q]
        v0 = initial_state[self._n_q :]

        if hasattr(self.engine, "set_joint_positions"):
            self.engine.set_joint_positions(q0)
        if hasattr(self.engine, "set_joint_velocities"):
            self.engine.set_joint_velocities(v0)

        for t in range(T):
            # Apply control
            if hasattr(self.engine, "set_joint_torques"):
                self.engine.set_joint_torques(controls[t])

            # Step simulation
            step = getattr(self.engine, "step", None)
            if callable(step):
                step(dt)

            # Record state
            if hasattr(self.engine, "get_joint_positions"):
                q = self.engine.get_joint_positions()
            else:
                q = trajectory[t, : self._n_q]

            if hasattr(self.engine, "get_joint_velocities"):
                v = self.engine.get_joint_velocities()
            else:
                v = trajectory[t, self._n_q :]

            trajectory[t + 1] = _as_finite_array(
                "simulated state",
                np.concatenate([q, v]),
                shape=(self._n_x,),
            )

        return _as_finite_array("trajectory", trajectory, shape=(T + 1, self._n_x))

    def compute_gradient(
        self,
        initial_state: NDArray[np.floating],
        controls: NDArray[np.floating],
        loss_fn: Callable[[NDArray[np.floating]], float],
        dt: float = 0.01,
    ) -> NDArray[np.floating]:
        """Compute gradient of loss with respect to controls.

        Uses numerical differentiation when autodiff not available.

        Args:
            initial_state: Initial state.
            controls: Control sequence (T, n_u).
            loss_fn: Loss function taking trajectory.
            dt: Simulation timestep.

        Returns:
            Gradient of loss w.r.t. controls (T, n_u).
        """
        initial_state = _as_finite_array(
            "initial_state",
            initial_state,
            shape=(self._n_x,),
        )
        controls = _as_finite_array("controls", controls, ndim=2)
        if controls.shape[1] != self._n_u:
            raise ValueError(
                f"controls must have shape (T, {self._n_u}); got {controls.shape}"
            )
        dt = _require_positive_dt(dt)
        T, n_u = controls.shape
        gradient = np.zeros_like(controls, dtype=float)

        # Baseline trajectory supplies the exact prefix state for suffix rollouts.
        baseline_traj = self.simulate_trajectory(initial_state, controls, dt)

        # Numerical gradient (issues #7557 and #7569).
        #
        # Perturbing ``controls[t, i]`` cannot change any state at or before
        # index ``t`` (the step closure is Markovian in the state vector), so
        # the baseline prefix ``baseline_traj[: t + 1]`` is reused and only the
        # suffix is re-simulated from ``baseline_traj[t]``. Central differences
        # double the suffix evaluations but reduce truncation error from O(h) to
        # O(h^2), and each element uses h = 1e-6 * max(1, abs(value)) to stay
        # scale-aware without exposing a wider public API.
        perturbed = controls.copy()
        traj_plus = baseline_traj.copy()
        traj_minus = baseline_traj.copy()
        for t in range(T):
            for i in range(n_u):
                step = _scaled_central_step(float(controls[t, i]))

                perturbed[t, i] = controls[t, i] + step
                suffix = self.simulate_trajectory(baseline_traj[t], perturbed[t:], dt)
                traj_plus[t:] = suffix
                loss_plus = _finite_loss(loss_fn, traj_plus)

                perturbed[t, i] = controls[t, i] - step
                suffix = self.simulate_trajectory(baseline_traj[t], perturbed[t:], dt)
                traj_minus[t:] = suffix
                loss_minus = _finite_loss(loss_fn, traj_minus)

                gradient[t, i] = (loss_plus - loss_minus) / (2.0 * step)
                perturbed[t, i] = controls[t, i]  # exact restore (no fp drift)

            # Restore this row so later steps expose the unperturbed prefix.
            traj_plus[t] = baseline_traj[t]
            traj_minus[t] = baseline_traj[t]

        return _as_finite_array("gradient", gradient, shape=controls.shape)

    def _set_engine_state(
        self,
        q: NDArray[np.floating],
        v: NDArray[np.floating],
        torques: NDArray[np.floating],
    ) -> None:
        q = _as_finite_array("q", q, shape=(self._n_q,))
        v = _as_finite_array("v", v, shape=(self._n_v,))
        torques = _as_finite_array("torques", torques, shape=(self._n_u,))
        if hasattr(self.engine, "set_joint_positions"):
            self.engine.set_joint_positions(q)
        if hasattr(self.engine, "set_joint_velocities"):
            self.engine.set_joint_velocities(v)
        if hasattr(self.engine, "set_joint_torques"):
            self.engine.set_joint_torques(torques)

    def _step_and_read_state(
        self,
        q: NDArray[np.floating],
        v: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        q = _as_finite_array("q", q, shape=(self._n_q,))
        v = _as_finite_array("v", v, shape=(self._n_v,))
        dt = _require_positive_dt(dt)
        step = getattr(self.engine, "step", None)
        if callable(step):
            step(dt)

        if hasattr(self.engine, "get_joint_positions"):
            q_new = self.engine.get_joint_positions()
        else:
            q_new = q + v * dt

        if hasattr(self.engine, "get_joint_velocities"):
            v_new = self.engine.get_joint_velocities()
        else:
            v_new = v

        return _as_finite_array(
            "next state",
            np.concatenate([q_new, v_new]),
            shape=(self._n_x,),
        )

    def _compute_nominal_next_state(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        state = _as_finite_array("state", state, shape=(self._n_x,))
        control = _as_finite_array("control", control, shape=(self._n_u,))
        dt = _require_positive_dt(dt)
        q = state[: self._n_q]
        v = state[self._n_q :]
        self._set_engine_state(q, v, control)
        return self._step_and_read_state(q, v, dt)

    def _compute_state_jacobian(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        state = _as_finite_array("state", state, shape=(self._n_x,))
        control = _as_finite_array("control", control, shape=(self._n_u,))
        dt = _require_positive_dt(dt)
        A = np.zeros((self._n_x, self._n_x))

        def evaluate(state_candidate: NDArray[np.floating]) -> NDArray[np.floating]:
            return self._compute_nominal_next_state(state_candidate, control, dt)

        for i in range(self._n_x):
            A[:, i] = _central_difference_vector(evaluate, state, i)

        return _as_finite_array("state jacobian", A, shape=(self._n_x, self._n_x))

    def _compute_control_jacobian(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        state = _as_finite_array("state", state, shape=(self._n_x,))
        control = _as_finite_array("control", control, shape=(self._n_u,))
        dt = _require_positive_dt(dt)
        B = np.zeros((self._n_x, self._n_u))

        def evaluate(control_candidate: NDArray[np.floating]) -> NDArray[np.floating]:
            return self._compute_nominal_next_state(state, control_candidate, dt)

        for i in range(self._n_u):
            B[:, i] = _central_difference_vector(evaluate, control, i)

        return _as_finite_array("control jacobian", B, shape=(self._n_x, self._n_u))

    def compute_jacobian(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
        dt: float = 0.01,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute Jacobians of dynamics.

        Args:
            state: Current state.
            control: Current control.
            dt: Timestep.

        Returns:
            Tuple of (df/dx, df/du) Jacobians.
        """
        state = _as_finite_array("state", state, shape=(self._n_x,))
        control = _as_finite_array("control", control, shape=(self._n_u,))
        dt = _require_positive_dt(dt)
        A = self._compute_state_jacobian(state, control, dt)
        B = self._compute_control_jacobian(state, control, dt)
        return A, B

    def optimize_trajectory(
        self,
        initial_state: NDArray[np.floating],
        goal_state: NDArray[np.floating],
        horizon: int,
        dt: float = 0.01,
        method: str = "adam",
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> OptimizationResult:
        """Optimize trajectory to reach goal using gradients.

        Args:
            initial_state: Initial state.
            goal_state: Target goal state.
            horizon: Number of timesteps.
            dt: Timestep.
            method: Optimization method ("adam", "sgd", "lbfgs").
            max_iterations: Maximum optimization iterations.
            learning_rate: Learning rate for gradient descent.

        Returns:
            Optimization result.
        """
        # Initialize controls
        if initial_state is None:
            raise ValueError("initial_state must be provided")
        controls = np.zeros((horizon, self._n_u))

        # Define loss function
        def loss_fn(trajectory: NDArray[np.floating]) -> float:
            """Compute squared error between the final state and the goal."""
            final_state = trajectory[-1]
            state_error = final_state - goal_state
            return float(np.vdot(state_error, state_error))

        # Adam optimizer state
        if method == "adam":
            m = np.zeros_like(controls)
            v = np.zeros_like(controls)
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

        best_loss = float("inf")
        best_controls = controls.copy()
        grad_norm = float("inf")
        iteration = -1

        for iteration in range(max_iterations):
            # Compute gradient
            gradient = self.compute_gradient(initial_state, controls, loss_fn, dt)

            # Compute current loss
            trajectory = self.simulate_trajectory(initial_state, controls, dt)
            current_loss = loss_fn(trajectory)

            # Track best
            if current_loss < best_loss:
                best_loss = current_loss
                best_controls = controls.copy()

            # Check convergence
            grad_norm = float(np.linalg.norm(gradient))
            if grad_norm < 1e-6:
                break

            # Update controls
            if method == "adam":
                m = beta1 * m + (1 - beta1) * gradient
                v = beta2 * v + (1 - beta2) * (gradient**2)
                m_hat = m / (1 - beta1 ** (iteration + 1))
                v_hat = v / (1 - beta2 ** (iteration + 1))
                controls = controls - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            else:
                controls = controls - learning_rate * gradient

        # Final trajectory with best controls
        optimal_trajectory = self.simulate_trajectory(initial_state, best_controls, dt)

        return OptimizationResult(
            success=best_loss < 0.1,
            optimal_states=optimal_trajectory,
            optimal_controls=best_controls,
            final_cost=best_loss,
            iterations=iteration + 1,
            gradient_norm=grad_norm,
        )


class ContactDifferentiableEngine(DifferentiableEngine):
    """Differentiable simulation through contact.

    Handles the non-smooth contact dynamics using smoothing
    or randomized smoothing techniques to enable gradient flow.

    Attributes:
        contact_method: Contact smoothing method.
        smoothing_factor: Smoothing parameter.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        contact_method: str = "smoothed",
        smoothing_factor: float = 0.01,
    ) -> None:
        """Initialize contact-aware differentiable engine.

        Args:
            engine: Physics engine.
            contact_method: "smoothed", "randomized", or "stochastic".
            smoothing_factor: Smoothing parameter.
        """
        if engine is None:
            raise ValueError("engine must be provided")
        super().__init__(engine)
        self.contact_method = contact_method
        self.smoothing_factor = smoothing_factor

    def compute_gradient(
        self,
        initial_state: NDArray[np.floating],
        controls: NDArray[np.floating],
        loss_fn: Callable[[NDArray[np.floating]], float],
        dt: float = 0.01,
    ) -> NDArray[np.floating]:
        """Compute gradient with contact smoothing.

        Args:
            initial_state: Initial state.
            controls: Control sequence.
            loss_fn: Loss function.
            dt: Timestep.

        Returns:
            Smoothed gradient.
        """
        if initial_state is None:
            raise ValueError("initial_state must be provided")
        if self.contact_method == "randomized":
            # Randomized smoothing: average gradients with noise
            n_samples = 10
            gradient = np.zeros_like(controls)

            for _ in range(n_samples):
                # Add noise to controls
                noise = np.random.randn(*controls.shape) * self.smoothing_factor
                controls_noisy = controls + noise

                grad = super().compute_gradient(
                    initial_state,
                    controls_noisy,
                    loss_fn,
                    dt,
                )
                gradient += grad / n_samples

            return gradient

        if self.contact_method == "stochastic":
            # Stochastic gradient with single sample
            noise = np.random.randn(*controls.shape) * self.smoothing_factor
            controls_noisy = controls + noise
            return super().compute_gradient(initial_state, controls_noisy, loss_fn, dt)

        # Standard smoothed gradient
        return super().compute_gradient(initial_state, controls, loss_fn, dt)

    def optimize_through_contact(
        self,
        initial_state: NDArray[np.floating],
        goal_state: NDArray[np.floating],
        contact_schedule: list[bool],
        horizon: int,
        dt: float = 0.01,
        contact_smoothing_multiplier: float = 5.0,
        contact_penalty_weight: float = 0.1,
    ) -> OptimizationResult:
        """Optimize trajectory with specified contact schedule.

        Applies phase-aware smoothing: during contact phases the smoothing
        factor is increased by ``contact_smoothing_multiplier`` to smooth
        the non-differentiable contact dynamics.  A contact-consistency
        penalty is added to discourage velocity discontinuities at
        contact/release transitions:

        .. math::
            C_{contact} = w_c \\sum_{t \\in \\mathcal{T}_{transition}}
                \\| v_{t+1} - v_t \\|^2

        Args:
            initial_state: Initial state [q, v].
            goal_state: Goal state [q, v].
            contact_schedule: Per-timestep contact flag (length >= horizon).
            horizon: Trajectory length.
            dt: Timestep.
            contact_smoothing_multiplier: Factor to increase smoothing
                during contact phases.
            contact_penalty_weight: Weight for contact-transition penalty.

        Returns:
            Optimization result.
        """
        if initial_state is None:
            raise ValueError("initial_state must be provided")
        original_smoothing = self.smoothing_factor

        schedule = self._pad_contact_schedule(contact_schedule, horizon)
        loss_fn = self._build_contact_loss(goal_state, schedule, contact_penalty_weight)

        controls = np.zeros((horizon, self._n_u))
        best_controls, best_loss, grad_norm, iteration = self._adam_optimize_contact(
            initial_state,
            controls,
            loss_fn,
            dt,
            schedule,
            original_smoothing,
            contact_smoothing_multiplier,
        )

        self.smoothing_factor = original_smoothing
        optimal_trajectory = self.simulate_trajectory(initial_state, best_controls, dt)

        return OptimizationResult(
            success=best_loss < 0.1,
            optimal_states=optimal_trajectory,
            optimal_controls=best_controls,
            final_cost=best_loss,
            iterations=iteration + 1,
            gradient_norm=grad_norm,
        )

    def _pad_contact_schedule(
        self,
        contact_schedule: list[bool],
        horizon: int,
    ) -> list[bool]:
        if len(contact_schedule) >= horizon:
            return contact_schedule[:horizon]
        return contact_schedule + [False] * (horizon - len(contact_schedule))

    def _build_contact_loss(
        self,
        goal_state: NDArray[np.floating],
        schedule: list[bool],
        contact_penalty_weight: float,
    ) -> Callable[[NDArray[np.floating]], float]:
        def loss_fn(trajectory: NDArray[np.floating]) -> float:
            """Compute goal error plus contact-transition velocity penalty."""
            diff_goal = trajectory[-1] - goal_state
            final_error = float(np.vdot(diff_goal, diff_goal))

            contact_penalty = 0.0
            n_q = self._n_q
            for t in range(min(len(schedule) - 1, len(trajectory) - 2)):
                if schedule[t] != schedule[t + 1]:
                    v_curr = trajectory[t + 1, n_q:]
                    v_next = trajectory[t + 2, n_q:]
                    diff_v = v_next - v_curr
                    contact_penalty += float(np.vdot(diff_v, diff_v))

            return final_error + contact_penalty_weight * contact_penalty

        return loss_fn

    def _adam_optimize_contact(
        self,
        initial_state: NDArray[np.floating],
        controls: NDArray[np.floating],
        loss_fn: Callable[[NDArray[np.floating]], float],
        dt: float,
        schedule: list[bool],
        original_smoothing: float,
        contact_smoothing_multiplier: float,
    ) -> tuple[NDArray[np.floating], float, float, int]:
        if initial_state is None:
            raise ValueError("initial_state must be provided")
        m = np.zeros_like(controls)
        v = np.zeros_like(controls)
        beta1, beta2 = 0.9, 0.999
        eps_adam = 1e-8
        lr = 0.01

        best_loss = float("inf")
        best_controls = controls.copy()
        grad_norm = float("inf")
        iteration = 0

        for iteration in range(100):
            self._apply_phase_smoothing(
                schedule,
                original_smoothing,
                contact_smoothing_multiplier,
            )

            gradient = self.compute_gradient(initial_state, controls, loss_fn, dt)
            trajectory = self.simulate_trajectory(initial_state, controls, dt)
            current_loss = loss_fn(trajectory)

            if current_loss < best_loss:
                best_loss = current_loss
                best_controls = controls.copy()

            grad_norm = float(np.linalg.norm(gradient))
            if grad_norm < 1e-6:
                break

            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))
            controls = controls - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        return best_controls, best_loss, grad_norm, iteration

    def _apply_phase_smoothing(
        self,
        schedule: list[bool],
        original_smoothing: float,
        contact_smoothing_multiplier: float,
    ) -> None:
        if any(schedule):
            self.smoothing_factor = original_smoothing * contact_smoothing_multiplier
        else:
            self.smoothing_factor = original_smoothing
