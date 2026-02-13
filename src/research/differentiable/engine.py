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
        T = len(controls)
        trajectory = np.zeros((T + 1, self._n_x))
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
            if hasattr(self.engine, "step"):
                self.engine.step(dt)

            # Record state
            if hasattr(self.engine, "get_joint_positions"):
                q = self.engine.get_joint_positions()
            else:
                q = trajectory[t, : self._n_q]

            if hasattr(self.engine, "get_joint_velocities"):
                v = self.engine.get_joint_velocities()
            else:
                v = trajectory[t, self._n_q :]

            trajectory[t + 1] = np.concatenate([q, v])

        return trajectory

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
        eps = 1e-5
        T, n_u = controls.shape
        gradient = np.zeros_like(controls)

        # Baseline trajectory and loss
        baseline_traj = self.simulate_trajectory(initial_state, controls, dt)
        baseline_loss = loss_fn(baseline_traj)

        # Numerical gradient
        for t in range(T):
            for i in range(n_u):
                controls_plus = controls.copy()
                controls_plus[t, i] += eps

                traj_plus = self.simulate_trajectory(initial_state, controls_plus, dt)
                loss_plus = loss_fn(traj_plus)

                gradient[t, i] = (loss_plus - baseline_loss) / eps

        return gradient

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
        eps = 1e-5

        # Compute nominal next state
        q = state[: self._n_q]
        v = state[self._n_q :]

        if hasattr(self.engine, "set_joint_positions"):
            self.engine.set_joint_positions(q)
        if hasattr(self.engine, "set_joint_velocities"):
            self.engine.set_joint_velocities(v)
        if hasattr(self.engine, "set_joint_torques"):
            self.engine.set_joint_torques(control)
        if hasattr(self.engine, "step"):
            self.engine.step(dt)

        if hasattr(self.engine, "get_joint_positions"):
            q_next = self.engine.get_joint_positions()
        else:
            q_next = q + v * dt

        if hasattr(self.engine, "get_joint_velocities"):
            v_next = self.engine.get_joint_velocities()
        else:
            v_next = v

        x_next = np.concatenate([q_next, v_next])

        # State Jacobian
        A = np.zeros((self._n_x, self._n_x))
        for i in range(self._n_x):
            state_plus = state.copy()
            state_plus[i] += eps

            q_plus = state_plus[: self._n_q]
            v_plus = state_plus[self._n_q :]

            if hasattr(self.engine, "set_joint_positions"):
                self.engine.set_joint_positions(q_plus)
            if hasattr(self.engine, "set_joint_velocities"):
                self.engine.set_joint_velocities(v_plus)
            if hasattr(self.engine, "set_joint_torques"):
                self.engine.set_joint_torques(control)
            if hasattr(self.engine, "step"):
                self.engine.step(dt)

            if hasattr(self.engine, "get_joint_positions"):
                q_new = self.engine.get_joint_positions()
            else:
                q_new = q_plus + v_plus * dt

            if hasattr(self.engine, "get_joint_velocities"):
                v_new = self.engine.get_joint_velocities()
            else:
                v_new = v_plus

            x_new = np.concatenate([q_new, v_new])
            A[:, i] = (x_new - x_next) / eps

        # Control Jacobian
        B = np.zeros((self._n_x, self._n_u))
        for i in range(self._n_u):
            control_plus = control.copy()
            control_plus[i] += eps

            if hasattr(self.engine, "set_joint_positions"):
                self.engine.set_joint_positions(q)
            if hasattr(self.engine, "set_joint_velocities"):
                self.engine.set_joint_velocities(v)
            if hasattr(self.engine, "set_joint_torques"):
                self.engine.set_joint_torques(control_plus)
            if hasattr(self.engine, "step"):
                self.engine.step(dt)

            if hasattr(self.engine, "get_joint_positions"):
                q_new = self.engine.get_joint_positions()
            else:
                q_new = q + v * dt

            if hasattr(self.engine, "get_joint_velocities"):
                v_new = self.engine.get_joint_velocities()
            else:
                v_new = v

            x_new = np.concatenate([q_new, v_new])
            B[:, i] = (x_new - x_next) / eps

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
        controls = np.zeros((horizon, self._n_u))

        # Define loss function
        def loss_fn(trajectory: NDArray[np.floating]) -> float:
            final_state = trajectory[-1]
            state_error = final_state - goal_state
            return float(np.sum(state_error**2))

        # Adam optimizer state
        if method == "adam":
            m = np.zeros_like(controls)
            v = np.zeros_like(controls)
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

        best_loss = float("inf")
        best_controls = controls.copy()

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
            final_error = float(np.sum((trajectory[-1] - goal_state) ** 2))

            contact_penalty = 0.0
            n_q = self._n_q
            for t in range(min(len(schedule) - 1, len(trajectory) - 2)):
                if schedule[t] != schedule[t + 1]:
                    v_curr = trajectory[t + 1, n_q:]
                    v_next = trajectory[t + 2, n_q:]
                    contact_penalty += float(np.sum((v_next - v_curr) ** 2))

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
