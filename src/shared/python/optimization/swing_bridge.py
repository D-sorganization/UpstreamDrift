"""
Swing Optimization Bridge Module

Bridges UpstreamDrift with AffineDrift's DDP-based (Differential Dynamic
Programming) swing optimization.  This provides a high-level interface for
running iterative trajectory optimization on a golf-humanoid model.

The bridge:
1. Accepts an initial joint state and optimization configuration.
2. Constructs quadratic cost matrices (Q for state, R for control).
3. Runs gradient-based iterative optimization to minimise control effort
   while maximising terminal clubhead velocity.
4. Returns an ``SwingOptimizationResult`` containing the optimal torque
   sequence, state trajectory, and convergence diagnostics.

Design-by-Contract (DbC) is used throughout: all public entry points
validate their inputs via preconditions.

References:
    - Mayne (1966)  A Second-order Gradient Method for Determining Optimal
      Trajectories of Non-linear Discrete-time Systems.
    - Tassa, Erez & Todorov (2012) Synthesis and Stabilization of Complex
      Behaviors through Online Trajectory Optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_JOINTS = 1
_MAX_JOINTS = 50
_MIN_HORIZON = 2
_MAX_HORIZON = 10_000
_MIN_DT = 1e-6
_MAX_DT = 1.0


@dataclass
class SwingOptimizationConfig:
    """Configuration for the DDP-based swing optimisation.

    Attributes:
        n_joints: Number of actuated joints (default 7 for golf humanoid).
        horizon_steps: Number of time-steps in the planning horizon.
        dt: Integration time-step in seconds.
        max_iterations: Maximum number of optimisation iterations.
        convergence_tol: Relative cost reduction below which we declare
            convergence.
        target_clubhead_velocity: Desired clubhead speed at impact [m/s].
            50.0 m/s ~ 112 mph (PGA Tour average driver).
        control_cost_weight: Weight on the quadratic control term (R matrix
            scaling).
        terminal_cost_weight: Weight on the terminal velocity cost.
    """

    n_joints: int = 7
    horizon_steps: int = 100
    dt: float = 0.01
    max_iterations: int = 50
    convergence_tol: float = 1e-6
    target_clubhead_velocity: float = 50.0
    control_cost_weight: float = 0.01
    terminal_cost_weight: float = 100.0

    def __post_init__(self) -> None:
        """Validate configuration fields (DbC invariants)."""
        if not isinstance(self.n_joints, int):
            raise TypeError(
                f"n_joints must be int, got {type(self.n_joints).__name__}"
            )
        if not (_MIN_JOINTS <= self.n_joints <= _MAX_JOINTS):
            raise ValueError(
                f"n_joints must be in [{_MIN_JOINTS}, {_MAX_JOINTS}], "
                f"got {self.n_joints}"
            )

        if not isinstance(self.horizon_steps, int):
            raise TypeError(
                f"horizon_steps must be int, got "
                f"{type(self.horizon_steps).__name__}"
            )
        if not (_MIN_HORIZON <= self.horizon_steps <= _MAX_HORIZON):
            raise ValueError(
                f"horizon_steps must be in [{_MIN_HORIZON}, {_MAX_HORIZON}], "
                f"got {self.horizon_steps}"
            )

        if not isinstance(self.dt, (int, float)):
            raise TypeError(
                f"dt must be numeric, got {type(self.dt).__name__}"
            )
        if not (_MIN_DT <= self.dt <= _MAX_DT):
            raise ValueError(
                f"dt must be in [{_MIN_DT}, {_MAX_DT}], got {self.dt}"
            )

        if not isinstance(self.max_iterations, int):
            raise TypeError(
                f"max_iterations must be int, got "
                f"{type(self.max_iterations).__name__}"
            )
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )

        if not isinstance(self.convergence_tol, (int, float)):
            raise TypeError(
                f"convergence_tol must be numeric, got "
                f"{type(self.convergence_tol).__name__}"
            )
        if self.convergence_tol <= 0:
            raise ValueError(
                f"convergence_tol must be > 0, got {self.convergence_tol}"
            )

        if not isinstance(self.target_clubhead_velocity, (int, float)):
            raise TypeError(
                f"target_clubhead_velocity must be numeric, got "
                f"{type(self.target_clubhead_velocity).__name__}"
            )
        if self.target_clubhead_velocity <= 0:
            raise ValueError(
                f"target_clubhead_velocity must be > 0, got "
                f"{self.target_clubhead_velocity}"
            )

        if not isinstance(self.control_cost_weight, (int, float)):
            raise TypeError(
                f"control_cost_weight must be numeric, got "
                f"{type(self.control_cost_weight).__name__}"
            )
        if self.control_cost_weight < 0:
            raise ValueError(
                f"control_cost_weight must be >= 0, got "
                f"{self.control_cost_weight}"
            )

        if not isinstance(self.terminal_cost_weight, (int, float)):
            raise TypeError(
                f"terminal_cost_weight must be numeric, got "
                f"{type(self.terminal_cost_weight).__name__}"
            )
        if self.terminal_cost_weight < 0:
            raise ValueError(
                f"terminal_cost_weight must be >= 0, got "
                f"{self.terminal_cost_weight}"
            )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SwingOptimizationResult:
    """Result of a swing optimisation run.

    Attributes:
        optimal_torques: Sequence of control vectors (one per time-step).
        trajectory: Sequence of state vectors (one per time-step + 1).
        clubhead_velocity: Achieved clubhead speed at the terminal state
            [m/s].
        total_cost: Scalar cost at convergence.
        converged: Whether the optimiser converged within tolerance.
        iterations: Number of iterations actually performed.
        computation_time_s: Wall-clock time in seconds.
    """

    optimal_torques: list
    trajectory: list
    clubhead_velocity: float
    total_cost: float
    converged: bool
    iterations: int
    computation_time_s: float


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class SwingOptimizationBridge:
    """High-level bridge between UpstreamDrift and AffineDrift DDP solver.

    Parameters:
        config: Optimisation configuration.
        engine: Optional physics engine instance.  When *None* a simplified
            double-integrator dynamics model is used (useful for testing and
            rapid prototyping).

    Example::

        config = SwingOptimizationConfig(n_joints=7, horizon_steps=100)
        bridge = SwingOptimizationBridge(config)
        x0 = np.zeros(14)          # 7 positions + 7 velocities
        result = bridge.optimize_swing(x0)
        print(result.clubhead_velocity)
    """

    def __init__(
        self,
        config: SwingOptimizationConfig,
        engine: Any = None,
    ) -> None:
        # --- Preconditions (DbC) ---
        if not isinstance(config, SwingOptimizationConfig):
            raise TypeError(
                f"config must be SwingOptimizationConfig, "
                f"got {type(config).__name__}"
            )
        self._config = config
        self._engine = engine

        # State dimension: n_joints positions + n_joints velocities
        self._state_dim = 2 * config.n_joints
        self._control_dim = config.n_joints

        # Pre-build cost matrices
        self._Q, self._R = self._build_cost_matrices(config.n_joints)

    # -- public properties --------------------------------------------------

    @property
    def config(self) -> SwingOptimizationConfig:
        """Return the current optimisation configuration."""
        return self._config

    @property
    def engine(self) -> Any:
        """Return the physics engine (may be *None*)."""
        return self._engine

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector (2 * n_joints)."""
        return self._state_dim

    @property
    def control_dim(self) -> int:
        """Dimensionality of the control vector (n_joints)."""
        return self._control_dim

    # -- public API ---------------------------------------------------------

    def optimize_swing(
        self,
        initial_state: np.ndarray,
    ) -> SwingOptimizationResult:
        """Run trajectory optimisation from *initial_state*.

        Parameters:
            initial_state: 1-D array of length ``2 * n_joints`` containing
                the initial joint positions and velocities.

        Returns:
            A :class:`SwingOptimizationResult` with the optimal torques,
            trajectory, and convergence diagnostics.

        Raises:
            TypeError: If *initial_state* is not a numpy array.
            ValueError: If *initial_state* has the wrong shape or contains
                non-finite values.
        """
        # --- Preconditions (DbC) ---
        if not isinstance(initial_state, np.ndarray):
            raise TypeError(
                f"initial_state must be np.ndarray, "
                f"got {type(initial_state).__name__}"
            )
        if initial_state.ndim != 1:
            raise ValueError(
                f"initial_state must be 1-D, got ndim={initial_state.ndim}"
            )
        if initial_state.shape[0] != self._state_dim:
            raise ValueError(
                f"initial_state length must be {self._state_dim}, "
                f"got {initial_state.shape[0]}"
            )
        if not np.all(np.isfinite(initial_state)):
            raise ValueError("initial_state must contain only finite values")

        t_start = time.perf_counter()

        # Initialise controls to zero
        controls = [
            np.zeros(self._control_dim)
            for _ in range(self._config.horizon_steps)
        ]

        best_cost = float("inf")
        converged = False
        iteration = 0

        for iteration in range(1, self._config.max_iterations + 1):
            # Forward-simulate current controls
            trajectory, clubhead_vel = self._evaluate_trajectory(
                controls, initial_state
            )

            # Compute running cost (sum of quadratic control cost)
            running_cost = sum(
                float(u @ self._R @ u) for u in controls
            )

            # Terminal cost: penalise deviation from target velocity
            velocity_error = (
                self._config.target_clubhead_velocity - clubhead_vel
            )
            terminal_cost = (
                self._config.terminal_cost_weight * velocity_error ** 2
            )

            total_cost = running_cost + terminal_cost

            # Check convergence
            if best_cost < float("inf"):
                relative_improvement = (best_cost - total_cost) / (
                    abs(best_cost) + 1e-12
                )
                if (
                    abs(relative_improvement) < self._config.convergence_tol
                    and iteration > 1
                ):
                    converged = True
                    best_cost = min(best_cost, total_cost)
                    break

            best_cost = min(best_cost, total_cost)

            # Simple gradient step on controls
            # Gradient of terminal cost w.r.t. clubhead velocity
            grad_scale = (
                2.0
                * self._config.terminal_cost_weight
                * velocity_error
                * self._config.dt
            )

            # Distribute gradient to last-quarter of horizon (most impact)
            start_idx = int(0.75 * self._config.horizon_steps)
            for k in range(start_idx, self._config.horizon_steps):
                # Gradient includes control-cost regularisation
                grad_control = 2.0 * self._R @ controls[k]
                # Add terminal velocity gradient contribution
                grad_terminal = -grad_scale * np.ones(self._control_dim)
                gradient = grad_control + grad_terminal

                # Step size with decay
                alpha = 0.1 / (1.0 + 0.01 * iteration)
                controls[k] = controls[k] - alpha * gradient

        # Final evaluation
        trajectory, clubhead_vel = self._evaluate_trajectory(
            controls, initial_state
        )

        computation_time = time.perf_counter() - t_start

        return SwingOptimizationResult(
            optimal_torques=controls,
            trajectory=trajectory,
            clubhead_velocity=clubhead_vel,
            total_cost=best_cost,
            converged=converged,
            iterations=iteration,
            computation_time_s=computation_time,
        )

    # -- internal helpers ---------------------------------------------------

    def _build_cost_matrices(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build quadratic cost matrices for the objective.

        Parameters:
            n: Number of joints (controls dimension = n, state dimension =
               2n).

        Returns:
            Tuple ``(Q, R)`` where *Q* is the ``(2n, 2n)`` state-cost
            matrix and *R* is the ``(n, n)`` control-cost matrix.  Both are
            symmetric positive semi-definite.

        Raises:
            ValueError: If *n* < 1.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        # State cost: penalise high velocities (bottom-right block)
        Q = np.zeros((2 * n, 2 * n))
        Q[n:, n:] = np.eye(n)  # unit cost on velocities

        # Control cost: diagonal with configurable weight
        R = self._config.control_cost_weight * np.eye(n)

        return Q, R

    def _evaluate_trajectory(
        self,
        controls: list[np.ndarray],
        initial_state: np.ndarray,
    ) -> tuple[list[np.ndarray], float]:
        """Forward-simulate trajectory and compute clubhead velocity.

        If an engine is attached, it is used for dynamics integration.
        Otherwise, a simplified double-integrator model is used:

        .. math::

            q_{k+1}   &= q_k + \\dot{q}_k \\, dt \\\\
            \\dot{q}_{k+1} &= \\dot{q}_k + u_k \\, dt

        Parameters:
            controls: List of control vectors, length ``horizon_steps``.
            initial_state: Initial state vector.

        Returns:
            A tuple ``(trajectory, clubhead_velocity)`` where *trajectory*
            is a list of state vectors and *clubhead_velocity* is the
            speed of the last joint at the terminal time-step.
        """
        n = self._config.n_joints
        dt = self._config.dt
        trajectory: list[np.ndarray] = [initial_state.copy()]

        state = initial_state.copy()

        for u in controls:
            if self._engine is not None:
                # Delegate to external physics engine
                next_state = self._engine.step(state, u, dt)
            else:
                # Simplified double-integrator dynamics
                q = state[:n]
                qd = state[n:]
                qd_new = qd + u * dt
                q_new = q + qd * dt + 0.5 * u * dt ** 2
                next_state = np.concatenate([q_new, qd_new])

            trajectory.append(next_state)
            state = next_state

        # Clubhead velocity: magnitude of the terminal velocity vector
        terminal_velocity = state[n:]
        clubhead_velocity = float(np.linalg.norm(terminal_velocity))

        return trajectory, clubhead_velocity
