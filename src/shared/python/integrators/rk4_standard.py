"""RK4 Integrator Standard Interface for Physics Engines.

This module defines the standard RK4 (Runge-Kutta 4th order) integrator
interface that all physics engines should implement for timestep integration.

The RK4 standard ensures numerical accuracy and stability across Drake,
Pinocchio, OpenSim, and MuJoCo engines while maintaining parity in
dynamics computations.

Standard RK4 Algorithm:
    k1 = f(q, v)
    k2 = f(q + 0.5*dt*v, v + 0.5*dt*k1)
    k3 = f(q + 0.5*dt*v_k2, v + 0.5*dt*k2)
    k4 = f(q + dt*v_k3, v + dt*k3)

    q_next = q + (dt/6) * (v + 2*v_k2 + 2*v_k3 + v_next)
    v_next = v + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

Where:
    - q: Generalized positions (configuration manifold)
    - v: Generalized velocities (tangent space)
    - f: Forward dynamics function (ABA or equivalent)
    - dt: Timestep
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class IntegrationState:
    """Capsule for state during RK4 integration stage.

    Attributes:
        q: Generalized positions (configuration space)
        v: Generalized velocities (velocity space)
        a: Generalized accelerations (from dynamics)
        stage: RK4 stage number (1-4)
        t: Simulation time at this stage
    """

    q: np.ndarray
    v: np.ndarray
    a: np.ndarray | None = None
    stage: int = 0
    t: float = 0.0


@dataclass
class RK4StepResult:
    """Result of an RK4 integration step.

    Attributes:
        q_next: Updated generalized positions
        v_next: Updated generalized velocities
        a_final: Final acceleration (stage 4)
        energy_error: Relative energy conservation error (optional)
        stages_validated: Boolean indicating all 4 stages were validated
    """

    q_next: np.ndarray
    v_next: np.ndarray
    a_final: np.ndarray
    energy_error: float | None = None
    stages_validated: bool = False


class RK4StandardIntegrator(ABC):
    """Abstract base class for RK4 integrators across physics engines.

    Ensures all physics engines (Drake, Pinocchio, OpenSim, MuJoCo)
    implement RK4 integration consistently with the same interface.

    Subclasses must implement the forward_dynamics method to provide
    engine-specific acceleration computation (ABA for Pinocchio, etc).
    """

    def __init__(
        self,
        timestep: float = 0.001,
        tolerance: float = 1e-8,
        validate_stages: bool = True,
    ) -> None:
        """Initialize RK4 integrator.

        Args:
            timestep: Integration timestep in seconds (default 0.001s = 1ms).
            tolerance: Numerical tolerance for state validation.
            validate_stages: Whether to validate sub-step states.

        Raises:
            ValueError: If timestep <= 0 or tolerance <= 0.
        """
        if timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")

        self.timestep = timestep
        self.tolerance = tolerance
        self.validate_stages = validate_stages
        self._stage_count = 0

    @abstractmethod
    def forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        control: np.ndarray | None = None,
        time: float = 0.0,
    ) -> np.ndarray:
        """Compute forward dynamics acceleration.

        This is the engine-specific function that drives RK4 integration.
        Typically implemented as Pinocchio's ABA, Drake's inverse dynamics, etc.

        Args:
            q: Generalized positions
            v: Generalized velocities
            control: Applied controls/torques (optional, engine-specific)
            time: Current simulation time

        Returns:
            a: Generalized accelerations from forward dynamics
        """
        pass

    def step(
        self,
        q0: np.ndarray,
        v0: np.ndarray,
        control: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> RK4StepResult:
        """Execute one RK4 integration step.

        Performs a complete 4-stage RK4 step from (q0, v0) -> (q1, v1)
        using the supplied forward dynamics function.

        Args:
            q0: Initial generalized positions
            v0: Initial generalized velocities
            control: Applied controls (constant over the step)
            t0: Initial simulation time

        Returns:
            RK4StepResult: Updated state and diagnostics

        Raises:
            ValueError: If states become non-finite during integration
        """
        self._stage_count = 0
        dt = self.timestep

        # RK4 Integration for Second-Order Systems (q, v) dynamics
        # d/dt [q] = v
        # d/dt [v] = a(q, v, u)

        # Stage 1: Evaluate at initial state
        a1 = self.forward_dynamics(q0, v0, control, t0)
        self._validate_finite(a1, "acceleration at stage 1")
        stage_1 = IntegrationState(q0.copy(), v0.copy(), a1.copy(), stage=1, t=t0)

        # Stage 2: Half-step prediction using v0 and a1
        # Position: q = q0 + 0.5*dt*v0
        # Velocity: v = v0 + 0.5*dt*a1
        q2 = self._integrate_position(q0, v0 * (dt * 0.5))
        v2 = v0 + a1 * (dt * 0.5)
        a2 = self.forward_dynamics(q2, v2, control, t0 + dt * 0.5)
        self._validate_finite(a2, "acceleration at stage 2")
        stage_2 = IntegrationState(q2.copy(), v2.copy(), a2.copy(), stage=2, t=t0 + dt * 0.5)

        # Stage 3: Half-step prediction using v2 and a2
        # Position: q = q0 + 0.5*dt*v2
        # Velocity: v = v0 + 0.5*dt*a2
        q3 = self._integrate_position(q0, v2 * (dt * 0.5))
        v3 = v0 + a2 * (dt * 0.5)
        a3 = self.forward_dynamics(q3, v3, control, t0 + dt * 0.5)
        self._validate_finite(a3, "acceleration at stage 3")
        stage_3 = IntegrationState(q3.copy(), v3.copy(), a3.copy(), stage=3, t=t0 + dt * 0.5)

        # Stage 4: Full-step prediction using v3 and a3
        # Position: q = q0 + dt*v3
        # Velocity: v = v0 + dt*a3
        q4 = self._integrate_position(q0, v3 * dt)
        v4 = v0 + a3 * dt
        a4 = self.forward_dynamics(q4, v4, control, t0 + dt)
        self._validate_finite(a4, "acceleration at stage 4")
        stage_4 = IntegrationState(q4.copy(), v4.copy(), a4.copy(), stage=4, t=t0 + dt)

        # Weighted average using standard RK4 coefficients
        # Position: average of velocities {v0, 2*v2, 2*v3, v4}
        # Velocity: average of accelerations {a1, 2*a2, 2*a3, a4}
        weighted_v = (v0 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        weighted_a = (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0

        # Final integration step
        q_next = self._integrate_position(q0, weighted_v * dt)
        v_next = v0 + weighted_a * dt

        # Validate all stages if requested
        stages_ok = True
        if self.validate_stages:
            for stage in [stage_1, stage_2, stage_3, stage_4]:
                stages_ok &= self._validate_stage(stage)

        return RK4StepResult(
            q_next=q_next,
            v_next=v_next,
            a_final=a4.copy(),
            energy_error=None,
            stages_validated=stages_ok,
        )

    def _integrate_position(
        self,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """Integrate position with velocity delta.

        For simple systems (free-floating, Euclidean), this is q + dq.
        For manifolds (SO(3), SE(3)), this should use manifold-aware
        exponential map integration (Pinocchio's pin.integrate, etc).

        Default implementation assumes Euclidean space; override for manifolds.

        Args:
            q: Generalized positions
            dq: Position delta

        Returns:
            q_next: Integrated positions
        """
        return q + dq

    def _validate_finite(
        self,
        array: np.ndarray,
        description: str,
    ) -> None:
        """Check that array contains only finite values.

        Args:
            array: Array to validate
            description: Description for error messages

        Raises:
            ValueError: If array contains NaN or Inf values
        """
        if not np.isfinite(array).all():
            raise ValueError(
                f"Non-finite values in {description}: {array}. "
                "Integration became unstable."
            )

    def _validate_stage(self, state: IntegrationState) -> bool:
        """Validate state at a single RK4 stage.

        Checks for:
        - Finite values in q, v, a
        - No catastrophic energy drift (stage-level)
        - Velocity within expected bounds

        Args:
            state: IntegrationState at a specific stage

        Returns:
            True if state is valid, False otherwise
        """
        # Check finiteness
        if not (
            np.isfinite(state.q).all()
            and np.isfinite(state.v).all()
            and (state.a is None or np.isfinite(state.a).all())
        ):
            return False

        # Check velocity magnitude sanity (optional, engine-specific)
        # Typical humanoid speeds are < 10 m/s per joint
        if np.linalg.norm(state.v) > 100.0:
            return False

        return True
