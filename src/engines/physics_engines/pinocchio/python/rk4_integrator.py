"""Pinocchio-specific RK4 Integrator Implementation.

Implements the shared RK4StandardIntegrator interface using Pinocchio's
Articulated Body Algorithm (ABA) for forward dynamics computation.

This integrator is designed for parity testing with Drake, OpenSim,
and MuJoCo RK4 implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from src.shared.python.integrators.rk4_standard import (
    RK4StandardIntegrator,
    RK4StepResult,
)

if TYPE_CHECKING:
    import pinocchio as pin

from src.shared.python.engine_core.engine_availability import PINOCCHIO_AVAILABLE

if PINOCCHIO_AVAILABLE:
    import pinocchio as pin


class PinocchioRK4Integrator(RK4StandardIntegrator):
    """RK4 Integrator using Pinocchio's ABA for forward dynamics.

    Integrates the Pinocchio tangent-space dynamics:
        d/dt [q] = [v]
        d/dt [v] = a(q, v, tau)

    Where a(q, v, tau) is computed via Pinocchio's Articulated Body Algorithm (ABA).

    Attributes:
        model: Pinocchio model
        data: Pinocchio data (mutable computation state)
        control: Current applied torques/forces
    """

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data | None = None,
        timestep: float = 0.001,
        tolerance: float = 1e-8,
        validate_stages: bool = True,
    ) -> None:
        """Initialize Pinocchio RK4 integrator.

        Args:
            model: Pinocchio model (must be finalized)
            data: Pinocchio data struct (created if None)
            timestep: Integration timestep in seconds
            tolerance: Numerical tolerance for validation
            validate_stages: Whether to validate intermediate RK4 stages

        Raises:
            ValueError: If model is None or invalid
            RuntimeError: If Pinocchio is not available
        """
        if not PINOCCHIO_AVAILABLE:
            raise RuntimeError("Pinocchio is not available")

        if model is None:
            raise ValueError("Pinocchio model must be provided")

        super().__init__(
            timestep=timestep,
            tolerance=tolerance,
            validate_stages=validate_stages,
        )

        self.model = model
        self.data = data if data is not None else model.createData()
        self.control: np.ndarray = np.zeros(self.model.nv)

    def set_control(self, tau: np.ndarray) -> None:
        """Set applied torques/forces for next step.

        Args:
            tau: Generalized forces vector (nv,)

        Raises:
            ValueError: If tau has wrong dimensions
        """
        if tau.shape[0] != self.model.nv:
            raise ValueError(
                f"Control dimension mismatch: expected {self.model.nv}, got {tau.shape[0]}"
            )
        self.control = tau.copy()

    def forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        control: np.ndarray | None = None,
        time: float = 0.0,
    ) -> np.ndarray:
        """Compute forward dynamics using Pinocchio's ABA.

        Args:
            q: Generalized positions (nq,)
            v: Generalized velocities (nv,)
            control: Applied torques (nv,). If None, uses self.control
            time: Simulation time (unused, for interface compliance)

        Returns:
            a: Generalized accelerations (nv,)

        Raises:
            ValueError: If states have wrong dimensions
        """
        if q.shape[0] != self.model.nq:
            raise ValueError(
                f"Position dimension mismatch: expected {self.model.nq}, got {q.shape[0]}"
            )
        if v.shape[0] != self.model.nv:
            raise ValueError(
                f"Velocity dimension mismatch: expected {self.model.nv}, got {v.shape[0]}"
            )

        tau = control if control is not None else self.control
        if tau.shape[0] != self.model.nv:
            raise ValueError(
                f"Control dimension mismatch: expected {self.model.nv}, got {tau.shape[0]}"
            )

        # ABA computes a = M^{-1} * (tau - C(q,v)*v - g(q))
        a = cast(
            np.ndarray,
            pin.aba(self.model, self.data, q, v, tau),
        )
        return a.copy()

    def _integrate_position(
        self,
        q: np.ndarray,
        dq: np.ndarray,
    ) -> np.ndarray:
        """Integrate position using Pinocchio's manifold-aware exponential map.

        For SE(3)-based systems (humanoid, robotics), this handles
        quaternion normalization and manifold constraints properly.

        Args:
            q: Generalized positions
            dq: Position velocity delta

        Returns:
            q_next: Integrated positions (on manifold)
        """
        return cast(
            np.ndarray,
            pin.integrate(self.model, q, dq),
        )

    def step(
        self,
        q0: np.ndarray,
        v0: np.ndarray,
        control: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> RK4StepResult:
        """Execute one RK4 step with Pinocchio dynamics.

        Args:
            q0: Initial positions
            v0: Initial velocities
            control: Applied torques (constant over step)
            t0: Initial simulation time

        Returns:
            RK4StepResult: New state and diagnostics
        """
        if control is not None:
            self.set_control(control)

        return super().step(q0, v0, control=self.control, t0=t0)
