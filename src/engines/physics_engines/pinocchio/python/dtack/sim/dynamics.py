"""Dynamics simulation and counterfactual analysis."""

from __future__ import annotations

import logging

import numpy as np  # noqa: TID253
import pinocchio as pin

logger = logging.getLogger(__name__)


class DynamicsEngine:
    """Wrapper for Pinocchio dynamics algorithms."""

    def __init__(self, model: pin.Model, data: pin.Data) -> None:
        """Initialize dynamics engine.

        Args:
            model: Pinocchio model
            data: Pinocchio data
        """
        self.model = model
        self.data = data

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, f_ext: list | None = None
    ) -> np.ndarray:
        """Compute forward dynamics (FD).

        Equation: M(q)a + C(q,v)v + g(q) = tau + J^T f_ext
        Returns: a (acceleration)

        Args:
            q: Joint configuration
            v: Joint velocity
            tau: Joint torques
            f_ext: External forces (optional)

        Returns:
            Joint acceleration 'a'
        """
        if f_ext is None:
            result = pin.aba(self.model, self.data, q, v, tau)
            return np.array(result, dtype=np.float64)
        result = pin.aba(self.model, self.data, q, v, tau, f_ext)
        return np.array(result, dtype=np.float64)

    def inverse_dynamics(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray, f_ext: list | None = None
    ) -> np.ndarray:
        """Compute inverse dynamics (ID).

        Returns: tau (torque)
        """
        if f_ext is None:
            result = pin.rnea(self.model, self.data, q, v, a)
            return np.array(result, dtype=np.float64)
        result = pin.rnea(self.model, self.data, q, v, a, f_ext)
        return np.array(result, dtype=np.float64)

    def compute_ztcf(
        self, q: np.ndarray, v: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Zero Torque Counterfactual (ZTCF).

        Simulates one step with tau=0.
        Represents pure passive dynamics (drift).

        Returns:
            (q_next, v_next)
        """
        tau_zero = np.zeros(self.model.nv)
        a = self.forward_dynamics(q, v, tau_zero)

        # Semi-implicit Euler
        v_next = v + a * dt
        q_next = pin.integrate(self.model, q, v_next * dt)
        return q_next, v_next

    def compute_zvcf(
        self, q: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Zero Velocity Counterfactual (ZVCF).

        Computes acceleration assuming v=0 (no Coriolis/Centrifugal/Damping).
        Represents pure control authority + static gravity.

        Returns:
            (q_next, v_next) starting from v=0
        """
        v_zero = np.zeros(self.model.nv)
        a = self.forward_dynamics(q, v_zero, tau)

        v_next = v_zero + a * dt
        q_next = pin.integrate(self.model, q, v_next * dt)
        return q_next, v_next

    def compute_induced_acceleration(
        self, q: np.ndarray, tau_source: np.ndarray
    ) -> np.ndarray:
        """Compute acceleration induced solely by a specific torque source.

        Equation: a = M(q)^-1 * tau_source
        This ignores gravity, Coriolis, and other forces.

        Args:
            q: Joint configuration
            tau_source: Torque vector from the specific source

        Returns:
            Induced acceleration vector
        """
        # Compute Mass Matrix Inverse
        pin.computeMinverse(self.model, self.data, q)
        M_inv = self.data.Minv

        return np.asarray(M_inv @ tau_source)
