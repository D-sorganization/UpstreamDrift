"""Induced Acceleration Analysis for MuJoCo models."""

from __future__ import annotations

import typing

import mujoco
import numpy as np


class InducedAccelerationResult(typing.TypedDict):
    """Dictionary containing induced acceleration components."""

    gravity: np.ndarray
    velocity: np.ndarray
    control: np.ndarray
    total: np.ndarray


class MuJoCoInducedAccelerationAnalyzer:
    """Analyzes induced accelerations (Gravity, Velocity, Control) for MuJoCo models."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize analyzer."""
        self.model = model
        self.data = data

    def compute_components(
        self, tau_app: np.ndarray | None = None
    ) -> InducedAccelerationResult:
        """Compute acceleration components induced by different forces.

        Decomposes Euler-Lagrange equation: M(q)qdd + C(q,qd)qd + G(q) = tau
        Into:
          qdd_g = -M^-1 * G(q)
          qdd_c = -M^-1 * C(q,qd)qd
          qdd_t = M^-1 * tau

        Args:
            tau_app: Applied control torques (optional).

        Returns:
            Dictionary of acceleration components.
        """
        nv = self.model.nv

        # 1. Mass Matrix M
        M = np.zeros((nv, nv))
        # Ensure inertia is updated
        mujoco.mj_fullM(self.model, M, self.data.qM)

        # 2. Compute G(q) (Gravity Force vector)
        # In MuJoCo qfrc_bias = C + G.
        # To get G only, we set qvel=0 temporarily.
        saved_qvel = self.data.qvel.copy()

        try:
            self.data.qvel[:] = 0

            # IMPORTANT: Run forward dynamics to update qfrc_bias with v=0
            # mj_forward recomputes everything including kinematics and bias
            mujoco.mj_forward(self.model, self.data)

            # qfrc_bias is now just G(q)
            term_G = self.data.qfrc_bias.copy()

        finally:
            # Restore qvel
            self.data.qvel[:] = saved_qvel
            # Restore full state dynamics
            mujoco.mj_forward(self.model, self.data)

        # 3. Compute C(q,v) (Coriolis/Centrifugal)
        # With qvel restored, qfrc_bias is C + G
        term_C_plus_G = self.data.qfrc_bias.copy()
        term_C = term_C_plus_G - term_G

        # 4. Solve for induced accelerations
        # Use simple linear solve (M is usually symmetric positive definite)
        # qdd = -M^-1 * Force

        acc_g = np.linalg.solve(M, -term_G)
        acc_c = np.linalg.solve(M, -term_C)

        if tau_app is not None:
            acc_t = np.linalg.solve(M, tau_app)
        else:
            acc_t = np.zeros(nv)

        total = acc_g + acc_c + acc_t

        return {
            "gravity": acc_g,
            "velocity": acc_c,
            "control": acc_t,
            "total": total,
        }
