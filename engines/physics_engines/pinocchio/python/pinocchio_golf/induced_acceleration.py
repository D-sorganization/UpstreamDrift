import logging

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)


class InducedAccelerationAnalyzer:
    """
    Analyzes induced accelerations (Gravity, Velocity, Control) for a Pinocchio model.
    Based on the equation of motion: M(q)q_ddot + C(q, q_dot)q_dot + G(q) = tau

    Induced Accelerations:
    - Gravity: q_ddot_g = -M^(-1) * G(q)
    - Velocity (Coriolis/Centrifugal): q_ddot_v = -M^(-1) * C(q, q_dot)q_dot
    - Control (Torque): q_ddot_t = M^(-1) * tau
    - Total: q_ddot = q_ddot_g + q_ddot_v + q_ddot_t
    """

    def __init__(self, model: pin.Model, data: pin.Data):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv

        # We need a secondary data object to avoid side effects on the main simulation
        self._temp_data = model.createData()

    def compute_components(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Compute induced acceleration components.

        Args:
            q: Joint configurations
            v: Joint velocities
            tau: Joint torques

        Returns:
            Dictionary with keys 'gravity', 'velocity', 'control', 'total'
            mapping to acceleration arrays.
        """
        # 1. Gravity Induced Acceleration
        # M * q_ddot_g = -G(q)
        # We can use ABA with v=0, tau=0, and gravity enabled.
        # equation: M*a + 0 + G = 0 => M*a = -G
        # Pinocchio ABA: a = aba(model, data, q, v, tau)
        q_ddot_g = pin.aba(
            self.model, self._temp_data, q, np.zeros(self.nv), np.zeros(self.nv)
        )

        # 2. Velocity Induced Acceleration
        # M * q_ddot_v = -C(q, v)v
        # We can use ABA with v=v, tau=0, and gravity disabled.
        # But modifying model.gravity is risky.
        # Alternative: Calculate C term explicitly and solve M * a = -C
        # C_term (Corios+Centrifugal) can be isolated:
        # nonLinearEffects(q, v) = C*v + G
        # computeGeneralizedGravity(q) = G
        # C*v = nonLinearEffects - G

        # Or simply:
        # q_ddot_v = aba(model_no_grav, data, q, v, 0)

        # Let's try the subtraction method which is safe and doesn't require modifying
        # model:
        # ABA(q, v, 0) gives a s.t. M*a + C*v + G = 0 => M*a = -C*v - G
        # This is (Gravity + Velocity) induced acc.
        q_ddot_gv = pin.aba(self.model, self._temp_data, q, v, np.zeros(self.nv))

        # q_ddot_v = q_ddot_gv - q_ddot_g
        q_ddot_v = q_ddot_gv - q_ddot_g

        # 3. Control Induced Acceleration
        # M * q_ddot_t = tau
        # We can use M.inverse() * tau.
        # Or ABA with: q, v=0, tau=tau, gravity=0 (impossible without mod).
        # OR:
        # ABA(q, v, tau) gives M*a + C*v + G = tau => M*a = tau - C*v - G
        # = tau - (C*v + G)
        # This is Total Acceleration.
        q_ddot_total = pin.aba(self.model, self._temp_data, q, v, tau)

        # q_ddot_t = q_ddot_total - q_ddot_gv
        q_ddot_t = q_ddot_total - q_ddot_gv

        return {
            "gravity": q_ddot_g,
            "velocity": q_ddot_v,
            "control": q_ddot_t,
            "total": q_ddot_total,
        }

    def compute_specific_control(
        self, q: np.ndarray, specific_tau: np.ndarray
    ) -> np.ndarray:
        """
        Compute induced acceleration for a specific control torque vector.

        Args:
            q: Joint configurations
            specific_tau: Torque vector to analyze

        Returns:
            Induced acceleration (M^-1 * specific_tau)
        """
        # M * a = tau  (assuming C=0, G=0, or just isolate tau contribution)
        # We want a = M^-1 * tau.
        # We can use ABA with v=0, gravity=0... but we can't easily turn off gravity in model.
        # But we know ABA(q, 0, tau) = M^-1 * (tau - G(q)).
        # And ABA(q, 0, 0) = M^-1 * (-G(q)).
        # So ABA(q, 0, tau) - ABA(q, 0, 0) = M^-1 * tau.

        a_tau_G = pin.aba(self.model, self._temp_data, q, np.zeros(self.nv), specific_tau)
        a_G = pin.aba(self.model, self._temp_data, q, np.zeros(self.nv), np.zeros(self.nv))

        return a_tau_G - a_G

    def compute_counterfactuals(
        self, q: np.ndarray, v: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Compute counterfactual metrics.

        Args:
            q: Joint configurations
            v: Joint velocities

        Returns:
            Dict with keys 'ztcf_accel' (Zero Torque Accel) and 'zvcf_torque' (Zero Velocity Torque)
        """
        # ZTCF: Acceleration if tau=0.
        # M*a + C*v + G = 0  => a = -M^-1 * (C*v + G)
        # This is just ABA with tau=0.
        ztcf_accel = pin.aba(self.model, self._temp_data, q, v, np.zeros(self.nv))

        # ZVCF: Torque/Force if v=0.
        # M*a + G = tau.
        # If we define ZVCF as "Forces to hold static posture" => a=0, v=0 => tau = G.
        # This is computeGeneralizedGravity(q).
        zvcf_torque = pin.computeGeneralizedGravity(self.model, self._temp_data, q)

        return {
            "ztcf_accel": ztcf_accel,
            "zvcf_torque": zvcf_torque,
        }
