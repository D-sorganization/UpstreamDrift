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
    constraint: np.ndarray
    total: np.ndarray


class MuJoCoInducedAccelerationAnalyzer:
    """Analyzes induced accelerations (Gravity, Velocity, Control) for MuJoCo models."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize analyzer."""
        self.model = model
        self.data = data

        # Pre-allocate Jacobian buffers
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def compute_components(
        self, tau_app: np.ndarray | None = None
    ) -> InducedAccelerationResult:
        """Compute acceleration components induced by different forces.

        Decomposes Euler-Lagrange equation: M(q)qdd + C(q,qd)qd + G(q) = tau + J^T f_c
        Into:
          qdd_g = -M^-1 * G(q)
          qdd_c = -M^-1 * C(q,qd)qd
          qdd_t = M^-1 * tau
          qdd_cn = M^-1 * J^T f_c

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
        # We also need to save qacc/cacc if we want to preserve them,
        # but mj_forward overwrites them anyway.
        # We assume the caller handles state restoration if needed,
        # or we restore strictly here. mj_forward updates everything.

        try:
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
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
            # Use actual applied controls if not specified?
            # self.data.qfrc_actuator contains actuator forces
            # self.data.ctrl contains inputs.
            # If tau_app is None, we assume we want the *current* actuator contribution.
            # But qfrc_actuator is force, not torque? No, generalized force.
            acc_t = np.linalg.solve(M, self.data.qfrc_actuator)

        # Constraints
        # qfrc_constraint contains constraint forces
        acc_cn = np.linalg.solve(M, self.data.qfrc_constraint)

        total = acc_g + acc_c + acc_t + acc_cn

        return {
            "gravity": acc_g,
            "velocity": acc_c,
            "control": acc_t,
            "constraint": acc_cn,
            "total": total,
        }

    def compute_task_space_components(
        self, body_name: str, qdd_comps: InducedAccelerationResult | None = None
    ) -> dict[str, np.ndarray] | None:
        """Compute task space (linear) induced accelerations for a body in World Frame.

        Decomposes body acceleration into:
        - Gravity: J * qdd_g
        - Control: J * qdd_t
        - Constraint: J * qdd_cn
        - Velocity: J * qdd_c + Bias (where Bias = Total - J * qdd_total)

        Args:
            body_name: Name of the body
            qdd_comps: Optional pre-computed joint space components.

        Returns:
            Dictionary of 3D acceleration vectors (World Frame) or None if not found.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return None

        # 1. Capture Current Dynamic State (Before any potential modification)
        # We need this for the "Bias" term calculation.
        # cacc: (3 rotational, 3 linear) in local body frame
        cacc_local = self.data.cacc[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()

        # 2. Get Joint Space Components
        # If not provided, compute them (this might temp modify data, but restores it)
        if qdd_comps is None:
            qdd_comps = self.compute_components()

        # 3. Get Jacobian (Linear)
        # Reshape buffers
        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, body_id)
        J = self._jacp  # 3 x nv

        # 4. Compute Task Space Components (J * qdd)
        # These are vectors in World Frame because J is in World Frame.
        a_g = J @ qdd_comps["gravity"]
        a_t = J @ qdd_comps["control"]
        a_cn = J @ qdd_comps["constraint"]
        a_c_joint = J @ qdd_comps["velocity"]  # Only J * qdd_c part

        # 5. Compute Bias (J_dot * q_dot)
        # We derive this from total acceleration.
        # a_total_actual = J * qdd_total_actual + J_dot * q_dot

        # Transform actual total acceleration to World Frame
        a_local_linear = cacc_local[3:6]
        a_total_actual_world = xmat @ a_local_linear

        # qdd_total_actual should match sum(qdd_comps)
        qdd_total = qdd_comps["total"]
        a_total_from_qdd = J @ qdd_total

        # Bias = Total (World) - J * qdd_total (World)
        a_bias = a_total_actual_world - a_total_from_qdd

        # 6. Combine Velocity terms
        # Velocity induced = J * qdd_c + Bias
        a_v = a_c_joint + a_bias

        return {
            "gravity": a_g,
            "velocity": a_v,
            "control": a_t,
            "constraint": a_cn,
            "total": a_total_actual_world,
        }
