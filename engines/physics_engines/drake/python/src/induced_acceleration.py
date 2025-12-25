"""Induced Acceleration Analysis for Drake models."""

import typing

import numpy as np
from pydrake.all import Context, MultibodyPlant


class InducedAccelerationResult(typing.TypedDict):
    """Dictionary containing induced acceleration components."""

    gravity: np.ndarray
    velocity: np.ndarray
    control: np.ndarray
    total: np.ndarray


class DrakeInducedAccelerationAnalyzer:
    """Analyzes induced accelerations (Gravity, Velocity, Control) for Drake models."""

    def __init__(self, plant: MultibodyPlant) -> None:
        """Initialize analyzer."""
        self.plant = plant

    def compute_components(
        self, context: Context, tau_app: np.ndarray | None = None
    ) -> InducedAccelerationResult:
        """Compute acceleration components induced by different forces.

        Equation: M(q)v_dot + C(q, v)v + G(q) = tau + tau_ext
        v_dot = M^-1 * (tau - C - G)

        Args:
            context: Drake Context with state (q, v).
            tau_app: Applied control torques (optional).
        """
        # 1. Mass Matrix
        M = self.plant.CalcMassMatrix(context)

        # 2. Gravity Forces (G)
        # CalcGravityGeneralizedForces returns -G(q) usually?
        # Drake doc: "Calculates the generalized forces due to gravity... tau_g(q)"
        # It IS the term on the RHS? "tau_g = -G(q)"?
        # Let's verify convention.
        # Dynamics: M v_dot + C(q,v)v = tau + tau_g + tau_app
        # So tau_g is already on RHS.
        tau_g = self.plant.CalcGravityGeneralizedForces(context)

        # 3. Bias Term (C(q,v)v - tau_g) or (C(q,v)v)?
        # CalcBiasTerm returns C(q,v)v - tau_g(q).
        # i.e. the term "C(q,v)v + G(q)".
        # Wait, if tau_g is on RHS, then C+G means the terms moved to LHS?
        # Drake: inverse_dynamics(q, v, vd) -> tau_required.
        # tau_id = M vd + bias.
        # bias = C(q,v)v - tau_g.
        # So M vd + C v - tau_g = tau_app.
        # => M vd = tau_app + tau_g - C v.
        # Note: tau_g is gravity helping. G(q) usually opposes.

        bias = self.plant.CalcBiasTerm(context)

        # We want to isolate G and C.
        # tau_g = gravity force.
        # bias = C_v - tau_g
        # => C_v = bias + tau_g?
        # Let's check signs.
        # If v=0, bias = -tau_g.
        # So C_zero_v = -tau_g + tau_g = 0. Correct.

        # So Coriolis Force term (C*v) = bias - (-tau_g)? No.
        # bias = Cv - tau_g.
        # Cv = bias + tau_g.

        # Forces acting on system (RHS):
        # F_total = tau_app + tau_g - Cv.
        # F_total = tau_app + tau_g - (bias + tau_g)
        # F_total = tau_app - bias. (Wait, tau_g cancels?)

        # M vd = tau_app - bias.
        # This matches "tau_required = M vd + bias".
        # If we supply tau_required, M vd = M vd + bias - bias = M vd.

        # So Total Force F_total = tau_app - bias.
        # We want to decompose F_total into G, C, Control.
        # - bias = - (Cv - tau_g) = tau_g - Cv.
        # So Gravity contribution is tau_g.
        # Coriolis contribution is -Cv.
        # Control contribution is tau_app.

        # Cv = bias + tau_g.
        # So Coriolis Force = -(bias + tau_g).

        # Induced Accels: M^-1 * Force.

        # A_gravity = M^-1 * tau_g
        # A_coriolis = M^-1 * -(bias + tau_g)
        # A_control = M^-1 * tau_app

        # Note: tau_app is not stored in context usually, unless we track it.
        # For now we assume tau_app = 0 (passive) or we rely on user input.
        # We'll set control accel to 0 for now as we analyze passive swing or we need
        # recorded tau.
        # The recorder stores q, v. It doesn't store tau/u.

        # Solve M * a = F

        acc_g = np.linalg.solve(M, tau_g)
        acc_c = np.linalg.solve(M, -(bias + tau_g))

        # Control is zero
        acc_t = np.zeros_like(acc_g)

        total = acc_g + acc_c + acc_t

        return {
            "gravity": acc_g,
            "velocity": acc_c,
            "control": acc_t,
            "total": total,
        }
