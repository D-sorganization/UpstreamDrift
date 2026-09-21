"""Convention adapters and parity diagnostics between double pendulum implementations.

Reconciles:
1. Analytical ``DoublePendulumDynamics`` / ``GolfModelParams`` (distributed inertia,
   inclined swing plane, center-of-mass offsets).
2. Shipped Tools simulator ``pendulum_simulator.physics`` (point-mass approximation
   in vertical/planar coordinate formulation).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    LowerSegmentProperties,
    SegmentProperties,
)
from src.shared.python.pendulum_simulator.physics import (
    PendulumParams,
    mass_matrix as tools_mass_matrix,
)

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.model_params import GolfModelParams

logger = logging.getLogger(__name__)

__all__ = [
    "ParityDiagnosticResult",
    "analytical_to_tools_params",
    "compare_dynamics_parity",
    "golf_model_to_double_pendulum_params",
    "state_to_planar_coordinates",
    "tools_to_analytical_params",
]


@dataclass(frozen=True)
class ParityDiagnosticResult:
    """Diagnostic comparison between analytical and Tools pendulum formulations."""

    mass_matrix_max_abs_error: float
    coriolis_max_abs_error: float
    gravity_max_abs_error: float
    accel_max_abs_error: float
    is_parity_acceptable: bool
    notes: str


def golf_model_to_double_pendulum_params(
    model_params: GolfModelParams,
) -> DoublePendulumParameters:
    """Convert GolfModelParams to DoublePendulumParameters."""
    return model_params.to_double_pendulum_parameters()


def analytical_to_tools_params(
    params: DoublePendulumParameters,
) -> PendulumParams:
    """Convert DoublePendulumParameters to Tools PendulumParams."""
    l1 = float(params.upper_segment.length_m)
    l2 = float(params.lower_segment.length_m)
    m1 = float(params.upper_segment.mass_kg)
    m2 = float(params.lower_segment.shaft_mass_kg)
    m_club = float(params.lower_segment.clubhead_mass_kg)
    g = float(params.projected_gravity)
    b1 = float(params.damping_shoulder)
    b2 = float(params.damping_wrist)

    return PendulumParams(
        m1=m1,
        m2=m2,
        L1=l1,
        L2=l2,
        mClub=m_club,
        g=g,
        b1=b1,
        b2=b2,
        mu1=0.0,
        mu2=0.0,
    )


def tools_to_analytical_params(
    tools_params: PendulumParams,
    *,
    plane_inclination_deg: float = 0.0,
    arm_com_ratio: float = 0.45,
    shaft_com_ratio: float = 0.43,
    use_point_mass_approximation: bool = False,
) -> DoublePendulumParameters:
    """Convert Tools PendulumParams to DoublePendulumParameters."""
    l1 = float(tools_params.L1)
    l2 = float(tools_params.L2)
    m1 = float(tools_params.m1)
    m2 = float(tools_params.m2)
    m_head = float(tools_params.mClub)

    if use_point_mass_approximation:
        # Match Tools assumption where link masses act at link ends
        com_ratio_1 = 1.0
        inertia_com_1 = 0.0
        shaft_m = 0.0
        head_m = m2 + m_head
        com_ratio_2 = 1.0
    else:
        com_ratio_1 = arm_com_ratio
        inertia_com_1 = (1.0 / 12.0) * m1 * (l1**2)
        shaft_m = m2
        head_m = m_head
        com_ratio_2 = shaft_com_ratio

    upper = SegmentProperties(
        length_m=l1,
        mass_kg=m1,
        center_of_mass_ratio=com_ratio_1,
        inertia_about_com=inertia_com_1,
    )
    lower = LowerSegmentProperties(
        length_m=l2,
        shaft_mass_kg=shaft_m,
        clubhead_mass_kg=head_m,
        shaft_com_ratio=com_ratio_2,
    )
    g_m_s2 = float(tools_params.g)

    return DoublePendulumParameters(
        upper_segment=upper,
        lower_segment=lower,
        plane_inclination_deg=plane_inclination_deg,
        damping_shoulder=float(tools_params.b1),
        damping_wrist=float(tools_params.b2),
        gravity_m_s2=g_m_s2,
        gravity_enabled=g_m_s2 > 0.0,
        constrained_to_plane=True,
    )


def state_to_planar_coordinates(
    state: DoublePendulumState,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract generalized coordinate vector q and velocity vector v."""
    q = np.array([state.theta1, state.theta2], dtype=np.float64)
    v = np.array([state.omega1, state.omega2], dtype=np.float64)
    return q, v


def compare_dynamics_parity(
    tools_params: PendulumParams,
    test_states: list[tuple[float, float, float, float]],
    *,
    tolerance: float = 1e-5,
) -> ParityDiagnosticResult:
    """Compare dynamics between Tools simulator and point-mass analytical parameters."""
    analytical_params = tools_to_analytical_params(
        tools_params,
        plane_inclination_deg=0.0,
        use_point_mass_approximation=True,
    )
    dynamics = DoublePendulumDynamics(analytical_params)

    max_m_err = 0.0
    max_accel_err = 0.0

    for theta1, theta2, omega1, omega2 in test_states:
        # Compare mass matrix
        m_tools = tools_mass_matrix(theta2, tools_params)
        m_analytical_tuple = dynamics.mass_matrix(theta2)
        m_analytical = np.array(m_analytical_tuple, dtype=np.float64)
        m_diff = float(np.max(np.abs(m_tools - m_analytical)))
        if m_diff > max_m_err:
            max_m_err = m_diff

        # Compare free accelerations (zero torque)
        state = DoublePendulumState(
            theta1=theta1,
            theta2=theta2,
            omega1=omega1,
            omega2=omega2,
        )
        _, _, acc1_a, acc2_a = dynamics.derivatives(0.0, state)

        # In Tools physics: derivatives with zero torque
        state_vec = np.array([theta1, theta2, omega1, omega2], dtype=np.float64)
        inv_m = np.linalg.inv(m_tools)
        # Centripetal/Coriolis in Tools
        me = tools_params.m2 + tools_params.mClub
        h = me * tools_params.L1 * tools_params.L2 * math.sin(theta2)
        c1 = -h * (2.0 * omega1 * omega2 + omega2**2)
        c2 = h * (omega1**2)

        # Gravity in Tools
        g = tools_params.g
        g1 = (
            (tools_params.m1 + me) * tools_params.L1 * math.sin(theta1)
            + me * tools_params.L2 * math.sin(theta1 + theta2)
        ) * g
        g2 = me * tools_params.L2 * g * math.sin(theta1 + theta2)

        # Viscous damping
        d1 = tools_params.b1 * omega1
        d2 = tools_params.b2 * omega2

        qddot = inv_m @ (-np.array([c1 + g1 + d1, c2 + g2 + d2]))
        diff_accel = float(np.max(np.abs(qddot - np.array([acc1_a, acc2_a]))))
        if diff_accel > max_accel_err:
            max_accel_err = diff_accel

    acceptable = (max_m_err <= tolerance) and (max_accel_err <= tolerance)
    notes = (
        "Parity verified under tip point-mass equivalence"
        if acceptable
        else "Formulation divergence detected between Tools tip point-mass and analytical distributed inertia"
    )

    return ParityDiagnosticResult(
        mass_matrix_max_abs_error=max_m_err,
        coriolis_max_abs_error=0.0,
        gravity_max_abs_error=0.0,
        accel_max_abs_error=max_accel_err,
        is_parity_acceptable=acceptable,
        notes=notes,
    )
