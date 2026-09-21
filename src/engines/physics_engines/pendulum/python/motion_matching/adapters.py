"""Convention and dynamics adapters between analytical and Tools double pendulums (TB-04 #10589).

Reconciles the analytical DoublePendulumDynamics/GolfModelParams implementation with the
shipped Tools simulator (src.shared.python.pendulum_simulator.physics) through explicit
parameter, state, FK, and torque convention adapters while keeping model identities distinct.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.model_params import GolfModelParams

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
    coriolis_vector as tools_coriolis,
    equations_of_motion,
    gravity_vector as tools_gravity,
    mass_matrix as tools_mass_matrix,
)

logger = logging.getLogger(__name__)

# Model identities per TB-00 (#10585)
MODEL_ID_ANALYTICAL: str = "driven_double_pendulum"
MODEL_ID_TOOLS: str = "driven_double_pendulum_tools"


def params_analytical_to_tools(
    params: DoublePendulumParameters | GolfModelParams,
) -> PendulumParams:
    """Convert analytical DoublePendulumParameters or GolfModelParams to Tools PendulumParams."""
    if isinstance(params, GolfModelParams):
        p_analytical = params.to_double_pendulum_parameters()
    else:
        p_analytical = params

    m1 = float(p_analytical.upper_segment.mass_kg)
    m2 = float(p_analytical.lower_segment.shaft_mass_kg)
    m_club = float(p_analytical.lower_segment.clubhead_mass_kg)
    l1 = float(p_analytical.upper_segment.length_m)
    l2 = float(p_analytical.lower_segment.length_m)
    g = float(p_analytical.projected_gravity)
    b1 = float(p_analytical.damping_shoulder)
    b2 = float(p_analytical.damping_wrist)

    return PendulumParams(
        m1=m1,
        m2=m2,
        L1=l1,
        L2=l2,
        mClub=m_club,
        g=g,
        b1=b1,
        b2=b2,
    )


def params_tools_to_analytical(
    params: PendulumParams,
    *,
    plane_inclination_deg: float = 0.0,
) -> DoublePendulumParameters:
    """Convert Tools PendulumParams to equivalent concentrated-mass DoublePendulumParameters.

    In the Tools formulation, links are idealized as concentrated masses at the joints:
    - Upper link mass m1 is concentrated at L1 (center_of_mass_ratio = 1.0, inertia_about_com = 0.0).
    - Lower link mass (shaft m2 + clubhead mClub) is concentrated at L2 (shaft_com_ratio = 1.0).
    Under this concentrated idealization, DoublePendulumDynamics achieves numerical parity
    with Tools physics to within 1e-10.
    """
    upper = SegmentProperties(
        length_m=float(params.L1),
        mass_kg=float(params.m1),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    lower = LowerSegmentProperties(
        length_m=float(params.L2),
        shaft_mass_kg=0.0,
        clubhead_mass_kg=float(params.m2) + float(params.mClub),
        shaft_com_ratio=1.0,
    )
    return DoublePendulumParameters(
        upper_segment=upper,
        lower_segment=lower,
        plane_inclination_deg=plane_inclination_deg,
        damping_shoulder=float(params.b1),
        damping_wrist=float(params.b2),
        gravity_m_s2=float(params.g),
        gravity_enabled=params.g > 0.0,
        constrained_to_plane=True,
    )


def forward_kinematics_2d(
    theta1: float,
    theta2: float,
    l1: float,
    l2: float,
    pivot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D in-plane forward kinematics for wrist (butt) and clubhead.

    Both analytical and Tools double pendulum use relative coordinates where:
    - theta1 is angle from downward vertical [0, -1], CCW positive.
    - theta2 is relative angle between arm and club, CCW positive.
    """
    p0 = np.zeros(2) if pivot is None else np.asarray(pivot, dtype=float)[:2]
    sin1 = math.sin(theta1)
    cos1 = math.cos(theta1)
    sin12 = math.sin(theta1 + theta2)
    cos12 = math.cos(theta1 + theta2)

    wrist = p0 + np.array([l1 * sin1, -l1 * cos1])
    head = wrist + np.array([l2 * sin12, -l2 * cos12])
    return wrist, head


def state_to_array(state: DoublePendulumState) -> np.ndarray:
    """Convert DoublePendulumState dataclass to length-4 state vector."""
    return np.array(
        [state.theta1, state.theta2, state.omega1, state.omega2], dtype=np.float64
    )


def array_to_state(arr: np.ndarray) -> DoublePendulumState:
    """Convert length-4 state vector to DoublePendulumState dataclass."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size < 4:
        raise ValueError(f"State array must have at least 4 elements, got {a.size}")
    return DoublePendulumState(
        theta1=float(a[0]),
        theta2=float(a[1]),
        omega1=float(a[2]),
        omega2=float(a[3]),
    )


class DoublePendulumAdapter:
    """Bidirectional adapter between Analytical and Tools double pendulum representations."""

    @staticmethod
    def to_tools_params(
        params: DoublePendulumParameters | GolfModelParams,
    ) -> PendulumParams:
        return params_analytical_to_tools(params)

    @staticmethod
    def to_analytical_params(
        params: PendulumParams,
        *,
        plane_inclination_deg: float = 0.0,
    ) -> DoublePendulumParameters:
        return params_tools_to_analytical(
            params, plane_inclination_deg=plane_inclination_deg
        )

    @staticmethod
    def forward_kinematics(
        theta1: float,
        theta2: float,
        l1: float,
        l2: float,
        pivot: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return forward_kinematics_2d(theta1, theta2, l1, l2, pivot)


def _make_forcing_fn(val: float) -> Callable[[float, DoublePendulumState], float]:
    def _forcing(_t: float, _s: DoublePendulumState) -> float:
        return val

    return _forcing


def _make_tools_torque_fn(
    t1: float, t2: float
) -> Callable[[float], tuple[float, float]]:
    def _torque(_t: float) -> tuple[float, float]:
        return (t1, t2)

    return _torque


def check_dynamics_parity(
    params_tools: PendulumParams | None = None,
) -> dict[str, Any]:
    """Verify numerical parity between analytical and Tools double pendulum dynamics."""
    p_tools = params_tools or PendulumParams(
        m1=5.0,
        m2=0.30,
        L1=0.65,
        L2=1.10,
        mClub=0.20,
        g=9.80665,
        b1=0.4,
        b2=0.25,
    )
    p_analytical = params_tools_to_analytical(p_tools)
    dyn_analytical = DoublePendulumDynamics(p_analytical)

    # Test grid of states and torques
    theta1_vals = [-1.5, -0.5, 0.0, 0.7, 1.8]
    theta2_vals = [-1.2, -0.2, 0.0, 0.5, 1.4]
    omega1_vals = [-5.0, 0.0, 8.0]
    omega2_vals = [-8.0, 0.0, 15.0]
    tau_vals = [(-100.0, -30.0), (0.0, 0.0), (120.0, 45.0)]

    max_mass_diff = 0.0
    max_acc_diff = 0.0

    for th1 in theta1_vals:
        for th2 in theta2_vals:
            # Mass matrix comparison
            m_tools = tools_mass_matrix(th2, p_tools)
            m_analytical = np.array(dyn_analytical.mass_matrix(th2))
            diff_m = float(np.max(np.abs(m_tools - m_analytical)))
            if diff_m > max_mass_diff:
                max_mass_diff = diff_m

            # Gravity vector comparison
            g_tools = tools_gravity(th1, th2, p_tools)
            g1_a, g2_a = dyn_analytical.gravity_vector(th1, th2)
            diff_g = float(np.max(np.abs(g_tools - [g1_a, g2_a])))
            if diff_g > max_acc_diff:
                max_acc_diff = diff_g

            for w1 in omega1_vals:
                for w2 in omega2_vals:
                    # Coriolis vector comparison
                    c_tools = tools_coriolis(th2, w1, w2, p_tools)
                    c1_a, c2_a = dyn_analytical.coriolis_vector(th2, w1, w2)
                    diff_c = float(np.max(np.abs(c_tools - [c1_a, c2_a])))
                    if diff_c > max_acc_diff:
                        max_acc_diff = diff_c

                    state_a = DoublePendulumState(
                        theta1=th1, theta2=th2, omega1=w1, omega2=w2
                    )

                    for tau1, tau2 in tau_vals:
                        t1_f = float(tau1)
                        t2_f = float(tau2)

                        dyn_analytical.forcing_functions = (
                            _make_forcing_fn(t1_f),
                            _make_forcing_fn(t2_f),
                        )
                        _, _, acc1_a, acc2_a = dyn_analytical.derivatives(0.0, state_a)

                        # Tools equations of motion
                        state_vec = np.array([th1, th2, w1, w2], dtype=np.float64)
                        eom_tools = equations_of_motion(
                            state_vec,
                            0.0,
                            p_tools,
                            _make_tools_torque_fn(t1_f, t2_f),
                        )
                        diff_acc = float(
                            np.max(np.abs(eom_tools[2:] - [acc1_a, acc2_a]))
                        )
                        if diff_acc > max_acc_diff:
                            max_acc_diff = diff_acc

    return {
        "max_mass_matrix_diff": max_mass_diff,
        "max_acceleration_diff": max_acc_diff,
        "parity_verified": max_acc_diff < 1e-9,
        "model_id_analytical": MODEL_ID_ANALYTICAL,
        "model_id_tools": MODEL_ID_TOOLS,
    }


def create_calibrated_double_pendulum_dynamics(
    l1: float, l2: float
) -> DoublePendulumDynamics:
    """Instantiate DoublePendulumDynamics configured with calibrated segment lengths."""
    dynamics = DoublePendulumDynamics()
    dynamics.parameters.upper_segment.length_m = float(l1)
    dynamics.parameters.lower_segment.length_m = float(l2)
    return dynamics
