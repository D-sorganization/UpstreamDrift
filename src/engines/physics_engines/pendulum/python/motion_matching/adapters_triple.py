"""Convention and dynamics adapters between analytical and Tools triple pendulums (TB-05 #10590).

Reconciles the analytical TriplePendulumDynamics implementation with the
shipped Tools simulator (src.shared.python.pendulum_simulator.physics_triple) through explicit
parameter, state, FK, and torque convention adapters while keeping model identities distinct.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumParameters,
    TriplePendulumState,
    TripleSegmentProperties,
)
from src.shared.python.pendulum_simulator.physics_triple import (
    TriplePendulumParams,
    coriolis_vector as tools_coriolis,
    equations_of_motion as tools_equations_of_motion,
    gravity_vector as tools_gravity,
    mass_matrix as tools_mass_matrix,
)

logger = logging.getLogger(__name__)

# Model identities per TB-00 (#10585) and registry (#10587)
MODEL_ID_TRIPLE_ANALYTICAL: str = "driven_triple_pendulum"
MODEL_ID_TRIPLE_TOOLS: str = "driven_triple_pendulum_tools"


def params_analytical_to_tools_triple(
    params: TriplePendulumParameters,
) -> TriplePendulumParams:
    """Convert analytical TriplePendulumParameters to Tools TriplePendulumParams."""
    segs = params.segments
    s0 = segs[0]
    s1 = segs[1]
    s2 = segs[2]

    damp = params.damping
    b1_val = float(damp[0])
    b2_val = float(damp[1])
    b3_val = float(damp[2])

    return TriplePendulumParams(
        m1=float(s0.mass_kg),
        m2=float(s1.mass_kg),
        m3=float(s2.mass_kg),
        L1=float(s0.length_m),
        L2=float(s1.length_m),
        L3=float(s2.length_m),
        g=float(params.gravity),
        b1=b1_val,
        b2=b2_val,
        b3=b3_val,
    )


def params_tools_to_analytical_triple(
    params: TriplePendulumParams,
) -> TriplePendulumParameters:
    """Convert Tools TriplePendulumParams to concentrated-mass TriplePendulumParameters."""
    s1 = TripleSegmentProperties(
        length_m=float(params.L1),
        mass_kg=float(params.m1),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    s2 = TripleSegmentProperties(
        length_m=float(params.L2),
        mass_kg=float(params.m2),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    s3 = TripleSegmentProperties(
        length_m=float(params.L3),
        mass_kg=float(params.m3),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    damping = (float(params.b1), float(params.b2), float(params.b3))
    return TriplePendulumParameters(
        segments=(s1, s2, s3),
        damping=damping,
        gravity_enabled=params.g > 0.0,
        gravity_m_s2=float(params.g),
    )


def forward_kinematics_3dof(
    theta1: float,
    theta2: float,
    theta3: float,
    l1: float,
    l2: float,
    l3: float,
    pivot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D in-plane forward kinematics for shoulder, wrist (grip), and tip (head).

    Coordinates:
    - theta1: angle of segment 1 from downward vertical [0, -1] CCW.
    - theta2: relative angle of segment 2 w.r.t segment 1 CCW.
    - theta3: relative angle of segment 3 w.r.t segment 2 CCW.
    """
    p0 = np.zeros(2) if pivot is None else np.asarray(pivot, dtype=float)[:2]
    sin1 = math.sin(theta1)
    cos1 = math.cos(theta1)
    sin12 = math.sin(theta1 + theta2)
    cos12 = math.cos(theta1 + theta2)
    sin123 = math.sin(theta1 + theta2 + theta3)
    cos123 = math.cos(theta1 + theta2 + theta3)

    shoulder = p0 + np.array([l1 * sin1, -l1 * cos1])
    wrist = shoulder + np.array([l2 * sin12, -l2 * cos12])
    head = wrist + np.array([l3 * sin123, -l3 * cos123])
    return shoulder, wrist, head


class TriplePendulumAdapter:
    """Bidirectional adapter between Analytical and Tools triple pendulum representations."""

    @staticmethod
    def to_tools_params(params: TriplePendulumParameters) -> TriplePendulumParams:
        return params_analytical_to_tools_triple(params)

    @staticmethod
    def to_analytical_params(params: TriplePendulumParams) -> TriplePendulumParameters:
        return params_tools_to_analytical_triple(params)

    @staticmethod
    def forward_kinematics(
        theta1: float,
        theta2: float,
        theta3: float,
        l1: float,
        l2: float,
        l3: float,
        pivot: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return forward_kinematics_3dof(theta1, theta2, theta3, l1, l2, l3, pivot)


def create_calibrated_triple_pendulum_dynamics(
    l1: float,
    l2: float,
    l3: float,
    m1: float = 2.0,
    m2: float = 1.5,
    m3: float = 0.35,
    damping: tuple[float, float, float] = (0.35, 0.30, 0.25),
) -> TriplePendulumDynamics:
    """Instantiate TriplePendulumDynamics configured with positive calibrated lengths."""
    s1 = TripleSegmentProperties(
        length_m=float(l1),
        mass_kg=float(m1),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    s2 = TripleSegmentProperties(
        length_m=float(l2),
        mass_kg=float(m2),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    s3 = TripleSegmentProperties(
        length_m=float(l3),
        mass_kg=float(m3),
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    params = TriplePendulumParameters(
        segments=(s1, s2, s3),
        damping=damping,
        gravity_enabled=True,
    )
    return TriplePendulumDynamics(params)


def check_triple_dynamics_parity(
    params_tools: TriplePendulumParams | None = None,
) -> dict[str, Any]:
    """Verify numerical parity between analytical and Tools triple pendulum dynamics."""
    p_tools = params_tools or TriplePendulumParams(
        m1=2.5,
        m2=1.8,
        m3=0.38,
        L1=0.28,
        L2=0.52,
        L3=1.05,
        g=9.80665,
        b1=0.35,
        b2=0.30,
        b3=0.25,
    )
    p_analytical = params_tools_to_analytical_triple(p_tools)
    dyn_analytical = TriplePendulumDynamics(p_analytical)

    theta1_vals = [-1.2, 0.0, 0.8]
    theta2_vals = [-0.5, 0.0, 0.6]
    theta3_vals = [-0.8, 0.0, 1.1]
    omega1_vals = [-2.0, 0.0, 3.5]
    omega2_vals = [-1.5, 0.0, 4.0]
    omega3_vals = [-3.0, 0.0, 6.0]
    tau_vals = [(-50.0, 20.0, -10.0), (0.0, 0.0, 0.0), (80.0, -40.0, 25.0)]

    max_mass_diff = 0.0
    max_acc_diff = 0.0

    for th1 in theta1_vals:
        for th2 in theta2_vals:
            for th3 in theta3_vals:
                # Mass matrix comparison
                m_tools = tools_mass_matrix(th2, th3, p_tools)
                state_temp = TriplePendulumState(
                    theta1=th1,
                    theta2=th2,
                    theta3=th3,
                    omega1=0.0,
                    omega2=0.0,
                    omega3=0.0,
                )
                m_analytical = dyn_analytical.mass_matrix(state_temp)
                diff_m = float(np.max(np.abs(m_tools - m_analytical)))
                if diff_m > max_mass_diff:
                    max_mass_diff = diff_m

                for w1, w2, w3 in zip(
                    omega1_vals, omega2_vals, omega3_vals, strict=True
                ):
                    state_a = TriplePendulumState(
                        theta1=th1,
                        theta2=th2,
                        theta3=th3,
                        omega1=w1,
                        omega2=w2,
                        omega3=w3,
                    )
                    state_vec = np.array([th1, th2, th3, w1, w2, w3], dtype=np.float64)

                    for tau in tau_vals:
                        acc_a = dyn_analytical.forward_dynamics(state_a, tau)

                        def _torque_fn(
                            _t: float, applied_tau=tau
                        ) -> tuple[float, float, float]:
                            return applied_tau

                        eom_tools = tools_equations_of_motion(
                            state_vec, 0.0, p_tools, _torque_fn
                        )
                        diff_acc = float(
                            np.max(np.abs(eom_tools[3:] - np.array(acc_a)))
                        )
                        if diff_acc > max_acc_diff:
                            max_acc_diff = diff_acc

    return {
        "max_mass_matrix_diff": max_mass_diff,
        "max_acceleration_diff": max_acc_diff,
        "parity_verified": bool(max_acc_diff < 1e-9 and max_mass_diff < 1e-11),
        "model_id_analytical": MODEL_ID_TRIPLE_ANALYTICAL,
        "model_id_tools": MODEL_ID_TRIPLE_TOOLS,
    }
