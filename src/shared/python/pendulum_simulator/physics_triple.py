"""
Triple pendulum physics using Lagrangian formulation with relative coordinates.

Coordinate convention:
    q1 (theta1): Absolute angle of segment 1 from downward vertical,
                 positive counterclockwise.
    q2 (phi1):   Angle of segment 2 relative to segment 1,
                 positive counterclockwise.
    q3 (phi2):   Angle of segment 3 relative to segment 2,
                 positive counterclockwise.

All segments modeled as uniform rods with mass concentrated at the tip
(point-mass approximation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import native_backend as _native_backend
from .constants import GRAVITY_MSS

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriplePendulumParams:
    """Immutable physical parameters for the triple pendulum.

    Contract:
        - All lengths and masses must be strictly positive.
        - Gravity must be non-negative.
        - Damping coefficients b1, b2, b3 must be non-negative  (NÂ·mÂ·s/rad).
        - Coulomb friction mu1, mu2, mu3 must be non-negative  (NÂ·m peak magnitude).
    """

    m1: float  # mass of segment 1 (kg)
    m2: float  # mass of segment 2 (kg)
    m3: float  # mass of segment 3 (kg)
    L1: float  # length of segment 1 (m)
    L2: float  # length of segment 2 (m)
    L3: float  # length of segment 3 (m)
    g: float = GRAVITY_MSS  # gravitational acceleration (m/s^2)
    # --- Dissipative parameters (default 0 = no losses) ---
    b1: float = 0.0  # viscous damping at joint 1 (NÂ·mÂ·s/rad)
    b2: float = 0.0  # viscous damping at joint 2 (NÂ·mÂ·s/rad)
    b3: float = 0.0  # viscous damping at joint 3 (NÂ·mÂ·s/rad)
    mu1: float = 0.0  # Coulomb friction at joint 1 (NÂ·m, constant magnitude)
    mu2: float = 0.0  # Coulomb friction at joint 2 (NÂ·m, constant magnitude)
    mu3: float = 0.0  # Coulomb friction at joint 3 (NÂ·m, constant magnitude)
    # --- Scapula offset (#1152) ---
    scapula_offset_rad: float = 0.0  # angular offset of hub anchor (rad)

    def __post_init__(self) -> None:
        assert self.m1 > 0, f"m1 must be positive, got {self.m1}"
        assert self.m2 > 0, f"m2 must be positive, got {self.m2}"
        assert self.m3 > 0, f"m3 must be positive, got {self.m3}"
        assert self.L1 > 0, f"L1 must be positive, got {self.L1}"
        assert self.L2 > 0, f"L2 must be positive, got {self.L2}"
        assert self.L3 > 0, f"L3 must be positive, got {self.L3}"
        assert self.g >= 0, f"g must be non-negative, got {self.g}"
        assert self.b1 >= 0, f"b1 must be non-negative, got {self.b1}"
        assert self.b2 >= 0, f"b2 must be non-negative, got {self.b2}"
        assert self.b3 >= 0, f"b3 must be non-negative, got {self.b3}"
        assert self.mu1 >= 0, f"mu1 must be non-negative, got {self.mu1}"
        assert self.mu2 >= 0, f"mu2 must be non-negative, got {self.mu2}"
        assert self.mu3 >= 0, f"mu3 must be non-negative, got {self.mu3}"


# Type alias: state vector [theta1, phi1, phi2, dtheta1, dphi1, dphi2]
State = npt.NDArray[np.float64]  # shape (6,)

# Torque function signature: (t) -> (tau1, tau2, tau3)
TorqueFunc = Callable[[float], tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Mass matrix and its components
# ---------------------------------------------------------------------------


def mass_matrix(phi1: float, phi2: float, params: TriplePendulumParams) -> np.ndarray:
    """Compute the 3x3 mass (inertia) matrix M(q).

    Preconditions:
        - phi1 and phi2 are finite floats.
    Postconditions:
        - Returns a 3x3 symmetric positive-definite matrix.
        - M[i,j] == M[j,i]  (symmetry).
        - All eigenvalues > 0  (positive definiteness).

    Parameters
    ----------
    phi1 : float
        Relative angle of segment 2 w.r.t. segment 1 (rad).
    phi2 : float
        Relative angle of segment 3 w.r.t. segment 2 (rad).
    params : TriplePendulumParams

    Returns
    -------
    M : np.ndarray, shape (3, 3)
    """
    assert np.isfinite(phi1), f"phi1 must be finite, got {phi1}"
    assert np.isfinite(phi2), f"phi2 must be finite, got {phi2}"
    native_mass_matrix = _native_backend.triple_mass_matrix(phi1, phi2, params)
    if native_mass_matrix is not None:
        return native_mass_matrix

    m1, m2, m3 = params.m1, params.m2, params.m3
    L1, L2, L3 = params.L1, params.L2, params.L3

    c1 = np.cos(phi1)
    c2 = np.cos(phi2)
    c12 = np.cos(phi1 + phi2)

    # M11: inertia of segment 1 + contributions from 2 and 3
    M11 = (
        (m1 + m2 + m3) * L1**2
        + (m2 + m3) * L2**2
        + m3 * L3**2
        + 2.0 * (m2 + m3) * L1 * L2 * c1
        + 2.0 * m3 * L1 * L3 * c12
        + 2.0 * m3 * L2 * L3 * c2
    )

    # M12: coupling between segments 1 and 2
    M12 = (
        (m2 + m3) * L2**2
        + m3 * L3**2
        + (m2 + m3) * L1 * L2 * c1
        + m3 * L1 * L3 * c12
        + 2.0 * m3 * L2 * L3 * c2
    )

    # M13: coupling between segments 1 and 3
    M13 = m3 * L3**2 + m3 * L1 * L3 * c12 + m3 * L2 * L3 * c2

    # M22: self-coupling of segment 2
    M22 = (m2 + m3) * L2**2 + m3 * L3**2 + 2.0 * m3 * L2 * L3 * c2

    # M23: coupling between segments 2 and 3
    M23 = m3 * L3**2 + m3 * L2 * L3 * c2

    # M33: self-coupling of segment 3
    M33 = m3 * L3**2

    M = np.array(
        [
            [M11, M12, M13],
            [M12, M22, M23],
            [M13, M23, M33],
        ]
    )

    # Postcondition: symmetry
    for i in range(3):
        for j in range(3):
            assert np.isclose(
                M[i, j], M[j, i]
            ), f"Mass matrix not symmetric at [{i},{j}]"

    return M


def mass_matrix_components(
    phi1: float, phi2: float, params: TriplePendulumParams
) -> dict:
    """Return individual mass matrix terms with labels.

    Returns
    -------
    dict with keys M11..M33 and M_full.
    """
    assert phi1 is not None, "phi1 must be provided"
    M = mass_matrix(phi1, phi2, params)
    return {
        "M11": M[0, 0],
        "M12": M[0, 1],
        "M13": M[0, 2],
        "M21": M[1, 0],
        "M22": M[1, 1],
        "M23": M[1, 2],
        "M31": M[2, 0],
        "M32": M[2, 1],
        "M33": M[2, 2],
        "M_full": M,
    }


# ---------------------------------------------------------------------------
# Coriolis and centrifugal terms
# ---------------------------------------------------------------------------


def coriolis_vector(
    phi1: float,
    phi2: float,
    dtheta1: float,
    dphi1: float,
    dphi2: float,
    params: TriplePendulumParams,
) -> np.ndarray:
    """Compute the Coriolis/centrifugal force vector C(q, qdot).

    This captures velocity-dependent forces.

    Preconditions:
        - All inputs are finite floats.

    Parameters
    ----------
    phi1 : float
        Relative angle of segment 2 (rad).
    phi2 : float
        Relative angle of segment 3 (rad).
    dtheta1 : float
        Angular velocity of segment 1 (rad/s).
    dphi1 : float
        Relative angular velocity of segment 2 (rad/s).
    dphi2 : float
        Relative angular velocity of segment 3 (rad/s).
    params : TriplePendulumParams

    Returns
    -------
    C_qdot : np.ndarray, shape (3,)
    """
    assert all(
        np.isfinite(v) for v in [phi1, phi2, dtheta1, dphi1, dphi2]
    ), "All inputs must be finite"
    native_coriolis = _native_backend.triple_coriolis_vector(
        phi1, phi2, dtheta1, dphi1, dphi2, params
    )
    if native_coriolis is not None:
        return native_coriolis

    m2, m3 = params.m2, params.m3
    L1, L2, L3 = params.L1, params.L2, params.L3

    s1 = np.sin(phi1)
    s2 = np.sin(phi2)
    s12 = np.sin(phi1 + phi2)

    # Coupling terms (derivatives of mass matrix w.r.t. relative angles)
    h12 = -(m2 + m3) * L1 * L2 * s1
    h13 = -m3 * L1 * L3 * s12
    h23 = -m3 * L2 * L3 * s2

    # Coriolis/centrifugal terms from Christoffel symbols of M(q)
    c1 = (h12 + h13) * (2.0 * dtheta1 + dphi1) * dphi1 + (h13 + h23) * (
        2.0 * dtheta1 + 2.0 * dphi1 + dphi2
    ) * dphi2
    c2 = -(h12 + h13) * dtheta1**2 + h23 * (2.0 * dtheta1 + 2.0 * dphi1 + dphi2) * dphi2
    c3 = -(h13 + h23) * dtheta1**2 - h23 * (2.0 * dtheta1 + dphi1) * dphi1

    result = np.array([c1, c2, c3])
    assert all(np.isfinite(result)), f"Coriolis vector has non-finite values: {result}"
    return result


# ---------------------------------------------------------------------------
# Gravity vector
# ---------------------------------------------------------------------------


def gravity_vector(
    theta1: float, phi1: float, phi2: float, params: TriplePendulumParams
) -> np.ndarray:
    """Compute the gravitational torque vector G(q).

    Derived from potential energy.

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 (rad).
    phi1 : float
        Relative angle of segment 2 (rad).
    phi2 : float
        Relative angle of segment 3 (rad).
    params : TriplePendulumParams

    Returns
    -------
    G : np.ndarray, shape (3,)
    """
    native_gravity = _native_backend.triple_gravity_vector(theta1, phi1, phi2, params)
    if native_gravity is not None:
        return native_gravity

    m1, m2, m3 = params.m1, params.m2, params.m3
    L1, L2, L3 = params.L1, params.L2, params.L3
    g = params.g

    abs_angle2 = theta1 + phi1
    abs_angle3 = theta1 + phi1 + phi2

    G1 = (
        (m1 + m2 + m3) * g * L1 * np.sin(theta1)
        + (m2 + m3) * g * L2 * np.sin(abs_angle2)
        + m3 * g * L3 * np.sin(abs_angle3)
    )

    G2 = (m2 + m3) * g * L2 * np.sin(abs_angle2) + m3 * g * L3 * np.sin(abs_angle3)

    G3 = m3 * g * L3 * np.sin(abs_angle3)

    result = np.array([G1, G2, G3])
    assert all(np.isfinite(result)), f"Gravity vector has non-finite values: {result}"
    return result


# ---------------------------------------------------------------------------
# Friction and damping
# ---------------------------------------------------------------------------


def friction_torque_vector(
    dtheta1: float, dphi1: float, dphi2: float, params: TriplePendulumParams
) -> np.ndarray:
    """Compute the total dissipative torque vector at the joints.

    Combines viscous (linear) damping and Coulomb (constant) friction.
    Both always oppose the direction of motion.

    Model:
        tau_friction_i = -b_i * qdot_i - mu_i * sign(qdot_i)

    Preconditions:
        - All velocities are finite.
    Postconditions:
        - Returns shape (3,), all values finite.

    Returns
    -------
    tau_f : np.ndarray, shape (3,)  [NÂ·m]
    """
    assert np.isfinite(dtheta1), f"dtheta1 must be finite, got {dtheta1}"
    assert np.isfinite(dphi1), f"dphi1 must be finite, got {dphi1}"
    assert np.isfinite(dphi2), f"dphi2 must be finite, got {dphi2}"

    tau_f1 = -params.b1 * dtheta1 - params.mu1 * np.sign(dtheta1)
    tau_f2 = -params.b2 * dphi1 - params.mu2 * np.sign(dphi1)
    tau_f3 = -params.b3 * dphi2 - params.mu3 * np.sign(dphi2)

    result = np.array([tau_f1, tau_f2, tau_f3])
    assert all(np.isfinite(result)), f"Friction torque has non-finite values: {result}"
    return result


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------


def equations_of_motion(
    state: State,
    t: float,
    params: TriplePendulumParams,
    torque_func: TorqueFunc,
    torque_limits: np.ndarray | None = None,
) -> State:
    """Compute the state derivative: dx/dt = f(x, t).

    State vector x = [theta1, phi1, phi2, dtheta1, dphi1, dphi2].

    M(q) * qddot = tau - C(q,qdot) - G(q)

    Preconditions:
        - state has shape (6,) with all finite values.
        - torque_func returns a 3-tuple of finite floats.
        - torque_limits, if provided, has shape (3,) with positive values.
    Postconditions:
        - Returns shape (6,) with all finite values.

    Parameters
    ----------
    state : np.ndarray, shape (6,)
    t : float
    params : TriplePendulumParams
    torque_func : callable (t) -> (tau1, tau2, tau3)
    torque_limits : np.ndarray, shape (3,), optional
        Per-joint torque saturation limits (#1150).

    Returns
    -------
    state_dot : np.ndarray, shape (6,)
    """
    assert state.shape == (6,), f"State must have shape (6,), got {state.shape}"
    assert all(np.isfinite(state)), f"State values must be finite: {state}"

    theta1, phi1, phi2, dtheta1, dphi1, dphi2 = state

    M = mass_matrix(phi1, phi2, params)
    C = coriolis_vector(phi1, phi2, dtheta1, dphi1, dphi2, params)
    G = gravity_vector(theta1, phi1, phi2, params)

    tau1, tau2, tau3 = torque_func(t)
    tau = np.array([tau1, tau2, tau3])

    # Torque saturation (#1150)
    if torque_limits is not None:
        from .physics import clamp_torque_ndof

        tau = clamp_torque_ndof(tau, torque_limits)

    tau_friction = friction_torque_vector(dtheta1, dphi1, dphi2, params)

    # Solve: M * qddot = tau + tau_friction - C - G
    rhs = tau + tau_friction - C - G
    qddot = np.linalg.solve(M, rhs)

    state_dot = np.array([dtheta1, dphi1, dphi2, qddot[0], qddot[1], qddot[2]])

    assert all(
        np.isfinite(state_dot)
    ), f"State derivative has non-finite values: {state_dot}"
    return state_dot


# ---------------------------------------------------------------------------
# Forward kinematics (for visualization)
# ---------------------------------------------------------------------------


def forward_kinematics(
    theta1: float, phi1: float, phi2: float, params: TriplePendulumParams
) -> dict:
    """Compute joint and tip positions in the world frame.

    Origin is at the hub (fixed pivot).
    x-axis points right, y-axis points up.

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 from downward vertical (rad).
    phi1 : float
        Relative angle of segment 2 (rad).
    phi2 : float
        Relative angle of segment 3 (rad).
    params : TriplePendulumParams

    Returns
    -------
    dict with 'hub', 'shoulder', 'wrist1', 'wrist2', 'tip' as (x, y) tuples.
    The shoulder is displaced from the hub by the scapula offset (#1152).
    """
    assert theta1 is not None, "theta1 must be provided"
    native_positions = _native_backend.triple_forward_kinematics(
        theta1, phi1, phi2, params
    )
    if native_positions is not None:
        return native_positions

    L1, L2, L3 = params.L1, params.L2, params.L3

    # Scapula offset (#1152): displaces shoulder anchor from hub
    # Only applies displacement when offset is non-zero.
    scap = params.scapula_offset_rad
    if abs(scap) > 1e-8:
        scap_len = L1 * 0.3  # scapula link length (30% of hub segment)
        ox = scap_len * np.sin(scap)
        oy = -scap_len * (1.0 - np.cos(scap))  # zero when scap=0
    else:
        ox = 0.0
        oy = 0.0

    abs_angle2 = theta1 + phi1
    abs_angle3 = theta1 + phi1 + phi2

    # Segment 1 endpoint (wrist1 / first joint)
    w1x = ox + L1 * np.sin(theta1)
    w1y = oy - L1 * np.cos(theta1)

    # Segment 2 endpoint (wrist2 / second joint)
    w2x = w1x + L2 * np.sin(abs_angle2)
    w2y = w1y - L2 * np.cos(abs_angle2)

    # Segment 3 endpoint (tip)
    tx = w2x + L3 * np.sin(abs_angle3)
    ty = w2y - L3 * np.cos(abs_angle3)

    return {
        "hub": (0.0, 0.0),
        "shoulder": (ox, oy),
        "wrist1": (w1x, w1y),
        "wrist2": (w2x, w2y),
        "tip": (tx, ty),
    }


def linear_accelerations(
    state: State, qddot: np.ndarray, params: TriplePendulumParams
) -> dict:
    """Compute linear accelerations of joints in world coordinates.

    Returns
    -------
    dict with keys: 'wrist1', 'wrist2', 'tip' as (ax, ay) tuples.
    """
    assert state.shape == (6,), f"State must have shape (6,), got {state.shape}"
    assert qddot.shape == (3,), f"qddot must have shape (3,), got {qddot.shape}"

    theta1, phi1, phi2, dtheta1, dphi1, dphi2 = state
    ddtheta1, ddphi1, ddphi2 = qddot

    L1, L2, L3 = params.L1, params.L2, params.L3
    abs_angle2 = theta1 + phi1
    abs_angle3 = theta1 + phi1 + phi2

    dabs2 = dtheta1 + dphi1
    ddabs2 = ddtheta1 + ddphi1
    dabs3 = dtheta1 + dphi1 + dphi2
    ddabs3 = ddtheta1 + ddphi1 + ddphi2

    ax_w1 = L1 * (-np.sin(theta1) * dtheta1**2 + np.cos(theta1) * ddtheta1)
    ay_w1 = L1 * (np.cos(theta1) * dtheta1**2 + np.sin(theta1) * ddtheta1)

    ax_w2 = ax_w1 + L2 * (-np.sin(abs_angle2) * dabs2**2 + np.cos(abs_angle2) * ddabs2)
    ay_w2 = ay_w1 + L2 * (np.cos(abs_angle2) * dabs2**2 + np.sin(abs_angle2) * ddabs2)

    ax_t = ax_w2 + L3 * (-np.sin(abs_angle3) * dabs3**2 + np.cos(abs_angle3) * ddabs3)
    ay_t = ay_w2 + L3 * (np.cos(abs_angle3) * dabs3**2 + np.sin(abs_angle3) * ddabs3)

    return {
        "wrist1": (ax_w1, ay_w1),
        "wrist2": (ax_w2, ay_w2),
        "tip": (ax_t, ay_t),
    }


def net_joint_forces(
    state: State, qddot: np.ndarray, params: TriplePendulumParams
) -> dict:
    """Compute net joint forces (proximal on distal) in world coordinates.

    Returns
    -------
    dict with keys: 'shoulder', 'wrist1', 'wrist2' as (fx, fy) tuples.
    """
    assert state is not None, "state must be provided"
    acc = linear_accelerations(state, qddot, params)
    g_vec = np.array([0.0, -params.g])

    m1, m2, m3 = params.m1, params.m2, params.m3
    a_w1 = np.array(acc["wrist1"])
    a_w2 = np.array(acc["wrist2"])
    a_t = np.array(acc["tip"])

    f_wrist2 = m3 * a_t - m3 * g_vec
    f_wrist1 = (m2 * a_w2 + m3 * a_t) - (m2 + m3) * g_vec
    f_shoulder = (m1 * a_w1 + m2 * a_w2 + m3 * a_t) - (m1 + m2 + m3) * g_vec

    return {
        "shoulder": (float(f_shoulder[0]), float(f_shoulder[1])),
        "wrist1": (float(f_wrist1[0]), float(f_wrist1[1])),
        "wrist2": (float(f_wrist2[0]), float(f_wrist2[1])),
    }


# ---------------------------------------------------------------------------
# Energy calculations
# ---------------------------------------------------------------------------


def kinetic_energy(state: State, params: TriplePendulumParams) -> float:
    """Compute total kinetic energy T = 0.5 * qdot^T M qdot."""
    assert state is not None, "state must be provided"
    from .physics_base import kinetic_energy_from_M

    theta1, phi1, phi2, dtheta1, dphi1, dphi2 = state
    M = mass_matrix(phi1, phi2, params)
    qdot = np.array([dtheta1, dphi1, dphi2])
    return kinetic_energy_from_M(M, qdot)


def potential_energy(state: State, params: TriplePendulumParams) -> float:
    """Compute total potential energy."""
    assert state is not None, "state must be provided"
    theta1, phi1, phi2, _, _, _ = state

    m1, m2, m3 = params.m1, params.m2, params.m3
    L1, L2, L3 = params.L1, params.L2, params.L3
    g = params.g

    abs_angle2 = theta1 + phi1
    abs_angle3 = theta1 + phi1 + phi2

    # Taking the shoulder as reference (zero potential energy)
    V = (
        -m1 * g * L1 * np.cos(theta1)
        - m2 * g * (L1 * np.cos(theta1) + L2 * np.cos(abs_angle2))
        - m3
        * g
        * (L1 * np.cos(theta1) + L2 * np.cos(abs_angle2) + L3 * np.cos(abs_angle3))
    )

    return float(V)


def total_energy(state: State, params: TriplePendulumParams) -> float:
    """Compute total energy E = T + V."""
    assert state is not None, "state must be provided"
    from .physics_base import total_energy_from_parts

    return total_energy_from_parts(
        kinetic_energy(state, params), potential_energy(state, params)
    )
