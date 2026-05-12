# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Double pendulum golf swing physics using Lagrangian formulation with relative coordinates.

Model: 2-segment pendulum (arms + shaft) with clubhead point mass at tip.

Coordinate convention:
    q1 (theta1): Absolute angle of segment 1 (arms) from downward vertical,
                 positive counterclockwise.
    q2 (phi):    Angle of segment 2 (shaft) relative to segment 1,
                 positive counterclockwise.

    When q1=0 and q2=0, both segments hang straight down (equilibrium).

Segments:
    Segment 1 = "Arms" (shoulder to wrist)
    Segment 2 = "Shaft" (wrist to clubhead)
    Clubhead  = point mass at tip of shaft
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import native_backend as _native_backend
from .constants import GRAVITY_MSS
from .physics_base import clamp_torque_ndof  # noqa: F401  (re-export, DRY)

_log = logging.getLogger(__name__)

# Condition number threshold above which a warning is issued for M.
_MASS_MATRIX_COND_WARN = 1e12

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendulumParams:
    """Immutable physical parameters for the double pendulum golf model.

    Contract:
        - All lengths and masses must be strictly positive (mClub >= 0).
        - Gravity must be non-negative.
        - Damping coefficients b1, b2 must be non-negative  (NÂ·mÂ·s/rad).
        - Coulomb friction mu1, mu2 must be non-negative  (NÂ·m peak magnitude).
    """

    m1: float  # mass of arms (kg), typical ~5.0
    m2: float  # mass of shaft (kg), typical ~0.30
    L1: float  # length of arms (m), typical ~0.65
    L2: float  # length of shaft (m), typical ~1.10
    mClub: float = 0.0  # clubhead mass (kg), typical ~0.20
    g: float = GRAVITY_MSS  # gravitational acceleration (m/sÂ²)
    b1: float = 0.0  # viscous damping at shoulder (NÂ·mÂ·s/rad)
    b2: float = 0.0  # viscous damping at wrist (NÂ·mÂ·s/rad)
    mu1: float = 0.0  # Coulomb friction at shoulder (NÂ·m)
    mu2: float = 0.0  # Coulomb friction at wrist (NÂ·m)

    def __post_init__(self) -> None:
        assert self.m1 > 0, f"m1 must be positive, got {self.m1}"
        assert self.m2 > 0, f"m2 must be positive, got {self.m2}"
        assert self.L1 > 0, f"L1 must be positive, got {self.L1}"
        assert self.L2 > 0, f"L2 must be positive, got {self.L2}"
        assert self.mClub >= 0, f"mClub must be non-negative, got {self.mClub}"
        assert self.g >= 0, f"g must be non-negative, got {self.g}"
        assert self.b1 >= 0, f"b1 must be non-negative, got {self.b1}"
        assert self.b2 >= 0, f"b2 must be non-negative, got {self.b2}"
        assert self.mu1 >= 0, f"mu1 must be non-negative, got {self.mu1}"
        assert self.mu2 >= 0, f"mu2 must be non-negative, got {self.mu2}"


@dataclass(frozen=True)
class JointLimits:
    """Joint angle limits for both shoulder (theta1) and wrist (phi).

    Contract:
        - For each pair, min < max.
        - stiffness > 0, damping >= 0.
        - theta1 limits default to Â±Ï€ (unconstrained).
    """

    # Wrist (phi) limits
    phi_min: float = -np.pi / 2  # rad
    phi_max: float = np.pi / 2  # rad

    # Shoulder (theta1) limits â€” defaults allow full rotation
    theta1_min: float = -np.pi  # rad
    theta1_max: float = np.pi  # rad

    # Shared penalty parameters
    stiffness: float = 500.0  # NÂ·m/rad
    damping: float = 20.0  # NÂ·mÂ·s/rad

    def __post_init__(self) -> None:
        assert self.phi_min < self.phi_max
        assert self.theta1_min < self.theta1_max
        assert self.stiffness > 0
        assert self.damping >= 0


@dataclass(frozen=True)
class TorqueClamp:
    """Torque saturation limits (symmetric Â± clamp).

    Contract:
        - Both limits must be non-zero; abs() is applied automatically.
        - Clamped range: [-|max_torque|, +|max_torque|] per DOF.

    Closes #1138: accepts negative values via abs() for usability.
    """

    max_torque1: float = float("inf")  # NÂ·m (magnitude, Â± symmetric)
    max_torque2: float = float("inf")  # NÂ·m (magnitude, Â± symmetric)

    def __post_init__(self) -> None:
        # Accept negative inputs by taking abs (#1138)
        object.__setattr__(self, "max_torque1", abs(self.max_torque1))
        object.__setattr__(self, "max_torque2", abs(self.max_torque2))
        assert (
            self.max_torque1 > 0
        ), f"|max_torque1| must be positive, got {self.max_torque1}"
        assert (
            self.max_torque2 > 0
        ), f"|max_torque2| must be positive, got {self.max_torque2}"


# Type aliases
State = npt.NDArray[np.float64]  # shape (4,)
TorqueFunc = Callable[[float], tuple[float, float]]


# ---------------------------------------------------------------------------
# Effective mass helper (DRY)
# ---------------------------------------------------------------------------


def _m2eff(params: PendulumParams) -> float:
    """Effective mass of segment 2: shaft + clubhead."""
    return params.m2 + params.mClub


# ---------------------------------------------------------------------------
# Mass matrix and its components
# ---------------------------------------------------------------------------


def mass_matrix(phi: float, params: PendulumParams) -> np.ndarray:
    """Compute the 2x2 mass (inertia) matrix M(q).

    Clubhead is modeled as a point mass at the tip of the shaft.

    Pre: phi is finite.
    Post: symmetric positive-definite 2x2 matrix.
    """
    assert np.isfinite(phi), f"phi must be finite, got {phi}"
    native_mass_matrix = _native_backend.double_mass_matrix(phi, params)
    if native_mass_matrix is not None:
        return native_mass_matrix

    m1, L1, L2 = params.m1, params.L1, params.L2
    me = _m2eff(params)
    cos_phi = np.cos(phi)

    M11 = (m1 + me) * L1**2 + me * L2**2 + 2.0 * me * L1 * L2 * cos_phi
    M12 = me * L2**2 + me * L1 * L2 * cos_phi
    M22 = me * L2**2

    M = np.array([[M11, M12], [M12, M22]])
    assert np.isclose(M[0, 1], M[1, 0]), "Mass matrix must be symmetric"
    return M


def mass_matrix_components(phi: float, params: PendulumParams) -> dict:
    """Return individual terms with physical labels."""
    assert phi is not None, "phi must be provided"
    M = mass_matrix(phi, params)
    return {
        "M11": M[0, 0],
        "M12": M[0, 1],
        "M21": M[1, 0],
        "M22": M[1, 1],
        "M_full": M,
    }


# ---------------------------------------------------------------------------
# Coriolis and centrifugal terms
# ---------------------------------------------------------------------------


def coriolis_vector(
    phi: float, dtheta1: float, dphi: float, params: PendulumParams
) -> np.ndarray:
    """Compute the Coriolis/centrifugal force vector C(q, qdot) * qdot.

    Pre: all inputs finite.
    """
    assert all(np.isfinite(v) for v in [phi, dtheta1, dphi])
    native_coriolis = _native_backend.double_coriolis_vector(phi, dtheta1, dphi, params)
    if native_coriolis is not None:
        return native_coriolis

    me = _m2eff(params)
    h = -me * params.L1 * params.L2 * np.sin(phi)
    c1 = h * (2.0 * dtheta1 * dphi + dphi**2)
    c2 = -h * dtheta1**2
    result = np.array([c1, c2])
    assert all(np.isfinite(result)), f"Coriolis vector has non-finite values: {result}"
    return result


# ---------------------------------------------------------------------------
# Gravity vector
# ---------------------------------------------------------------------------


def gravity_vector(theta1: float, phi: float, params: PendulumParams) -> np.ndarray:
    """Compute the gravitational torque vector G(q)."""
    native_gravity = _native_backend.double_gravity_vector(theta1, phi, params)
    if native_gravity is not None:
        return native_gravity

    me = _m2eff(params)
    L1, L2, g = params.L1, params.L2, params.g
    abs_angle2 = theta1 + phi
    G1 = (params.m1 + me) * g * L1 * np.sin(theta1) + me * g * L2 * np.sin(abs_angle2)
    G2 = me * g * L2 * np.sin(abs_angle2)

    result = np.array([G1, G2])
    assert all(np.isfinite(result)), f"Gravity vector has non-finite values: {result}"
    return result


# ---------------------------------------------------------------------------
# Friction and damping
# ---------------------------------------------------------------------------


def friction_torque_vector(
    dtheta1: float, dphi: float, params: PendulumParams
) -> np.ndarray:
    """Compute dissipative torque vector (viscous + Coulomb).

    Pre: dtheta1, dphi finite.
    Post: opposes motion direction.
    """
    assert np.isfinite(dtheta1) and np.isfinite(dphi)
    tau_f1 = -params.b1 * dtheta1 - params.mu1 * np.sign(dtheta1)
    tau_f2 = -params.b2 * dphi - params.mu2 * np.sign(dphi)
    result = np.array([tau_f1, tau_f2])
    assert all(np.isfinite(result)), f"Friction torque has non-finite values: {result}"
    return result


# ---------------------------------------------------------------------------
# Joint limit penalty torque (smooth barrier)
# ---------------------------------------------------------------------------


def _hermite_penalty(
    pen: float, vel: float, transition: float, stiffness: float, damping: float
) -> float:
    """Hermite smoothstep penalty magnitude for a single DOF at penetration depth ``pen``.

    The smoothstep ramps from 0 (at pen=0) to 1 (at pen>=transition), providing
    a C1-continuous blend from the free region into the full penalty region.

    Parameters
    ----------
    pen : float
        Penetration depth (positive, in radians).
    vel : float
        Signed joint velocity (rad/s); only the component into the limit is penalised.
    transition : float
        Transition width (rad) over which smoothstep blends from 0 to 1.
    stiffness : float
        Spring constant (NÂ·m/rad).
    damping : float
        Damping coefficient (NÂ·mÂ·s/rad); acts only when velocity pushes further into limit.

    Returns
    -------
    float
        Non-negative penalty magnitude (caller applies sign based on limit side).
    """
    assert pen is not None, "pen must be provided"
    blend = min(1.0, pen / transition)
    smooth = blend * blend * (3 - 2 * blend)
    return smooth * (stiffness * pen + damping * max(0.0, vel))


def joint_limit_torque(
    phi: float,
    dphi: float,
    limits: JointLimits,
    theta1: float = 0.0,
    dtheta1: float = 0.0,
) -> np.ndarray:
    """Smooth joint limit penalty using Hermite smoothstep.

    Pre: all angle/velocity args finite.
    Post: penalty is 0 when angles are within limits.
    Returns shape (2,): [tau_penalty_shoulder, tau_penalty_wrist].
    """
    assert np.isfinite(phi) and np.isfinite(dphi)
    transition = 0.05  # rad (~3 degrees)

    def _penalty(angle: float, vel: float, lo: float, hi: float) -> float:
        """Compute signed smoothstep penalty for a single DOF."""
        assert angle is not None, "angle must be provided"
        if angle < lo:
            return _hermite_penalty(
                lo - angle, -vel, transition, limits.stiffness, limits.damping
            )
        if angle > hi:
            return -_hermite_penalty(
                angle - hi, vel, transition, limits.stiffness, limits.damping
            )
        return 0.0

    tau1 = _penalty(theta1, dtheta1, limits.theta1_min, limits.theta1_max)
    tau2 = _penalty(phi, dphi, limits.phi_min, limits.phi_max)

    return np.array([tau1, tau2])


# ---------------------------------------------------------------------------
# Torque clamping
# ---------------------------------------------------------------------------


def clamp_torque(tau: np.ndarray, clamp: TorqueClamp) -> np.ndarray:
    """Clamp torques to saturation limits.

    Pre: tau shape (2,), clamp limits > 0.
    Post: |result[i]| <= limit[i].
    """
    return np.array(
        [
            np.clip(tau[0], -clamp.max_torque1, clamp.max_torque1),
            np.clip(tau[1], -clamp.max_torque2, clamp.max_torque2),
        ]
    )


@dataclass(frozen=True)
class JointLimitsNDOF:
    """N-DOF joint angle limits with Hermite smoothstep penalties.

    Generalisation of ``JointLimits`` (which is 2-DOF specific) to
    arbitrary joint counts.  Used by the triple and golfer models.

    Contract:
        - angle_min.shape == angle_max.shape == (n_dof,)
        - angle_min[i] < angle_max[i]  for all i
        - stiffness > 0, damping >= 0
    """

    angle_min: np.ndarray  # (n_dof,) in radians
    angle_max: np.ndarray  # (n_dof,) in radians
    stiffness: float = 500.0  # NÂ·m/rad
    damping: float = 20.0  # NÂ·mÂ·s/rad

    def __post_init__(self) -> None:
        assert self.angle_min.ndim == 1, "angle_min must be 1D"
        assert self.angle_max.ndim == 1, "angle_max must be 1D"
        assert self.angle_min.shape == self.angle_max.shape, "Shape mismatch"
        assert np.all(
            self.angle_min < self.angle_max
        ), "min must be < max for all joints"
        assert self.stiffness > 0, f"stiffness must be positive, got {self.stiffness}"
        assert self.damping >= 0, f"damping must be non-negative, got {self.damping}"


def joint_limit_torque_ndof(
    angles: np.ndarray,
    velocities: np.ndarray,
    limits: JointLimitsNDOF,
) -> np.ndarray:
    """Smooth joint limit penalty using Hermite smoothstep, N-DOF.

    Pre: angles.shape == velocities.shape == limits.angle_min.shape
    Post: penalty is 0 when all angles are within limits.
    Returns shape (n_dof,).
    """
    n = len(angles)
    assert angles.shape == (n,) and velocities.shape == (n,)
    assert limits.angle_min.shape == (n,)
    transition = 0.05  # rad (~3 degrees)
    result = np.zeros(n)
    for i in range(n):
        angle, vel = angles[i], velocities[i]
        lo, hi = limits.angle_min[i], limits.angle_max[i]
        if angle < lo:
            result[i] = _hermite_penalty(
                lo - angle, -vel, transition, limits.stiffness, limits.damping
            )
        elif angle > hi:
            result[i] = -_hermite_penalty(
                angle - hi, vel, transition, limits.stiffness, limits.damping
            )
    return result


# ---------------------------------------------------------------------------
# Forward kinematics (for visualization)
# ---------------------------------------------------------------------------


def forward_kinematics(theta1: float, phi: float, params: PendulumParams) -> dict:
    """Compute joint and tip positions in the world frame.

    Origin is at the shoulder pivot (fixed in world).
    x-axis points right, y-axis points up.

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 from downward vertical (rad).
    phi : float
        Relative angle of segment 2 (rad).
    params : PendulumParams

    Returns
    -------
    dict with 'hub', 'wrist', 'tip' as (x, y) tuples.
    """
    assert theta1 is not None, "theta1 must be provided"
    native_positions = _native_backend.double_forward_kinematics(theta1, phi, params)
    if native_positions is not None:
        return native_positions

    L1, L2 = params.L1, params.L2
    abs_angle2 = theta1 + phi

    # Segment 1 endpoint (wrist)
    wx = L1 * np.sin(theta1)
    wy = -L1 * np.cos(theta1)

    # Segment 2 endpoint (tip)
    tx = wx + L2 * np.sin(abs_angle2)
    ty = wy - L2 * np.cos(abs_angle2)

    return {
        "hub": (0.0, 0.0),
        "wrist": (wx, wy),
        "tip": (tx, ty),
    }


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------


def equations_of_motion(
    state: State,
    t: float,
    params: PendulumParams,
    torque_func: TorqueFunc,
    limits: JointLimits | None = None,
    clamp: TorqueClamp | None = None,
) -> np.ndarray:
    """Compute the state derivative: dx/dt = f(x, t).

    State vector x = [theta1, phi, dtheta1, dphi].

    M(q) * qddot = tau - C(q,qdot) - G(q)

    Parameters
    ----------
    state : np.ndarray, shape (4,)
    t : float
    params : PendulumParams
    torque_func : callable (t) -> (tau1, tau2)
    limits : JointLimits, optional
        Joint angle limits (stiffness/damping penalty).
    clamp : TorqueClamp, optional
        Torque saturation limits.

    Returns
    -------
    state_dot : np.ndarray, shape (4,)
    """
    assert state.shape == (4,)
    theta1, phi, dtheta1, dphi = state

    M = mass_matrix(phi, params)
    C = coriolis_vector(phi, dtheta1, dphi, params)
    G = gravity_vector(theta1, phi, params)

    tau_drive = np.array(torque_func(t))

    # Apply torque clamping if configured
    if clamp is not None:
        tau_drive = clamp_torque(tau_drive, clamp)

    # Dissipative and limit torques
    tau_friction = friction_torque_vector(dtheta1, dphi, params)
    tau_limit = (
        np.zeros(2)
        if limits is None
        else joint_limit_torque(phi, dphi, limits, theta1=theta1, dtheta1=dtheta1)
    )

    # Solve for accelerations: M * qddot = tau_total - C - G
    rhs = tau_drive + tau_friction + tau_limit - C - G
    qddot = np.linalg.solve(M, rhs)

    return np.array([dtheta1, dphi, qddot[0], qddot[1]])


def joint_velocities(state: State, params: PendulumParams) -> dict:
    """Compute linear velocities at each joint via Jacobian.

    Returns dict with 'wrist_speed', 'tip_speed', 'wrist_vel', 'tip_vel'.
    """
    assert state is not None, "state must be provided"
    theta1, phi, dtheta1, dphi = state
    abs_angle2 = theta1 + phi
    dabs2 = dtheta1 + dphi

    vwx = params.L1 * np.cos(theta1) * dtheta1
    vwy = params.L1 * np.sin(theta1) * dtheta1
    vtx = vwx + params.L2 * np.cos(abs_angle2) * dabs2
    vty = vwy + params.L2 * np.sin(abs_angle2) * dabs2

    return {
        "wrist_speed": float(np.sqrt(vwx**2 + vwy**2)),
        "tip_speed": float(np.sqrt(vtx**2 + vty**2)),
        "wrist_vel": (float(vwx), float(vwy)),
        "tip_vel": (float(vtx), float(vty)),
    }


# ---------------------------------------------------------------------------
# Base (shoulder) reaction force
# ---------------------------------------------------------------------------


def base_force(state: State, qddot: np.ndarray, params: PendulumParams) -> dict:
    """Compute reaction force at the shoulder pivot.

    Returns dict with 'fx', 'fy', 'magnitude'.
    """
    assert state is not None, "state must be provided"
    theta1, phi, dtheta1, dphi = state
    qdd1, qdd2 = qddot
    abs_angle2 = theta1 + phi
    dabs2 = dtheta1 + dphi
    ddabs2 = qdd1 + qdd2
    me = _m2eff(params)

    # Arm COM acceleration (at L1/2)
    ax1 = (params.L1 / 2) * (np.cos(theta1) * qdd1 - np.sin(theta1) * dtheta1**2)
    ay1 = (params.L1 / 2) * (np.sin(theta1) * qdd1 + np.cos(theta1) * dtheta1**2)

    # Wrist acceleration
    awx = params.L1 * (np.cos(theta1) * qdd1 - np.sin(theta1) * dtheta1**2)
    awy = params.L1 * (np.sin(theta1) * qdd1 + np.cos(theta1) * dtheta1**2)

    # Tip acceleration (clubhead)
    atx = awx + params.L2 * (
        np.cos(abs_angle2) * ddabs2 - np.sin(abs_angle2) * dabs2**2
    )
    aty = awy + params.L2 * (
        np.sin(abs_angle2) * ddabs2 + np.cos(abs_angle2) * dabs2**2
    )

    # Shaft COM at L2/2 from wrist
    asx = awx + (params.L2 / 2) * (
        np.cos(abs_angle2) * ddabs2 - np.sin(abs_angle2) * dabs2**2
    )
    asy = awy + (params.L2 / 2) * (
        np.sin(abs_angle2) * ddabs2 + np.cos(abs_angle2) * dabs2**2
    )

    fx = params.m1 * ax1 + params.m2 * asx + params.mClub * atx
    fy = (
        params.m1 * ay1
        + params.m2 * asy
        + params.mClub * aty
        - (params.m1 + me) * params.g
    )

    return {
        "fx": float(fx),
        "fy": float(fy),
        "magnitude": float(np.sqrt(fx**2 + fy**2)),
    }


# ---------------------------------------------------------------------------
# Zero-torque counterfactual
# ---------------------------------------------------------------------------


def ztcf_accelerations(
    state: State,
    params: PendulumParams,
    limits: JointLimits | None = None,
) -> np.ndarray:
    """Compute accelerations under zero driving torque."""
    assert state is not None, "state must be provided"
    theta1, phi, dtheta1, dphi = state
    M = mass_matrix(phi, params)
    C = coriolis_vector(phi, dtheta1, dphi, params)
    G = gravity_vector(theta1, phi, params)
    tau_f = friction_torque_vector(dtheta1, dphi, params)
    tau_lim = (
        np.zeros(2)
        if limits is None
        else joint_limit_torque(phi, dphi, limits, theta1=theta1, dtheta1=dtheta1)
    )
    rhs = tau_f + tau_lim - C - G
    return np.linalg.solve(M, rhs)


def control_vector(
    state: State,
    qddot_actual: np.ndarray,
    params: PendulumParams,
    limits: JointLimits | None = None,
) -> dict:
    """Control vector: difference between actual and ZTCF base forces.

    Returns dict with 'cvx', 'cvy', 'magnitude'.
    """
    assert state is not None, "state must be provided"
    qddot_ztcf = ztcf_accelerations(state, params, limits)
    f_actual = base_force(state, qddot_actual, params)
    f_ztcf = base_force(state, qddot_ztcf, params)
    cvx = f_actual["fx"] - f_ztcf["fx"]
    cvy = f_actual["fy"] - f_ztcf["fy"]
    return {"cvx": cvx, "cvy": cvy, "magnitude": np.sqrt(cvx**2 + cvy**2)}


# ---------------------------------------------------------------------------
# Linear accelerations and joint forces
# ---------------------------------------------------------------------------


def linear_accelerations(
    state: State, qddot: np.ndarray, params: PendulumParams
) -> dict:
    """Compute linear accelerations of joints in world coordinates."""
    assert state.shape == (4,) and qddot.shape == (2,)
    theta1, phi, dtheta1, dphi = state
    ddtheta1, ddphi = qddot
    L1, L2 = params.L1, params.L2
    abs_angle2 = theta1 + phi
    dabs2 = dtheta1 + dphi
    ddabs2 = ddtheta1 + ddphi

    ax_w = L1 * (-np.sin(theta1) * dtheta1**2 + np.cos(theta1) * ddtheta1)
    ay_w = L1 * (np.cos(theta1) * dtheta1**2 + np.sin(theta1) * ddtheta1)
    ax_t = ax_w + L2 * (-np.sin(abs_angle2) * dabs2**2 + np.cos(abs_angle2) * ddabs2)
    ay_t = ay_w + L2 * (np.cos(abs_angle2) * dabs2**2 + np.sin(abs_angle2) * ddabs2)

    return {"wrist": (ax_w, ay_w), "tip": (ax_t, ay_t)}


def net_joint_forces(state: State, qddot: np.ndarray, params: PendulumParams) -> dict:
    """Compute net joint forces (proximal on distal) in world coordinates."""
    assert state is not None, "state must be provided"
    acc = linear_accelerations(state, qddot, params)
    g_vec = np.array([0.0, -params.g])
    me = _m2eff(params)

    a_w = np.array(acc["wrist"])
    a_t = np.array(acc["tip"])
    f_wrist = me * a_t - me * g_vec
    f_shoulder = (params.m1 * a_w + me * a_t) - (params.m1 + me) * g_vec

    return {
        "shoulder": (float(f_shoulder[0]), float(f_shoulder[1])),
        "wrist": (float(f_wrist[0]), float(f_wrist[1])),
    }


# ---------------------------------------------------------------------------
# Energy calculations
# ---------------------------------------------------------------------------


def kinetic_energy(state: State, params: PendulumParams) -> float:
    """Compute total kinetic energy T = 0.5 * qdot^T M qdot."""
    assert state is not None, "state must be provided"
    from .physics_base import kinetic_energy_from_M

    _, phi, dtheta1, dphi = state
    M = mass_matrix(phi, params)
    qdot = np.array([dtheta1, dphi])
    return kinetic_energy_from_M(M, qdot)


def potential_energy(state: State, params: PendulumParams) -> float:
    """Compute gravitational potential energy (zero at shoulder height)."""
    assert state is not None, "state must be provided"
    theta1, phi = state[0], state[1]
    me = _m2eff(params)
    abs_angle2 = theta1 + phi
    V = -(params.m1 + me) * params.g * params.L1 * np.cos(
        theta1
    ) - me * params.g * params.L2 * np.cos(abs_angle2)
    return float(V)


def total_energy(state: State, params: PendulumParams) -> float:
    """Total mechanical energy E = T + V."""
    assert state is not None, "state must be provided"
    from .physics_base import total_energy_from_parts

    return total_energy_from_parts(
        kinetic_energy(state, params), potential_energy(state, params)
    )
