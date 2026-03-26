"""
Net force and equivalent couple of the two-hand club action.

Computes the resultant force system at a user-configurable action point
on the club from the two grip forces (right hand, left hand).

Supports three force decompositions:
    1. Overall forces — from full constrained dynamics
    2. ZTCF forces — from zero-torque counterfactual
    3. DELTA forces — from M-pseudoinverse (zero-velocity) decomposition

Design by Contract
------------------
- action_point alpha in [-1, +1]: -1 = right grip, 0 = midpoint, +1 = left grip
- All returned force/moment values are finite floats.
- net_force_on_club returns shape (2,) array.
- moment_of_net_force returns a scalar (2D cross product z-component).
- equivalent_couple returns a scalar.

DRY
---
Reuses golfer_kinematics.forward_kinematics for positions and
golfer_constraints.net_joint_forces / counterfactual_golfer for force data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .physics_golfer import N_DOF, GolferParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def club_action_point(
    q: np.ndarray,
    p: GolferParams,
    alpha: float = 0.0,
) -> np.ndarray:
    """Compute the action point on the club parameterised by alpha.

    Parameters
    ----------
    q : np.ndarray, shape (8,) or (16,)
        Generalized coordinates.
    p : GolferParams
    alpha : float in [-1, +1]
        -1 = right grip, 0 = midpoint between grips, +1 = left grip.

    Returns
    -------
    np.ndarray, shape (2,) — (x, y) position of the action point.

    Design by Contract
    ------------------
    Pre:  -1.0 <= alpha <= 1.0
    Post: returned array is shape (2,) and finite.
    """
    if not (-1.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [-1, 1], got {alpha}")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    from .golfer_kinematics import forward_kinematics

    fk = forward_kinematics(q, p)
    grip_r = np.array(fk["grip_right"])
    grip_l = np.array(fk["grip_left"])

    # Interpolate: t = (alpha + 1) / 2 maps [-1, +1] to [0, 1]
    t = (alpha + 1.0) / 2.0
    result = (1.0 - t) * grip_r + t * grip_l

    if not (result.shape == (2):
        raise ValueError(), f"Expected shape (2,), got {result.shape}")
    if not (np.all(np.isfinite(result))):
        raise ValueError(f"Action point is not finite: {result}")
    return result


# ---------------------------------------------------------------------------
# Force decomposition
# ---------------------------------------------------------------------------


def net_force_on_club(
    f_right: tuple[float, float] | np.ndarray,
    f_left: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """Compute the net (resultant) force of the two hands on the club.

    Parameters
    ----------
    f_right : (fx, fy) — force from the right hand
    f_left  : (fx, fy) — force from the left hand

    Returns
    -------
    np.ndarray, shape (2,) — net force vector

    Design by Contract
    ------------------
    Post: returned array is shape (2,) and finite.
    """
    net = np.array(f_right, dtype=float) + np.array(f_left, dtype=float)
    if not (net.shape == (2):
        raise ValueError())
    if not (np.all(np.isfinite(net))):
        raise ValueError(f"Net force is not finite: {net}")
    return net  # type: ignore[no-any-return]


def _cross_2d(r: np.ndarray, f: np.ndarray) -> float:
    """2D cross product (z-component): r × F = rx*Fy - ry*Fx."""
    return float(r[0] * f[1] - r[1] * f[0])


def moment_of_net_force(
    net_force: np.ndarray,
    force_point: np.ndarray,
    action_point: np.ndarray,
) -> float:
    """Compute the moment of the net force about the action point.

    In 2D, moment = r × F where r = force_point - action_point.

    Parameters
    ----------
    net_force : np.ndarray, shape (2,)
    force_point : np.ndarray, shape (2,) — where the net force acts
    action_point : np.ndarray, shape (2,) — point about which to compute moment

    Returns
    -------
    float — scalar moment (positive = counterclockwise)

    Design by Contract
    ------------------
    Pre:  all inputs are shape (2,) and finite
    Post: returned value is finite
    """
    r = force_point - action_point
    m = _cross_2d(r, net_force)
    if not (np.isfinite(m)):
        raise ValueError(f"Moment is not finite: {m}")
    return m


def equivalent_couple(
    f_right: np.ndarray,
    pos_right: np.ndarray,
    f_left: np.ndarray,
    pos_left: np.ndarray,
    action_point: np.ndarray,
) -> float:
    """Compute the equivalent couple to replace two-hand action with net force.

    The couple is the difference between the total moment produced by the
    two individual hand forces and the moment of the net force acting at
    the action point.

    couple = (M_right + M_left) - M_net

    where M_right = r_right × F_right, M_left = r_left × F_left, and
    M_net = moment of (F_right + F_left) acting at the action point itself
    (which is zero since the net force is placed there by definition).

    Actually: the net force is considered to act at the action point, so
    M_net = 0 (zero moment arm). The couple captures the remaining torque:

    couple = r_right × F_right + r_left × F_left

    where r_right = pos_right - action_point, r_left = pos_left - action_point.

    Parameters
    ----------
    f_right : np.ndarray, shape (2,)
    pos_right : np.ndarray, shape (2,) — right hand position
    f_left : np.ndarray, shape (2,)
    pos_left : np.ndarray, shape (2,) — left hand position
    action_point : np.ndarray, shape (2,)

    Returns
    -------
    float — scalar couple (positive = counterclockwise)

    Design by Contract
    ------------------
    Pre:  all inputs are shape (2,) and finite
    Post: returned value is finite
    """
    r_right = pos_right - action_point
    r_left = pos_left - action_point

    # Total moment from individual hand forces about the action point
    m_right = _cross_2d(r_right, f_right)
    m_left = _cross_2d(r_left, f_left)
    total_moment = m_right + m_left

    # The net force is considered to act at the action point,
    # so its moment about the action point is zero.
    # Couple = total moment - moment of net force at action point = total moment - 0
    couple = total_moment

    if not (np.isfinite(couple)):
        raise ValueError(f"Couple is not finite: {couple}")
    return couple


# ---------------------------------------------------------------------------
# High-level decomposition for all three force types
# ---------------------------------------------------------------------------


def club_force_decomposition(
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    p: GolferParams,
    forces: dict[str, tuple[float, float]],
    alpha: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """Decompose the two-hand club action into net force, moment, and couple.

    Parameters
    ----------
    q : np.ndarray, shape (8,) — generalized coordinates
    qdot : np.ndarray, shape (8,) — generalized velocities
    qddot : np.ndarray, shape (8,) — generalized accelerations
    p : GolferParams
    forces : dict — output of net_joint_forces() with keys 'rh' and 'lh'
    alpha : float — action point parameter [-1, +1]

    Returns
    -------
    dict with:
        'net_force': np.ndarray, shape (2,)
        'action_point': np.ndarray, shape (2,)
        'moment': float
        'couple': float
        'f_right': np.ndarray, shape (2,)
        'f_left': np.ndarray, shape (2,)
        'pos_right': np.ndarray, shape (2,)
        'pos_left': np.ndarray, shape (2,)

    Design by Contract
    ------------------
    Pre:  'rh' and 'lh' in forces dict
    Post: all values are finite
    """
    if not ("rh" in forces):
        raise ValueError("forces dict must contain 'rh' (right hand)")
    if not ("lh" in forces):
        raise ValueError("forces dict must contain 'lh' (left hand)")

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    from .golfer_kinematics import forward_kinematics

    fk = forward_kinematics(q, p)
    pos_right = np.array(fk["grip_right"])
    pos_left = np.array(fk["grip_left"])

    f_right = np.array(forces["rh"], dtype=float)
    f_left = np.array(forces["lh"], dtype=float)

    action_pt = club_action_point(q, p, alpha)
    net = net_force_on_club(f_right, f_left)

    # For the moment of the net force, we compute it about the action point.
    # The "force point" is the centroid of the two hand positions weighted by
    # force magnitudes — but since we're placing the net force at the action
    # point, the moment of net force about the action point is 0.
    # The couple captures the full residual.
    moment = 0.0  # Net force acts at the action point by definition

    couple_val = equivalent_couple(f_right, pos_right, f_left, pos_left, action_pt)

    return {
        "net_force": net,
        "action_point": action_pt,
        "moment": moment,
        "couple": couple_val,
        "f_right": f_right,
        "f_left": f_left,
        "pos_right": pos_right,
        "pos_left": pos_left,
    }


def overall_club_decomposition(
    state: np.ndarray,
    t: float,
    p: GolferParams,
    torque_func: Any,
    alpha: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """Club force decomposition using overall (full dynamics) forces.

    Uses the actual constrained dynamics to get accelerations and
    computes net joint forces from F = ma - mg.
    """
    if not (state is not None):
        raise ValueError("state must be provided")
    from .constraint_solver import constrained_accelerations
    from .physics_golfer import net_joint_forces

    q = state[:N_DOF]
    qdot = state[N_DOF:]
    qddot = constrained_accelerations(state, t, p, torque_func)
    forces = net_joint_forces(q, qdot, qddot, p)
    return club_force_decomposition(q, qdot, qddot, p, forces, alpha)


def ztcf_club_decomposition(
    state: np.ndarray,
    p: GolferParams,
    alpha: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """Club force decomposition using zero-torque counterfactual forces."""
    if not (state is not None):
        raise ValueError("state must be provided")
    from .counterfactual_golfer import zero_torque_accelerations
    from .physics_golfer import net_joint_forces

    q = state[:N_DOF]
    qdot = state[N_DOF:]
    qddot = zero_torque_accelerations(state, p)
    forces = net_joint_forces(q, qdot, qddot, p)
    return club_force_decomposition(q, qdot, qddot, p, forces, alpha)


def delta_club_decomposition(
    state: np.ndarray,
    tau: np.ndarray,
    p: GolferParams,
    alpha: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """Club force decomposition using DELTA (M-pseudoinverse) forces.

    DELTA accelerations = M^+ * tau at zero velocity.
    """
    if not (state is not None):
        raise ValueError("state must be provided")
    from .jacobians_golfer import delta_matrix
    from .physics_golfer import net_joint_forces

    q = state[:N_DOF]
    qdot = np.zeros(N_DOF)  # DELTA assumes zero velocity

    D = delta_matrix(q, p)
    qddot = D @ tau

    forces = net_joint_forces(q, qdot, qddot, p)
    return club_force_decomposition(q, qdot, qddot, p, forces, alpha)
