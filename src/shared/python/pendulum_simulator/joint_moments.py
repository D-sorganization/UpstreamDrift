"""
Joint moment and torque vector calculations for pendulum models.

Computes three quantities at each joint (proximal-on-distal convention):
    1. Applied joint torque (the motor/muscle torque)
    2. Moment of net force (cross product of position Ã— force)
    3. Total moment (applied torque + moment of net force)

All functions are pure, stateless, and model-agnostic where possible.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2-D cross-product helper (scalar result for planar mechanics)
# ---------------------------------------------------------------------------


def cross_2d(r: np.ndarray, f: np.ndarray) -> float:
    """Compute 2-D cross product r Ã— f (scalar, positive = CCW).

    Preconditions:
        r, f are shape (2,) and finite.
    Postconditions:
        Returns a finite float.
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(f, dtype=float)
    assert r.shape == (2,), f"r must be shape (2,), got {r.shape}"
    assert f.shape == (2,), f"f must be shape (2,), got {f.shape}"
    assert np.all(np.isfinite(r)), f"r must be finite: {r}"
    assert np.all(np.isfinite(f)), f"f must be finite: {f}"
    result = float(r[0] * f[1] - r[1] * f[0])
    assert np.isfinite(result), f"cross product is non-finite: {result}"
    return result


# ---------------------------------------------------------------------------
# Per-joint moment calculation
# ---------------------------------------------------------------------------


def moment_of_force(
    joint_position: np.ndarray,
    distal_com_position: np.ndarray,
    net_force: np.ndarray,
) -> float:
    """Moment of the net joint force about the distal segment's COM.

    M = r Ã— F where r = distal_com - joint_position.

    Preconditions:
        All arrays shape (2,), finite.
    Postconditions:
        Returns finite float (NÂ·m, positive = CCW).
    """
    assert joint_position is not None, "joint_position must be provided"
    r = np.asarray(distal_com_position, dtype=float) - np.asarray(
        joint_position, dtype=float
    )
    return cross_2d(r, np.asarray(net_force, dtype=float))


def total_moment_at_joint(
    applied_torque: float,
    joint_position: np.ndarray,
    distal_com_position: np.ndarray,
    net_force: np.ndarray,
) -> float:
    """Total moment = applied joint torque + moment of net force.

    Preconditions:
        applied_torque finite; arrays shape (2,), finite.
    Postconditions:
        Returns finite float.
    """
    assert np.isfinite(applied_torque), f"torque must be finite, got {applied_torque}"
    m_force = moment_of_force(joint_position, distal_com_position, net_force)
    result = applied_torque + m_force
    assert np.isfinite(result), f"total moment is non-finite: {result}"
    return result


# ---------------------------------------------------------------------------
# Double pendulum joint moments
# ---------------------------------------------------------------------------


def double_pendulum_moments(
    positions: dict,
    joint_forces: dict,
    applied_torques: tuple[float, float],
    params: object,
) -> dict:
    """Compute all joint moments for the double pendulum.

    Parameters
    ----------
    positions : dict
        Output of forward_kinematics: {shoulder, wrist, tip}.
    joint_forces : dict
        Output of net_joint_forces: {shoulder: (fx,fy), wrist: (fx,fy)}.
    applied_torques : tuple
        (tau_shoulder, tau_wrist) from torque function.
    params : PendulumParams
        Physical parameters.

    Returns
    -------
    dict with keys per joint:
        shoulder_applied_torque, shoulder_moment_of_force, shoulder_total_moment,
        wrist_applied_torque, wrist_moment_of_force, wrist_total_moment.
    """
    assert positions is not None, "positions must be provided"
    shoulder = np.array(positions["shoulder"])
    wrist = np.array(positions["wrist"])
    tip = np.array(positions["tip"])

    # COM positions (midpoints for uniform segments)
    arm_com = (shoulder + wrist) / 2.0
    shaft_com = (wrist + tip) / 2.0

    f_shoulder = np.array(joint_forces["shoulder"])
    f_wrist = np.array(joint_forces["wrist"])

    # Shoulder: moment about arm COM
    m_shoulder = moment_of_force(shoulder, arm_com, f_shoulder)
    total_shoulder = total_moment_at_joint(
        applied_torques[0], shoulder, arm_com, f_shoulder
    )

    # Wrist: moment about shaft COM
    m_wrist = moment_of_force(wrist, shaft_com, f_wrist)
    total_wrist = total_moment_at_joint(applied_torques[1], wrist, shaft_com, f_wrist)

    return {
        "shoulder_applied_torque": applied_torques[0],
        "shoulder_moment_of_force": m_shoulder,
        "shoulder_total_moment": total_shoulder,
        "wrist_applied_torque": applied_torques[1],
        "wrist_moment_of_force": m_wrist,
        "wrist_total_moment": total_wrist,
    }


# ---------------------------------------------------------------------------
# Triple pendulum joint moments
# ---------------------------------------------------------------------------


def triple_pendulum_moments(
    positions: dict,
    joint_forces: dict,
    applied_torques: tuple[float, float, float],
    params: object,
) -> dict:
    """Compute all joint moments for the triple pendulum.

    Parameters
    ----------
    positions : dict
        {shoulder, elbow, wrist, tip}.
    joint_forces : dict
        {shoulder: (fx,fy), elbow: (fx,fy), wrist: (fx,fy)}.
    applied_torques : tuple
        (tau_shoulder, tau_elbow, tau_wrist).

    Returns
    -------
    dict with applied_torque, moment_of_force, total_moment for each joint.
    """
    assert positions is not None, "positions must be provided"
    joints = ["shoulder", "elbow", "wrist"]
    endpoints = ["elbow", "wrist", "tip"]

    result = {}
    for i, (jname, ename) in enumerate(zip(joints, endpoints, strict=False)):
        j_pos = np.array(positions[jname])
        e_pos = np.array(positions[ename])
        com = (j_pos + e_pos) / 2.0
        f_joint = np.array(joint_forces[jname])

        m_force = moment_of_force(j_pos, com, f_joint)
        total = total_moment_at_joint(applied_torques[i], j_pos, com, f_joint)

        result[f"{jname}_applied_torque"] = applied_torques[i]
        result[f"{jname}_moment_of_force"] = m_force
        result[f"{jname}_total_moment"] = total

    return result


# ---------------------------------------------------------------------------
# Golfer (7-DOF) joint moments
# ---------------------------------------------------------------------------


def golfer_pendulum_moments(
    positions: dict,
    joint_forces: dict,
    applied_torques: tuple[float, ...],
    params: object,
) -> dict:
    """Compute all joint moments for the golfer upper-body model.

    Parameters
    ----------
    positions : dict
        Output of forward_kinematics: {hub, rs, re, rh, ls, le, lh, club_tip, ...}.
    joint_forces : dict
        Output of net_joint_forces: {hub, rs, re, rh, ls, le, lh, ...} each (fx,fy).
    applied_torques : tuple
        (tau_hub, tau_rs, tau_re, tau_rh, tau_ls, tau_le, tau_lh) from torque function.
    params : GolferParams
        Physical parameters (used for segment topology).

    Returns
    -------
    dict with applied_torque, moment_of_force, total_moment for each of the
    7 actuated joints.

    Contract
    --------
    Pre:  len(applied_torques) >= 7 and all required keys present in positions/forces.
    Post: Returns dict with 21 keys (3 per joint Ã— 7 joints), all finite.
    """
    assert (
        len(applied_torques) >= 7
    ), f"Need >= 7 applied torques, got {len(applied_torques)}"
    # Joint â†’ distal endpoint pairs (joint connects to next link's endpoint)
    joints = ["hub", "rs", "re", "rh", "ls", "le", "lh"]
    endpoints = ["rs", "re", "rh", "club_tip", "le", "lh", "club_tip"]

    result = {}
    for i, (jname, ename) in enumerate(zip(joints, endpoints, strict=False)):
        j_pos_raw = positions.get(jname)
        e_pos_raw = positions.get(ename)
        f_raw = joint_forces.get(jname)

        if j_pos_raw is None or e_pos_raw is None or f_raw is None:
            # Missing data â€” store zeros
            result[f"{jname}_applied_torque"] = applied_torques[i]
            result[f"{jname}_moment_of_force"] = 0.0
            result[f"{jname}_total_moment"] = applied_torques[i]
            continue

        j_pos = np.array(j_pos_raw, dtype=float)
        e_pos = np.array(e_pos_raw, dtype=float)
        com = (j_pos + e_pos) / 2.0
        f_joint = np.array(f_raw, dtype=float)

        m_force = moment_of_force(j_pos, com, f_joint)
        total = total_moment_at_joint(applied_torques[i], j_pos, com, f_joint)

        result[f"{jname}_applied_torque"] = applied_torques[i]
        result[f"{jname}_moment_of_force"] = m_force
        result[f"{jname}_total_moment"] = total

    return result


# ---------------------------------------------------------------------------
# Torque vector direction for 2-D rendering
# ---------------------------------------------------------------------------


def torque_arrow_direction(
    joint_position: np.ndarray,
    segment_angle: float,
    torque_value: float,
    arrow_length: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute start/end points for a curved torque arrow.

    The arrow is tangent to a circle around the joint, with direction
    determined by the sign of the torque (positive = CCW).

    Parameters
    ----------
    joint_position : ndarray, shape (2,)
    segment_angle : float
        Absolute angle of the distal segment (rad).
    torque_value : float
        Signed torque value (positive = CCW).
    arrow_length : float
        Visual length of the arrow.

    Returns
    -------
    (start, end) : tuple of ndarray, shape (2,)
        Start and end points in world coordinates.
    """
    joint_position = np.asarray(joint_position, dtype=float)
    assert joint_position.shape == (2,)
    assert np.isfinite(torque_value)

    if abs(torque_value) < 1e-10:
        return joint_position.copy(), joint_position.copy()

    sign = np.sign(torque_value)
    # Arrow starts perpendicular to segment, arcs in torque direction
    perp_angle = segment_angle + sign * np.pi / 4
    radius = arrow_length * 0.5

    start = joint_position + radius * np.array(
        [np.cos(perp_angle - sign * 0.3), np.sin(perp_angle - sign * 0.3)]
    )
    end = joint_position + radius * np.array(
        [np.cos(perp_angle + sign * 0.3), np.sin(perp_angle + sign * 0.3)]
    )

    return start, end
