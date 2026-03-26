"""Forward kinematics for golfer upper-body model.

Topology (from the sketch)
--------------------------
Fixed hub connects via a standoff to two shoulder joints that branch
into independent arm chains.  Both wrist endpoints attach to different
points on a shared club segment, closing the kinematic loop.

    Hub ─── RS (right shoulder)
     │           │
     │          RE (right elbow)
     │           │
     │          RH (right hand / wrist)──┐
     │                                   Club ── Clubhead
     │          LH (left hand / wrist)───┘
     │           │
     │          LE (left elbow)
     │           │
     └── LS (left shoulder)

Coordinate convention:
    Angles measured from downward vertical, positive counterclockwise.
    World frame: x→right, y→up, origin at hub.
"""

from __future__ import annotations

import numpy as np

from . import native_backend as _native_backend
from .physics_golfer import N_DOF, GolferParams


def _hub_position(theta_hub: float, p: GolferParams) -> tuple[float, float]:
    """Hub endpoint position (end of standoff from fixed origin).

    Reversed direction (#1103): hub extends upward (inside the arm loop)
    to simulate rotation around the combined center of mass.
    """
    if not (theta_hub is not None):
        raise ValueError("theta_hub must be provided")
    x = -p.L_hub * np.sin(theta_hub)
    y = p.L_hub * np.cos(theta_hub)
    return (x, y)


def _shoulder_position(
    hub_xy: tuple[float, float],
    theta_hub: float,
    d_shoulder: float,
    side: float,
) -> tuple[float, float]:
    """Shoulder joint position.

    The shoulder bar is perpendicular to the hub standoff.
    side: +1 for right, -1 for left (perpendicular offset direction).
    """
    # Perpendicular direction to hub standoff (rotated 90° from hub direction)
    if not (hub_xy is not None):
        raise ValueError("hub_xy must be provided")
    perp_x = side * np.cos(theta_hub)
    perp_y = side * np.sin(theta_hub)
    x = hub_xy[0] + d_shoulder * perp_x
    y = hub_xy[1] + d_shoulder * perp_y
    return (x, y)


def _chain_endpoint(
    origin: tuple[float, float],
    angles_abs: list[float],
    lengths: list[float],
) -> tuple[float, float]:
    """Compute endpoint of a serial chain from origin.

    Parameters
    ----------
    origin : (x, y) start position
    angles_abs : list of absolute angles for each segment
    lengths : list of segment lengths

    Returns
    -------
    (x, y) endpoint position
    """
    if not (origin is not None):
        raise ValueError("origin must be provided")
    x, y = origin
    for angle, length in zip(angles_abs, lengths, strict=False):
        x += length * np.sin(angle)
        y -= length * np.cos(angle)
    return (x, y)


def _absolute_angles(theta_hub: float, relative_angles: list[float]) -> list[float]:
    """Convert relative joint angles to absolute angles for a chain.

    Each relative angle is added cumulatively to the hub angle.
    """
    if not (theta_hub is not None):
        raise ValueError("theta_hub must be provided")
    result = []
    cumulative = theta_hub
    for rel in relative_angles:
        cumulative += rel
        result.append(cumulative)
    return result


def forward_kinematics(
    q: np.ndarray, p: GolferParams
) -> dict[str, tuple[float, float]]:
    """Compute all joint positions in world frame.

    Parameters
    ----------
    q : np.ndarray, shape (8,) or (16,) — generalized coordinates
        [theta_hub, alpha_rs, alpha_re, alpha_rh,
         alpha_ls, alpha_le, alpha_lh, theta_club]

    Returns
    -------
    dict with keys: 'hub', 'rs', 're', 'rh', 'ls', 'le', 'lh',
                    'club_base', 'club_tip', 'grip_right', 'grip_left'
    """
    native_positions = _native_backend.golfer_forward_kinematics(q, p)
    if native_positions is not None:
        return native_positions

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]
    if not (q.shape == (N_DOF):
        raise ValueError(), f"q must have shape ({N_DOF},), got {q.shape}")

    th_hub = q[0]
    alpha_rs, alpha_re, alpha_rh = q[1], q[2], q[3]
    alpha_ls, alpha_le, alpha_lh = q[4], q[5], q[6]
    th_club = q[7]

    hub = _hub_position(th_hub, p)

    # Scapula and shoulder positions (#1104)
    # The hub bar endpoint is where the shoulder (or scapula) connects.
    rs_bar = _shoulder_position(hub, th_hub, p.d_rs, +1.0)
    ls_bar = _shoulder_position(hub, th_hub, p.d_ls, -1.0)

    # If scapula link is present, the shoulder is offset downward from the bar
    if p.L_rscap > 0:
        rscap = rs_bar  # scapula joint sits at bar endpoint
        # Shoulder extends from scapula along hub angle (downward)
        rs = (
            rscap[0] + p.L_rscap * np.sin(th_hub),
            rscap[1] - p.L_rscap * np.cos(th_hub),
        )
    else:
        rscap = rs_bar
        rs = rs_bar

    if p.L_lscap > 0:
        lscap = ls_bar
        ls = (
            lscap[0] + p.L_lscap * np.sin(th_hub),
            lscap[1] - p.L_lscap * np.cos(th_hub),
        )
    else:
        lscap = ls_bar
        ls = ls_bar

    # Right arm chain: RS → RE → RH
    r_abs = _absolute_angles(th_hub, [alpha_rs, alpha_re, alpha_rh])
    re = _chain_endpoint(rs, [r_abs[0]], [p.L_r_upper])
    rh = _chain_endpoint(rs, r_abs[:2], [p.L_r_upper, p.L_r_fore])

    # Left arm chain: LS → LE → LH
    l_abs = _absolute_angles(th_hub, [alpha_ls, alpha_le, alpha_lh])
    le = _chain_endpoint(ls, [l_abs[0]], [p.L_l_upper])
    lh = _chain_endpoint(ls, l_abs[:2], [p.L_l_upper, p.L_l_fore])

    # Club direction unit vector
    club_dx = np.sin(th_club)
    club_dy = -np.cos(th_club)

    # Club base defined from right-hand grip position along club direction
    club_base = (
        rh[0] - p.grip_right * club_dx,
        rh[1] + p.grip_right * club_dy,
    )
    grip_l_on_club = (
        club_base[0] + p.grip_left * club_dx,
        club_base[1] - p.grip_left * club_dy,
    )
    club_tip = (
        club_base[0] + p.L_club * club_dx,
        club_base[1] - p.L_club * club_dy,
    )

    result = {
        "origin": (0.0, 0.0),
        "hub": hub,
        "rs": rs,
        "re": re,
        "rh": rh,
        "ls": ls,
        "le": le,
        "lh": lh,
        "club_base": club_base,
        "club_tip": club_tip,
        "grip_right": rh,
        "grip_left": grip_l_on_club,
    }

    # Add scapula positions if present (#1104)
    if p.L_rscap > 0:
        result["rscap"] = rscap
    if p.L_lscap > 0:
        result["lscap"] = lscap

    return result
