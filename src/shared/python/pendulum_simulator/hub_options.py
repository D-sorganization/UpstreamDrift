"""
Hub standoff options: massless hub, COM-tracking, and manual offset.

Provides utilities for modifying the hub standoff parameters in the
golfer model to support massless hub, rotation centre at system COM,
and manual rotation centre adjustment.

Design by Contract
------------------
- effective_hub_mass always returns a positive value.
- compute_system_com returns shape (2,) with finite values.
- make_massless_hub_params returns a valid GolferParams.

DRY
---
Reuses forward_kinematics for position computation.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from .physics_golfer import N_DOF, GolferParams

logger = logging.getLogger(__name__)

# Minimum hub mass to maintain numerical stability (kg)
_HUB_MASS_EPSILON = 1e-6


def effective_hub_mass(m_hub: float, massless: bool = False) -> float:
    """Return the effective hub mass.

    Parameters
    ----------
    m_hub : float — nominal hub mass
    massless : bool — if True, return epsilon mass

    Returns
    -------
    float — always positive

    Design by Contract
    ------------------
    Post: result > 0
    """
    assert m_hub is not None, "m_hub must be provided"
    if massless:
        return _HUB_MASS_EPSILON
    return m_hub


def make_massless_hub_params(p: GolferParams) -> GolferParams:
    """Create a copy of GolferParams with effectively massless hub.

    Returns a new GolferParams where m_hub is set to epsilon (1e-6 kg),
    making the hub standoff effectively massless in the dynamics.

    Parameters
    ----------
    p : GolferParams — original parameters

    Returns
    -------
    GolferParams with m_hub ~ 0
    """
    fields = dataclasses.asdict(p)
    fields["m_hub"] = _HUB_MASS_EPSILON
    return GolferParams(**fields)


def compute_system_com(
    q: np.ndarray,
    p: GolferParams,
) -> np.ndarray:
    """Compute the centre of mass of the entire golfer system.

    Uses forward kinematics to get joint positions, then computes
    the mass-weighted average position. Each segment's mass is
    assumed to act at the midpoint of the segment.

    Parameters
    ----------
    q : np.ndarray, shape (8,) or (16,)
    p : GolferParams

    Returns
    -------
    np.ndarray, shape (2,) — (x, y) position of system COM

    Design by Contract
    ------------------
    Post: result is shape (2,) and finite
    """
    from .golfer_kinematics import forward_kinematics

    if q.shape[0] > N_DOF:
        q = q[:N_DOF]

    fk = forward_kinematics(q, p)

    # Build list of (mass, position) pairs for COM calculation
    # Segment midpoints (mass acts at midpoint of each segment)
    origin = np.array([0.0, 0.0])
    hub = np.array(fk["hub"])
    rs = np.array(fk["rs"])
    re = np.array(fk["re"])
    rh = np.array(fk["grip_right"])
    ls = np.array(fk["ls"])
    le = np.array(fk["le"])
    lh = np.array(fk["grip_left"])
    club_tip = np.array(fk["club_tip"])
    club_base = np.array(fk["club_base"])

    masses_positions = [
        (p.m_hub, 0.5 * (origin + hub)),  # hub standoff midpoint
        (p.m_r_upper, 0.5 * (rs + re)),  # right upper arm midpoint
        (p.m_r_fore, 0.5 * (re + rh)),  # right forearm midpoint
        (p.m_l_upper, 0.5 * (ls + le)),  # left upper arm midpoint
        (p.m_l_fore, 0.5 * (le + lh)),  # left forearm midpoint
        (p.m_club, 0.5 * (club_base + club_tip)),  # club midpoint
        (p.m_clubhead, np.array(fk["club_tip"])),  # clubhead point mass at tip
    ]

    total_mass = sum(m for m, _ in masses_positions)
    assert total_mass > 0, "Total system mass must be positive"

    com = sum(m * pos for m, pos in masses_positions) / total_mass

    assert com.shape == (2,), f"Expected shape (2,), got {com.shape}"
    assert np.all(np.isfinite(com)), f"COM is not finite: {com}"
    return com  # type: ignore[no-any-return]


def hub_offset_for_com(
    q: np.ndarray,
    p: GolferParams,
) -> tuple[float, float]:
    """Compute the hub offset needed to place the rotation centre at the system COM.

    Returns the (dx, dy) that should be added to the hub origin so that
    the rotation centre coincides with the system centre of mass.

    Parameters
    ----------
    q : np.ndarray, shape (8,) or (16,)
    p : GolferParams

    Returns
    -------
    tuple (dx, dy)
    """
    assert q is not None, "q must be provided"
    com = compute_system_com(q, p)
    # The hub origin is at (0, 0) in world frame
    return (float(com[0]), float(com[1]))
