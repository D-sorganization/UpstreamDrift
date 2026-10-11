"""Static force and moment balance of a held club (issue #11739, OSV-7).

When the hands are held still the only load on the club besides the hands is
gravity, which acts at the centre of mass.  Equilibrium therefore requires,
for the per-hand wrenches ``(F_s, tau_s)`` at the grip points ``p_s``::

    sum_s F_s + m g = 0
    sum_s [ tau_s + (p_s - c) x F_s ] = 0          (moments about the centre of mass c)

This is the quasi-static check of the wrench extraction (signs, frames, lever
arms) for any grip model: holding the club in a sequence of attitudes is a
quasi-static rotation of the club, and both residuals must vanish at each.
The residuals are reported next to the scale of the gravity moment about the
hand midpoint, ``m |(c - P) x g|``, which is what the hands must supply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import GripKineticsSeries

__all__ = ["StaticBalance", "static_balance"]


@dataclass(frozen=True)
class StaticBalance:
    """Residuals of the static balance, one row per sample (world axes)."""

    force_residual_n: np.ndarray
    moment_residual_nm: np.ndarray
    gravity_moment_nm: np.ndarray  # m |(c - P) x g| about the hand midpoint
    weight_n: float

    def max_force_error(self) -> float:
        """Largest force residual as a fraction of the weight."""
        return float(
            np.linalg.norm(self.force_residual_n, axis=1).max() / self.weight_n
        )

    def max_moment_error(self) -> float:
        """Largest moment residual as a fraction of the gravity moment scale."""
        scale = np.maximum(self.gravity_moment_nm, 1e-12)
        return float((np.linalg.norm(self.moment_residual_nm, axis=1) / scale).max())


def static_balance(
    series: GripKineticsSeries,
    interface: GripInterface,
    club: ClubDynamics,
    gravity_m_s2: np.ndarray,
) -> StaticBalance:
    """Force and moment residuals of ``series`` taken as a sequence of holds.

    Preconditions: the hands were still at every sample and the club had
    settled (the caller checks this); ``gravity_m_s2`` is a 3-vector.

    Raises:
        ValueError: if ``gravity_m_s2`` is not a finite 3-vector.
    """
    g = np.asarray(gravity_m_s2, dtype=float)
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        raise ValueError("gravity_m_s2 must be a finite 3-vector")
    rot = series.club_rotation
    origin = series.grip_point_m["R"] - np.einsum(
        "nij,j->ni", rot, np.asarray(interface.right.position_m, dtype=float)
    )
    com = origin + np.einsum("nij,j->ni", rot, np.asarray(club.com_m, dtype=float))
    force = sum(series.force_on_club_n[s] for s in "LR")
    moment = sum(
        series.torque_on_club_nm[s]
        + np.cross(series.grip_point_m[s] - com, series.force_on_club_n[s])
        for s in "LR"
    )
    midpoint = 0.5 * (series.grip_point_m["L"] + series.grip_point_m["R"])
    gravity_moment = club.mass_kg * np.linalg.norm(np.cross(com - midpoint, g), axis=1)
    return StaticBalance(
        force_residual_n=force + club.mass_kg * g,
        moment_residual_nm=moment,
        gravity_moment_nm=gravity_moment,
        weight_n=club.mass_kg * float(np.linalg.norm(g)),
    )
