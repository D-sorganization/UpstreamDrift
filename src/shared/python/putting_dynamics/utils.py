"""Display-edge conversions and energy audit helpers (#8345).

Reused shared infrastructure (AGENTS.md section A discovery): ball
constants from ``src.shared.python.core.physics_constants``; DbC
helpers from ``src.shared.python.contracts``.

The package is SI-only internally; these converters exist so UI code
converts *at the display edge* (yards default for distances per fleet
direction), never inside the models.
"""

from __future__ import annotations

import math

from src.shared.python.contracts import ensure, require, require_finite
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)

from .types import CollisionReport, PutterState

__all__ = [
    "ball_kinetic_energy_j",
    "energy_balance_error_j",
    "m_to_feet",
    "m_to_yards",
    "mps_to_mph",
]

_FOOT_M = 0.3048
_YARD_M = 0.9144
_MPH_PER_MPS = 2.2369362920516


def m_to_yards(distance_m: float) -> float:
    """Meters to yards (fleet display default for distances)."""
    require_finite(distance_m, "distance_m")
    return distance_m / _YARD_M


def m_to_feet(distance_m: float) -> float:
    """Meters to feet (stimp/green-reading displays)."""
    require_finite(distance_m, "distance_m")
    return distance_m / _FOOT_M


def mps_to_mph(speed_mps: float) -> float:
    """Meters per second to miles per hour."""
    require_finite(speed_mps, "speed_mps")
    return speed_mps * _MPH_PER_MPS


def ball_kinetic_energy_j(speed_mps: float, spin_rad_s: float = 0.0) -> float:
    """Ball translational + rotational kinetic energy [J].

    Uses the USGA ball mass and the solid-sphere inertia
    ``(2/5) m r^2`` (matching
    ``GOLF_BALL_MOMENT_OF_INERTIA_KG_M2``).

    Args:
        speed_mps: Ball speed magnitude [m/s], >= 0.
        spin_rad_s: Spin magnitude [rad/s].

    Returns:
        Kinetic energy [J], >= 0.
    """
    require_finite(speed_mps, "speed_mps")
    require(speed_mps >= 0.0, "speed must be >= 0", speed_mps)
    require_finite(spin_rad_s, "spin_rad_s")
    inertia = 0.4 * GOLF_BALL_MASS_KG * GOLF_BALL_RADIUS_M**2
    energy = 0.5 * GOLF_BALL_MASS_KG * speed_mps**2 + 0.5 * inertia * spin_rad_s**2
    ensure(energy >= 0.0, "energy must be >= 0", energy)
    return energy


def energy_balance_error_j(report: CollisionReport, putter: PutterState) -> float:
    """Residual of the impact energy audit [J].

    Books the head's translational KE change (CG slowdown), the ball's
    launch KE (translation + spin), and the reported dissipation
    against the pre-impact head KE.  The residual is the energy parked
    in head rotation by off-center strikes plus the sub-percent
    cross-coupling of the split normal/tangential treatment; the
    contract test pins it to a small fraction of the incoming KE for
    centered strikes.

    Args:
        report: Impact report from :func:`~.collision.strike`.
        putter: The putter state that produced the report.

    Returns:
        Absolute energy residual [J].
    """
    ke_head_before = 0.5 * putter.head_mass_kg * putter.speed_mps**2
    ke_head_after = (
        0.5 * putter.head_mass_kg * ((putter.speed_mps - report.putter_dv_mps) ** 2)
    )
    ke_ball = ball_kinetic_energy_j(report.ball_speed_mps, report.spin_rad_s)
    residual = ke_head_before - (ke_head_after + ke_ball + report.kinetic_energy_loss_j)
    return math.fabs(residual)
