"""Divot model for iron shots, including fat shots (#15523).

The sole path near impact is approximated by a parabola about its low point::

    z(x) = z_low + (x - x_low)**2 / (2 R)

``x`` is measured from the ball toward the target, ``z`` from the turf surface
(negative = below turf) and ``R`` is the arc radius of curvature. Where
``z(x) < 0`` the sole is cutting turf; that span is the divot.

* ``x_low > 0`` and the divot starts after the ball: a clean, ball-first
  strike with the divot ahead of the ball. Nothing is lost before impact.
* The divot starts before the ball (entry ``x < 0``): a **fat shot**. The
  turf work done between entry and the ball is taken from the clubhead's
  kinetic energy, and the grass/dirt wedged at the face costs spin.

Provenance: the shear strength scale and the spin exponent are model
heuristics (like ``GRASS_RESISTANCE_COEFFICIENT`` in ``_terrain_physics``),
pinned by behavioural tests, not measured constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import ensure, require
from src.shared.python.physics.terrain.materials import SurfaceMaterial

#: Turf shear strength [Pa] at ``hardness == 1``; scales linearly with
#: ``SurfaceMaterial.hardness`` (heuristic).
TURF_SHEAR_STRENGTH_PA_AT_FULL_HARDNESS = 2.0e5
#: Effective clubhead mass that carries the strike [kg] (head plus the share
#: of shaft/hands mass felt at impact).
DEFAULT_EFFECTIVE_MASS_KG = 0.30
#: Typical iron divot width [m].
DEFAULT_DIVOT_WIDTH_M = 0.07
#: Radius of curvature of the sole path near the low point [m] (~ arm + club).
DEFAULT_ARC_RADIUS_M = 1.5
#: Spin retention = speed_retention ** exponent: turf trapped at the face hurts
#: spin more than speed (heuristic).
SPIN_RETENTION_EXPONENT = 2.0
#: Integration resolution along the divot.
_INTEGRATION_POINTS = 401


@dataclass(frozen=True)
class SwingArc:
    """Sole path low point relative to the ball and the turf surface.

    Attributes:
        low_point_offset_m: Along-target position of the arc bottom relative
            to the ball (positive = ahead of the ball, negative = behind).
        low_point_height_m: Sole height at the arc bottom relative to the turf
            surface (negative = below the surface, i.e. digging).
    """

    low_point_offset_m: float
    low_point_height_m: float


@dataclass(frozen=True)
class DivotResult:
    """Divot geometry and the fat-shot penalty applied to the strike."""

    has_divot: bool
    is_fat_shot: bool
    entry_offset_m: float
    exit_offset_m: float
    length_m: float
    max_depth_m: float
    volume_m3: float
    pre_ball_length_m: float
    energy_loss_j: float
    speed_retention: float
    spin_retention: float


def _no_divot() -> DivotResult:
    return DivotResult(
        has_divot=False,
        is_fat_shot=False,
        entry_offset_m=math.nan,
        exit_offset_m=math.nan,
        length_m=0.0,
        max_depth_m=0.0,
        volume_m3=0.0,
        pre_ball_length_m=0.0,
        energy_loss_j=0.0,
        speed_retention=1.0,
        spin_retention=1.0,
    )


def compute_divot(
    arc: SwingArc,
    clubhead_speed_mps: float,
    material: SurfaceMaterial,
    *,
    arc_radius_m: float = DEFAULT_ARC_RADIUS_M,
    divot_width_m: float = DEFAULT_DIVOT_WIDTH_M,
    effective_mass_kg: float = DEFAULT_EFFECTIVE_MASS_KG,
) -> DivotResult:
    """Compute the divot and, for a fat shot, the pre-ball speed/spin loss.

    Args:
        arc: Sole path low point (offset from ball, height above turf).
        clubhead_speed_mps: Clubhead speed arriving at the divot entry [m/s].
        material: Turf the club strikes; ``hardness`` sets resistance.
        arc_radius_m: Radius of curvature of the sole path [m].
        divot_width_m: Width of the cut [m].
        effective_mass_kg: Mass whose kinetic energy the turf work depletes.

    Returns:
        :class:`DivotResult`. With no ground contact the result carries no
        divot and unit retentions.
    """
    require(
        math.isfinite(clubhead_speed_mps) and clubhead_speed_mps > 0.0,
        "clubhead_speed_mps must be finite and positive",
        clubhead_speed_mps,
    )
    require(
        math.isfinite(arc_radius_m) and arc_radius_m > 0.0,
        "arc_radius_m must be finite and positive",
        arc_radius_m,
    )
    require(
        math.isfinite(divot_width_m) and divot_width_m > 0.0,
        "divot_width_m must be finite and positive",
        divot_width_m,
    )
    require(
        math.isfinite(effective_mass_kg) and effective_mass_kg > 0.0,
        "effective_mass_kg must be finite and positive",
        effective_mass_kg,
    )
    require(
        math.isfinite(arc.low_point_offset_m) and math.isfinite(arc.low_point_height_m),
        "swing arc must be finite",
    )

    z_low = arc.low_point_height_m
    if z_low >= 0.0:
        return _no_divot()

    x_low = arc.low_point_offset_m
    half_len = math.sqrt(-2.0 * arc_radius_m * z_low)
    entry, exit_ = x_low - half_len, x_low + half_len
    max_depth = -z_low
    # Parabolic section area = (2/3) * length * depth; prismatic in width.
    volume = (2.0 / 3.0) * (2.0 * half_len) * max_depth * divot_width_m

    is_fat = entry < 0.0
    pre_ball_length = -entry if is_fat else 0.0
    energy_loss = 0.0
    if is_fat:
        x = np.linspace(entry, min(0.0, exit_), _INTEGRATION_POINTS)
        depth = np.clip(-(z_low + (x - x_low) ** 2 / (2.0 * arc_radius_m)), 0.0, None)
        shear_pa = TURF_SHEAR_STRENGTH_PA_AT_FULL_HARDNESS * material.hardness
        energy_loss = float(shear_pa * divot_width_m * np.trapezoid(depth, x))

    kinetic = 0.5 * effective_mass_kg * clubhead_speed_mps**2
    speed_retention = math.sqrt(max(0.0, 1.0 - energy_loss / kinetic))
    spin_retention = speed_retention**SPIN_RETENTION_EXPONENT

    ensure(
        0.0 <= spin_retention <= speed_retention <= 1.0, "retentions must be ordered"
    )
    return DivotResult(
        has_divot=True,
        is_fat_shot=is_fat,
        entry_offset_m=entry,
        exit_offset_m=exit_,
        length_m=2.0 * half_len,
        max_depth_m=max_depth,
        volume_m3=volume,
        pre_ball_length_m=pre_ball_length,
        energy_loss_j=energy_loss,
        speed_retention=speed_retention,
        spin_retention=spin_retention,
    )
