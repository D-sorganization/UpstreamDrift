"""Penetrometer firmness scale for bunker sand (issue #8610).

USGA / Turf & Soil Diagnostics rate a bunker sand's tendency to bury a ball by
pushing a penetrometer with a golf ball on the tip into oven-dried, loosened
sand until the ball is buried to its hemisphere, and reading the force in
kg/cm^2:

==================  ==========  ===============
tendency to bury    kg/cm^2     rating
==================  ==========  ===============
high                < 1.8       undesirable
moderate            1.8 - 2.2   acceptable
slight              2.2 - 2.4   acceptable
very low            > 2.4       **desirable**
==================  ==========  ===============

The USGA's own caveat is that the absolute number means little but the
*relative* comparison is useful -- which is exactly what a design sweep needs.
The four sweep points 1.6 / 2.0 / 2.4 / 2.8 span undesirable to desirable.

Values are stored internally in pascals; the kg/cm^2 form is the published
unit and is converted at the boundary.
"""

from __future__ import annotations

import math
from enum import StrEnum

from .exceptions import SandModelError

__all__ = [
    "FIRMNESS_DENSE_ANCHOR_KG_PER_CM2",
    "FIRMNESS_LOOSE_ANCHOR_KG_PER_CM2",
    "FIRMNESS_SWEEP_KG_PER_CM2",
    "KG_PER_CM2_IN_PASCAL",
    "FirmnessRating",
    "firmness_kg_per_cm2_from_pa",
    "firmness_pa_from_kg_per_cm2",
    "firmness_rating",
    "relative_density_from_firmness",
]

STANDARD_GRAVITY_M_S2 = 9.80665
KG_PER_CM2_IN_PASCAL = STANDARD_GRAVITY_M_S2 / 1.0e-4
"""One kg-force per square centimetre in pascals (98066.5 Pa)."""

FIRMNESS_SWEEP_KG_PER_CM2: tuple[float, float, float, float] = (1.6, 2.0, 2.4, 2.8)
"""Sweep points spanning the published rating scale."""

FIRMNESS_LOOSE_ANCHOR_KG_PER_CM2 = 1.4
"""Firmness mapped to relative density 0. A modelling convention, not data."""

FIRMNESS_DENSE_ANCHOR_KG_PER_CM2 = 3.0
"""Firmness mapped to relative density 1. A modelling convention, not data."""

_UNDESIRABLE_CEILING_KG_PER_CM2 = 1.8
_ACCEPTABLE_CEILING_KG_PER_CM2 = 2.4


class FirmnessRating(StrEnum):
    """USGA rating band for a penetrometer reading."""

    UNDESIRABLE = "undesirable"
    ACCEPTABLE = "acceptable"
    DESIRABLE = "desirable"


def _require_positive_firmness(value: float, unit: str) -> float:
    if not math.isfinite(value):
        raise SandModelError(f"penetrometer firmness must be finite, got {value!r}")
    if value <= 0.0:
        raise SandModelError(
            f"penetrometer firmness must be positive, got {value!r} {unit}"
        )
    return float(value)


def firmness_pa_from_kg_per_cm2(firmness_kg_per_cm2: float) -> float:
    """Convert a published penetrometer reading to pascals.

    Raises:
        SandModelError: if the reading is not a positive finite number.
    """
    value = _require_positive_firmness(firmness_kg_per_cm2, "kg/cm^2")
    return value * KG_PER_CM2_IN_PASCAL


def firmness_kg_per_cm2_from_pa(firmness_pa: float) -> float:
    """Convert an internal pascal value back to the published unit.

    Raises:
        SandModelError: if the value is not a positive finite number.
    """
    value = _require_positive_firmness(firmness_pa, "Pa")
    return value / KG_PER_CM2_IN_PASCAL


def firmness_rating(firmness_kg_per_cm2: float) -> FirmnessRating:
    """Return the USGA rating band for a penetrometer reading.

    Raises:
        SandModelError: if the reading is not a positive finite number.
    """
    value = _require_positive_firmness(firmness_kg_per_cm2, "kg/cm^2")
    if value < _UNDESIRABLE_CEILING_KG_PER_CM2:
        return FirmnessRating.UNDESIRABLE
    if value <= _ACCEPTABLE_CEILING_KG_PER_CM2:
        return FirmnessRating.ACCEPTABLE
    return FirmnessRating.DESIRABLE


def relative_density_from_firmness(firmness_kg_per_cm2: float) -> float:
    """Map a penetrometer reading onto a relative density in [0, 1].

    **This mapping is a modelling convention, not a measured correlation.** No
    published relation between penetrometer firmness and relative density for
    golf bunker sand was found; the anchors are chosen so the published sweep
    points 1.6-2.8 kg/cm^2 span loose to dense. Presets record it as such.

    Raises:
        SandModelError: if the reading is not a positive finite number.
    """
    value = _require_positive_firmness(firmness_kg_per_cm2, "kg/cm^2")
    span = FIRMNESS_DENSE_ANCHOR_KG_PER_CM2 - FIRMNESS_LOOSE_ANCHOR_KG_PER_CM2
    ratio = (value - FIRMNESS_LOOSE_ANCHOR_KG_PER_CM2) / span
    return min(1.0, max(0.0, ratio))
