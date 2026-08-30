"""The sand a strike accelerates, as an interval rather than a point.

The prismatic divot volume counts only sand the sole swept. Issue #8659
showed that is inadmissible: dividing the solver's impulse by it implied
sand leaving at 47 m/s from a 25 m/s head, and sand cannot leave faster
than the thing that threw it.

F1 measured the shortfall at 2.845-3.898x over the workbench's design
space. The correction is an **interval**, not a corrected point, because
F1 is plane strain and structurally blind to the divot's walls: the
lower edge takes F1 literally with no lateral spread, the upper widens
the trench at the bed's own friction angle, and the central value is a
stated convention rather than a measurement.

It lives apart from :mod:`~bunkershot3d.metrics.divot` because it is a
distinct concept with its own provenance -- a consistency correction
between two uncalibrated models, not a measurement of sand.
:mod:`~bunkershot3d.metrics.divot` re-exports these names, so existing
imports keep working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..exceptions import BunkerShot3DValueError

__all__ = [
    "ACCELERATED_MASS_CONSISTENCY_REASON",
    "ACCELERATED_MASS_LATERAL_REASON",
    "F1_ENTRAINMENT_FACTOR_BOUNDS",
    "AcceleratedSandMass",
    "lateral_spread_factor",
]

F1_ENTRAINMENT_FACTOR_BOUNDS: tuple[float, float] = (2.84, 3.90)


ACCELERATED_MASS_CONSISTENCY_REASON = (
    "the accelerated sand mass is a **consistency** correction between two "
    "uncalibrated models and not a measurement. Its in-plane factor of "
    f"{F1_ENTRAINMENT_FACTOR_BOUNDS[0]:.3g}-"
    f"{F1_ENTRAINMENT_FACTOR_BOUNDS[1]:.3g} was read off the "
    "F1 MPM tier, which is BEYOND_VALIDATION, whose published-speed ceiling is "
    "1.44 m/s against a 25 m/s greenside delivery and whose NASA-STD-7009B "
    "validation is 0 of 4; no ejecta mass has ever been measured on a real "
    "bunker shot (issue #8616). Two uncalibrated tiers agreeing is two "
    "uncalibrated tiers agreeing. What the comparison can do is falsify, and "
    "it did: the prismatic mass was inadmissible against the head's own entry "
    "speed and the corrected interval is not (issue #8659)"
)
ACCELERATED_MASS_LATERAL_REASON = (
    "the upper edge of the accelerated mass widens the divot's walls to the "
    "bed's own friction angle, on the argument that a trench cut in a "
    "cohesionless sand cannot stand steeper than the material it is cut in. "
    "The angle is the bed's, so nothing new is fitted, but the shape is a "
    "model: F1 is plane strain and cannot see out of the plane, so no tier in "
    "this package has measured what the walls actually do (issue #8659)"
)


@dataclass(frozen=True, slots=True)
class AcceleratedSandMass:
    """The sand one strike set in motion, as an **interval** and not a point.

    Why an interval. The swept prism is not the mass that shared the delivered
    momentum, and issue #8659 is the arithmetic that proves it: at the nominal
    greenside shot the solver's 2.917 N.s over the prism's 63.7 g implies sand
    leaving at 45.8 m/s from a 25.0 m/s head. Something has to be bigger, and
    it is the mass.

    Two separate things are missing from the prism and only one of them has
    been looked at by a solver. **In plane**, the head throws a bow wave ahead
    of its leading edge and heaves material above the original surface, and
    the F1 MPM tier resolves that -- see
    :data:`F1_ENTRAINMENT_FACTOR_BOUNDS` for what it measured and under what
    licence. **Out of plane**, the divot's walls slope away from the sole and
    a plane-strain tier cannot see them at all, so that half is a stated model
    on the bed's own friction angle rather than a measurement.

    The interval is built so that each edge says which of those it rests on:

    * :attr:`lower_kg` is the prism scaled by the *smallest* in-plane factor
      the F1 sweep produced and no lateral spread whatsoever -- the
      plane-strain reading, taken literally.
    * :attr:`upper_kg` is the prism scaled by the largest in-plane factor and
      by the lateral widening a wall at the bed's friction angle implies.
    * :attr:`central_kg` is their geometric mean, because the two edges are
      multiplicative factors rather than additive offsets. A convention, and
      the only number here that is neither measured nor derived.

    None of this is calibrated against a bunker; nothing in this package is.
    :data:`ACCELERATED_MASS_CONSISTENCY_REASON` says so in the words a
    verdict carries.

    Attributes:
        prismatic_kg: The swept-prism mass this was formed from [kg], which
            is :attr:`DivotMetrics.mass_kg` unchanged.
        entrainment_lower: Smallest in-plane factor of the F1 sweep.
        entrainment_upper: Largest in-plane factor of the F1 sweep.
        lateral_factor: Out-of-plane widening, ``>= 1``, from the wall model.
        wall_angle_deg: The friction angle the walls were laid back at.
    """

    prismatic_kg: float
    entrainment_lower: float
    entrainment_upper: float
    lateral_factor: float
    wall_angle_deg: float

    def __post_init__(self) -> None:
        """Refuse an interval that is not one.

        Raises:
            ValueError: If the prism is not positive, if either factor is not
                at least one, if the factors are out of order, or if the
                lateral widening is below one. A plain ``raise`` and not a
                contract: ball launch divides by the number this object
                produces, and ``python -O`` strips assertions.
        """
        if not np.isfinite(self.prismatic_kg) or self.prismatic_kg <= 0.0:
            raise ValueError(
                "the prismatic divot mass must be positive and finite, got "
                f"{self.prismatic_kg!r}"
            )
        for name in ("entrainment_lower", "entrainment_upper", "lateral_factor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 1.0:
                raise ValueError(
                    f"{name} must be a finite factor of at least 1 -- the sand "
                    "outside the swept prism can only add mass, never remove "
                    f"it -- got {value!r}"
                )
        if self.entrainment_upper < self.entrainment_lower:
            raise ValueError(
                "the entrainment bounds are out of order: "
                f"{self.entrainment_lower!r} above {self.entrainment_upper!r}"
            )
        if not np.isfinite(self.wall_angle_deg) or not 0.0 < self.wall_angle_deg < 90.0:
            raise ValueError(
                "the wall angle must lie in (0, 90) degrees, got "
                f"{self.wall_angle_deg!r}"
            )

    @property
    def lower_kg(self) -> float:
        """Smallest admissible accelerated mass [kg]: F1 in plane, no walls."""
        return self.prismatic_kg * self.entrainment_lower

    @property
    def upper_kg(self) -> float:
        """Largest [kg]: the F1 sweep's widest factor, walls laid back."""
        return self.prismatic_kg * self.entrainment_upper * self.lateral_factor

    @property
    def central_kg(self) -> float:
        """Geometric mean of the two edges [kg]. A convention, not a value."""
        return float(np.sqrt(self.lower_kg * self.upper_kg))

    @property
    def bounds_kg(self) -> tuple[float, float]:
        """The interval, ``(lower, upper)`` [kg]."""
        return (self.lower_kg, self.upper_kg)

    def summary(self) -> str:
        """A line fit for a report, carrying the interval and not the point."""
        return (
            f"{self.central_kg * 1e3:.4g} g "
            f"[{self.lower_kg * 1e3:.4g}-{self.upper_kg * 1e3:.4g} g], "
            f"{self.central_kg / self.prismatic_kg:.3g}x the swept prism "
            f"({self.prismatic_kg * 1e3:.4g} g)"
        )


def lateral_spread_factor(
    section_area_m2: float,
    depth_squared_integral_m3: float,
    *,
    width_m: float,
    wall_angle_deg: float,
) -> float:
    """How much the divot's sloping walls widen the swept prism.

    A trench of bottom width ``w`` and depth ``d(s)`` whose walls lie back at
    the bed's friction angle ``phi`` measured from the horizontal has section
    ``w d + d^2 cot(phi)``, so the whole volume is
    ``w * integral d ds + cot(phi) * integral d^2 ds`` and the widening is the
    ratio of that to the prism. The argument for the angle is that a trench
    cut in a cohesionless sand cannot stand steeper than the material it is
    cut in; the argument for the *shape* is only that it is the simplest one
    that has the right limits, and :data:`ACCELERATED_MASS_LATERAL_REASON`
    says so.

    Args:
        section_area_m2: ``integral of depth ds`` over the divot [m^2].
        depth_squared_integral_m3: ``integral of depth^2 ds`` over the same
            window [m^3].
        width_m: Sole width in contact [m].
        wall_angle_deg: Friction angle of the bed [deg], from the sand state.

    Returns:
        The widening factor, at least 1.

    Raises:
        ValueError: If any argument is out of range. A ``raise`` for the same
            reason :class:`AcceleratedSandMass` uses one.
    """
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise ValueError(f"width_m must be positive and finite, got {width_m}")
    if not np.isfinite(wall_angle_deg) or not 0.0 < wall_angle_deg < 90.0:
        raise ValueError(
            f"wall_angle_deg must lie in (0, 90) degrees, got {wall_angle_deg}"
        )
    for name, value in (
        ("section_area_m2", section_area_m2),
        ("depth_squared_integral_m3", depth_squared_integral_m3),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if section_area_m2 <= 0.0:
        return 1.0
    cotangent = 1.0 / np.tan(np.radians(float(wall_angle_deg)))
    return 1.0 + float(cotangent) * float(depth_squared_integral_m3) / (
        float(width_m) * float(section_area_m2)
    )
