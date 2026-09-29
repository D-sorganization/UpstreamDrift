"""Parametric wedge design vector (issue #8609, ADR-0032).

``WedgeGeometry`` is a frozen value object holding the OEM design vector
for a wedge sole, following the Acushnet sole-geometry patent family
(US10143900B2 / US10661131B2).  Every measurement is taken in a vertical
plane perpendicular to the leading edge, with

* ``x`` rearward from the leading-edge (LE) point,
* ``z`` up, and the ground plane tangent to the sole at the trailing
  contact point.

Schema (patent symbols in brackets):

===========================  ======  ==========================================
Field                        Symbol  Meaning
===========================  ======  ==========================================
``sole_width_m``             d1      LE point to trailing contact point (run)
``datum_offset_m``           d2      measurement datum, 1.2 mm rearward of LE
``entry_height_m``           d3      sole drop over the datum offset
``sole_entry_angle_deg``     Phi     derived: ``atan2(d3, d2)``
``leading_edge_radius_m``    rho1    sole radius over the first 1.2 mm
``trailing_edge_radius_m``   rho2    sole radius over the last 1.2 mm
``geometric_bounce``         theta   LE point to trailing contact point
``sole_camber_area_m2``      -       area between the sole and the LE/TC chord
===========================  ======  ==========================================

``Phi`` is *derived* rather than stored: it is fixed by ``d3`` and the
1.2 mm datum, and storing it separately would create two authoritative
representations of one fact.

Units are SI throughout (metres, kilograms); angles are named with an
explicit ``_deg`` / ``_rad`` suffix.  ``from_millimetres`` is the
convenience constructor for the patent's own units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum

from .bounce import GeometricBounce, MarketedBounce, marketed_from_geometric

__all__ = ["PatentBand", "WedgeGeometry"]

PATENT_DATUM_OFFSET_M = 1.2e-3
"""The Acushnet measurement datum: exactly 1.2 mm rearward of the LE point."""

_MAX_HEAD_MASS_KG = 1.0
_MAX_RELIEF_FRACTION = 0.9


class PatentBand(Enum):
    """Which band of the Acushnet claim set a parameter falls into.

    Band edges are treated as inclusive: a value exactly on a threshold
    counts as being inside the tighter band.
    """

    OUT_OF_RANGE = "out_of_range"
    BROAD = "broad"
    PREFERRED = "preferred"
    MOST_PREFERRED = "most_preferred"


def _band_at_least(
    value: float, broad: float, preferred: float, most: float
) -> PatentBand:
    """Classify a value whose claim thresholds are lower bounds."""
    if value >= most:
        return PatentBand.MOST_PREFERRED
    if value >= preferred:
        return PatentBand.PREFERRED
    if value >= broad:
        return PatentBand.BROAD
    return PatentBand.OUT_OF_RANGE


def _band_at_most(
    value: float, broad: float, preferred: float, most: float
) -> PatentBand:
    """Classify a value whose claim thresholds are upper bounds."""
    if value <= most:
        return PatentBand.MOST_PREFERRED
    if value <= preferred:
        return PatentBand.PREFERRED
    if value <= broad:
        return PatentBand.BROAD
    return PatentBand.OUT_OF_RANGE


def _band_between(
    value: float,
    lower_broad: float,
    lower_preferred: float,
    lower_most: float,
    upper: float,
) -> PatentBand:
    """Classify a value with a shared upper bound and rising lower bounds."""
    if value > upper or value < lower_broad:
        return PatentBand.OUT_OF_RANGE
    if value >= lower_most:
        return PatentBand.MOST_PREFERRED
    if value >= lower_preferred:
        return PatentBand.PREFERRED
    return PatentBand.BROAD


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_fraction(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= _MAX_RELIEF_FRACTION:
        raise ValueError(
            f"{name} must lie in [0, {_MAX_RELIEF_FRACTION}], got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class WedgeGeometry:
    """The design vector of a wedge head.

    Preconditions (enforced with ``raise``, never ``assert``, because
    ``python -O`` strips assertions):

    * loft in ``(0, 90)`` degrees, lie in ``(0, 90]`` degrees;
    * every length, area, radius and mass strictly positive and finite;
    * ``sole_width_m > datum_offset_m`` - there must be sole behind the
      datum to measure;
    * ``leading_edge_radius_m < trailing_edge_radius_m`` - the schema's
      Sole Contour Ratio is defined as a number below one;
    * ``entry_height_m`` exceeds the bounce chord's drop at the datum,
      otherwise the sole is concave and the camber area is not defined;
    * relief fractions in ``[0, 0.9]``;
    * head mass in ``(0, 1)`` kg.
    """

    loft_deg: float
    lie_deg: float
    geometric_bounce: GeometricBounce
    sole_width_m: float
    entry_height_m: float
    leading_edge_radius_m: float
    trailing_edge_radius_m: float
    sole_camber_area_m2: float
    centre_rocker_radius_m: float
    heel_rocker_radius_m: float
    toe_rocker_radius_m: float
    trailing_relief_fraction: float
    heel_relief_fraction: float
    toe_relief_fraction: float
    face_progression_m: float
    blade_length_m: float
    face_height_m: float
    topline_width_m: float
    head_mass_kg: float
    datum_offset_m: float = PATENT_DATUM_OFFSET_M

    def __post_init__(self) -> None:
        if type(self.geometric_bounce) is not GeometricBounce:
            raise TypeError(
                "geometric_bounce must be a GeometricBounce (patent theta, "
                "measured to the true trailing contact point), got "
                f"{type(self.geometric_bounce).__name__}"
            )
        if not math.isfinite(self.loft_deg) or not 0.0 < self.loft_deg < 90.0:
            raise ValueError(
                f"loft_deg must lie strictly in (0, 90), got {self.loft_deg!r}"
            )
        if not math.isfinite(self.lie_deg) or not 0.0 < self.lie_deg <= 90.0:
            raise ValueError(f"lie_deg must lie in (0, 90], got {self.lie_deg!r}")

        for name in (
            "sole_width_m",
            "entry_height_m",
            "leading_edge_radius_m",
            "trailing_edge_radius_m",
            "sole_camber_area_m2",
            "centre_rocker_radius_m",
            "heel_rocker_radius_m",
            "toe_rocker_radius_m",
            "blade_length_m",
            "face_height_m",
            "topline_width_m",
            "head_mass_kg",
            "datum_offset_m",
        ):
            _require_positive_finite(name, float(getattr(self, name)))
        _require_finite("face_progression_m", float(self.face_progression_m))
        for name in (
            "trailing_relief_fraction",
            "heel_relief_fraction",
            "toe_relief_fraction",
        ):
            _require_fraction(name, float(getattr(self, name)))

        if self.head_mass_kg >= _MAX_HEAD_MASS_KG:
            raise ValueError(
                f"head_mass_kg must be below {_MAX_HEAD_MASS_KG} kg (a wedge "
                f"head is 0.29-0.31 kg), got {self.head_mass_kg!r}"
            )
        if self.sole_width_m <= 2.0 * self.datum_offset_m:
            raise ValueError(
                "sole_width_m must exceed twice the measurement datum "
                f"({self.sole_width_m} m vs {self.datum_offset_m} m): the "
                "leading and trailing sole radii are measured over the first "
                "and last datum offsets and cannot overlap"
            )
        if self.leading_edge_radius_m >= self.trailing_edge_radius_m:
            raise ValueError(
                "leading_edge_radius_m must be smaller than "
                "trailing_edge_radius_m (Sole Contour Ratio is defined below "
                f"one), got {self.leading_edge_radius_m} >= "
                f"{self.trailing_edge_radius_m}"
            )
        tangent = math.tan(self.geometric_bounce.angle_rad)
        chord_drop_at_datum_m = self.datum_offset_m * tangent
        if self.entry_height_m <= chord_drop_at_datum_m:
            raise ValueError(
                "entry_height_m must exceed the bounce chord's drop at the "
                f"datum ({chord_drop_at_datum_m:.6g} m), otherwise the sole "
                f"is concave; got {self.entry_height_m!r}"
            )
        relieved_width_m = self.sole_width_m * (
            1.0 - max(self.heel_relief_fraction, self.toe_relief_fraction)
        )
        if relieved_width_m * tangent <= self.entry_height_m:
            admissible = 1.0 - self.entry_height_m / (self.sole_width_m * tangent)
            raise ValueError(
                "heel/toe relief moves the trailing contact point in front of "
                "the sole entry: a relieved sole of "
                f"{relieved_width_m * 1e3:.3g} mm drops only "
                f"{relieved_width_m * tangent * 1e3:.3g} mm, less than the "
                f"{self.entry_height_m * 1e3:.3g} mm entry height. The most "
                f"this sole admits is {admissible:.3f}"
            )

    # -- alternative constructor ------------------------------------------

    @classmethod
    def from_millimetres(
        cls,
        *,
        loft_deg: float,
        lie_deg: float,
        geometric_bounce: GeometricBounce,
        sole_width_mm: float,
        entry_height_mm: float,
        leading_edge_radius_mm: float,
        trailing_edge_radius_mm: float,
        sole_camber_area_mm2: float,
        centre_rocker_radius_mm: float,
        heel_rocker_radius_mm: float,
        toe_rocker_radius_mm: float,
        trailing_relief_fraction: float,
        heel_relief_fraction: float,
        toe_relief_fraction: float,
        face_progression_mm: float,
        blade_length_mm: float,
        face_height_mm: float,
        topline_width_mm: float,
        head_mass_g: float,
        datum_offset_mm: float = 1.2,
    ) -> WedgeGeometry:
        """Build from the patent's own units (mm, mm^2, grams)."""
        return cls(
            loft_deg=float(loft_deg),
            lie_deg=float(lie_deg),
            geometric_bounce=geometric_bounce,
            sole_width_m=float(sole_width_mm) * 1e-3,
            entry_height_m=float(entry_height_mm) * 1e-3,
            leading_edge_radius_m=float(leading_edge_radius_mm) * 1e-3,
            trailing_edge_radius_m=float(trailing_edge_radius_mm) * 1e-3,
            sole_camber_area_m2=float(sole_camber_area_mm2) * 1e-6,
            centre_rocker_radius_m=float(centre_rocker_radius_mm) * 1e-3,
            heel_rocker_radius_m=float(heel_rocker_radius_mm) * 1e-3,
            toe_rocker_radius_m=float(toe_rocker_radius_mm) * 1e-3,
            trailing_relief_fraction=float(trailing_relief_fraction),
            heel_relief_fraction=float(heel_relief_fraction),
            toe_relief_fraction=float(toe_relief_fraction),
            face_progression_m=float(face_progression_mm) * 1e-3,
            blade_length_m=float(blade_length_mm) * 1e-3,
            face_height_m=float(face_height_mm) * 1e-3,
            topline_width_m=float(topline_width_mm) * 1e-3,
            head_mass_kg=float(head_mass_g) * 1e-3,
            datum_offset_m=float(datum_offset_mm) * 1e-3,
        )

    # -- angles ------------------------------------------------------------

    @property
    def loft_rad(self) -> float:
        """Static loft in radians."""
        return math.radians(self.loft_deg)

    @property
    def lie_rad(self) -> float:
        """Static lie angle (shaft to ground) in radians."""
        return math.radians(self.lie_deg)

    # -- derived schema quantities ----------------------------------------

    @property
    def sole_entry_angle_deg(self) -> float:
        """Patent ``Phi``: the LE-to-datum chord angle above the horizontal."""
        return math.degrees(math.atan2(self.entry_height_m, self.datum_offset_m))

    @property
    def sole_contour_ratio(self) -> float:
        """Patent Sole Contour Ratio ``rho1 / rho2`` (dimensionless)."""
        return self.leading_edge_radius_m / self.trailing_edge_radius_m

    @property
    def camber_to_bounce_ratio_mm2_per_deg(self) -> float:
        """Camber area per degree of geometric bounce, in mm^2/deg."""
        return (self.sole_camber_area_m2 * 1e6) / self.geometric_bounce.angle_deg

    @property
    def trailing_contact_drop_m(self) -> float:
        """How far the trailing contact point sits below the LE point."""
        return self.sole_width_m * math.tan(self.geometric_bounce.angle_rad)

    @property
    def marketed_bounce(self) -> MarketedBounce:
        """The same sole in the published (ground-plane) convention."""
        return marketed_from_geometric(
            self.geometric_bounce,
            sole_width_m=self.sole_width_m,
            entry_height_m=self.entry_height_m,
            datum_offset_m=self.datum_offset_m,
        )

    # -- introspection -----------------------------------------------------

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Every field of the design vector, in declaration order."""
        return tuple(field.name for field in fields(cls))

    @staticmethod
    def patent_parameters() -> tuple[str, ...]:
        """The parameters the Acushnet claim set puts a numeric band on."""
        return (
            "sole_width",
            "entry_height",
            "sole_entry_angle",
            "leading_edge_radius",
            "trailing_edge_radius",
            "geometric_bounce",
            "sole_camber_area",
            "sole_contour_ratio",
            "camber_to_bounce_ratio",
        )

    def patent_compliance(self) -> dict[str, PatentBand]:
        """Classify each schema parameter against the patent's bands.

        Out-of-range parameters are *reported*, not rejected: the claim
        bands describe the patented design space, and a design tool must
        be able to explore outside it.

        Returns:
            One :class:`PatentBand` per :meth:`patent_parameters` entry.
        """
        width_mm = self.sole_width_m * 1e3
        entry_mm = self.entry_height_m * 1e3
        return {
            "sole_width": _band_between(width_mm, 5.0, 10.0, 15.0, 22.0),
            "entry_height": _band_between(entry_mm, 2.0, 2.5, 3.0, 8.0),
            "sole_entry_angle": _band_at_least(
                self.sole_entry_angle_deg, 60.0, 65.0, 67.5
            ),
            "leading_edge_radius": _band_at_most(
                self.leading_edge_radius_m * 1e3, 10.0, 9.0, 8.0
            ),
            "trailing_edge_radius": _band_at_least(
                self.trailing_edge_radius_m * 1e3, 40.0, 41.0, 42.0
            ),
            # The claim threshold is 20 deg; the lower bands are anchored on
            # the patent's own worked examples (15.99, 18.42, 20.78 deg).
            "geometric_bounce": _band_at_least(
                self.geometric_bounce.angle_deg, 15.99, 18.0, 20.0
            ),
            "sole_camber_area": _band_at_least(
                self.sole_camber_area_m2 * 1e6, 42.0, 45.0, 48.0
            ),
            "sole_contour_ratio": _band_at_most(
                self.sole_contour_ratio, 0.25, 0.21, 0.19
            ),
            "camber_to_bounce_ratio": _band_at_least(
                self.camber_to_bounce_ratio_mm2_per_deg, 2.0, 2.5, 3.0
            ),
        }
