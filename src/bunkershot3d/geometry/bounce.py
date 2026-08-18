"""Bounce-angle conventions for wedge sole geometry (issue #8609).

Two conventions are in circulation and they are *not* interchangeable:

``GeometricBounce``
    The Acushnet patent family's ``theta`` (US10143900B2 / US10661131B2):
    the angle of the chord from the **leading-edge point** to the sole's
    true trailing contact point.  Because the sole plunges by the sole
    entry height ``d3`` within the first 1.2 mm behind the leading edge,
    this angle is large - the patent claims ``> 20 deg`` and its worked
    examples are 15.99, 18.42 and 20.78 deg.

``MarketedBounce``
    What a wedge is *sold* as (4-14 deg): the angle measured to the
    ground-contact plane, i.e. from the effective leading edge at the
    1.2 mm datum rather than from the sharp leading-edge point.

They are modelled as **distinct types**, so mixing them is a
``TypeError`` rather than a silent numerical error, and converting
between them requires the datum geometry explicitly.

All angles are stored in degrees with an explicit ``_deg`` suffix and a
``.angle_rad`` accessor; every length is SI (metres).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeVar

__all__ = [
    "BounceAngle",
    "BounceConvention",
    "GeometricBounce",
    "MarketedBounce",
    "geometric_from_marketed",
    "marketed_from_geometric",
]

_MAX_BOUNCE_DEG = 90.0

BounceT = TypeVar("BounceT", bound="BounceAngle")


class BounceConvention(Enum):
    """Which reference the bounce angle was measured against."""

    GEOMETRIC = "geometric"
    """Chord from the leading-edge point to the true trailing contact point."""

    MARKETED = "marketed"
    """Angle to the effective ground-contact plane, as published by OEMs."""


@dataclass(frozen=True, slots=True)
class BounceAngle:
    """Base class for a convention-tagged bounce angle.

    Instantiate ``GeometricBounce`` or ``MarketedBounce``; this base type
    exists so that shared behaviour is written once, not so that untagged
    bounce angles can be created.

    Preconditions:
        ``angle_deg`` is a finite real number in ``(-90, 90)``.
    """

    angle_deg: float

    convention: ClassVar[BounceConvention]

    def __post_init__(self) -> None:
        if isinstance(self.angle_deg, bool) or not isinstance(
            self.angle_deg, (int, float)
        ):
            raise TypeError(
                f"bounce angle must be a real number, got "
                f"{type(self.angle_deg).__name__}"
            )
        value = float(self.angle_deg)
        if not math.isfinite(value):
            raise ValueError(f"bounce angle must be finite, got {self.angle_deg!r}")
        if abs(value) >= _MAX_BOUNCE_DEG:
            raise ValueError(
                "bounce angle must lie strictly between -90 and 90 degrees, "
                f"got {value}"
            )
        object.__setattr__(self, "angle_deg", value)

    @property
    def angle_rad(self) -> float:
        """The bounce angle in radians."""
        return math.radians(self.angle_deg)

    def shifted_by(self: BounceT, delta_deg: float) -> BounceT:
        """Return the same convention shifted by ``delta_deg`` degrees."""
        return type(self)(self.angle_deg + float(delta_deg))

    def _checked(self: BounceT, other: object, operator: str) -> float:
        if type(other) is not type(self):
            raise TypeError(
                f"cannot {operator} {type(self).__name__} and "
                f"{type(other).__name__}: bounce conventions do not mix; "
                "convert explicitly with marketed_from_geometric() or "
                "geometric_from_marketed()"
            )
        return float(other.angle_deg)  # type: ignore[attr-defined]

    def __add__(self: BounceT, other: object) -> BounceT:
        return type(self)(self.angle_deg + self._checked(other, "add"))

    def __sub__(self: BounceT, other: object) -> BounceT:
        return type(self)(self.angle_deg - self._checked(other, "subtract"))

    def __str__(self) -> str:
        return f"{self.angle_deg:.2f} deg ({self.convention.value})"


@dataclass(frozen=True, slots=True)
class GeometricBounce(BounceAngle):
    """Patent ``theta``: measured to the true trailing contact point."""

    convention: ClassVar[BounceConvention] = BounceConvention.GEOMETRIC


@dataclass(frozen=True, slots=True)
class MarketedBounce(BounceAngle):
    """Published bounce: measured to the effective ground-contact plane."""

    convention: ClassVar[BounceConvention] = BounceConvention.MARKETED


def _validate_datum(
    sole_width_m: float, entry_height_m: float, datum_offset_m: float
) -> None:
    for name, value in (
        ("sole_width_m", sole_width_m),
        ("entry_height_m", entry_height_m),
        ("datum_offset_m", datum_offset_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if sole_width_m <= datum_offset_m:
        raise ValueError(
            "sole width must exceed the measurement datum "
            f"({sole_width_m} m <= {datum_offset_m} m): there is no sole left "
            "behind the datum to measure a ground-contact plane against"
        )


def marketed_from_geometric(
    bounce: GeometricBounce,
    *,
    sole_width_m: float,
    entry_height_m: float,
    datum_offset_m: float,
) -> MarketedBounce:
    """Convert patent ``theta`` to the marketed (ground-plane) convention.

    The trailing contact point sits ``d1 * tan(theta)`` below the
    leading-edge point.  The effective leading edge for the marketed
    measurement is the sole entry point at the datum, ``d3`` below the
    leading-edge point and ``d2`` behind it, so

        theta_marketed = atan2(d1 * tan(theta) - d3, d1 - d2).

    Args:
        bounce: Geometric (patent) bounce angle.
        sole_width_m: ``d1``, rearward run from the leading-edge point to
            the trailing contact point.
        entry_height_m: ``d3``, sole drop over the datum offset.
        datum_offset_m: ``d2``, 1.2 mm in the patent schema.

    Returns:
        The same sole expressed in the marketed convention.

    Raises:
        TypeError: If ``bounce`` is not a ``GeometricBounce``.
        ValueError: If the datum geometry is not physically measurable.
    """
    if type(bounce) is not GeometricBounce:
        raise TypeError(
            f"marketed_from_geometric() needs a GeometricBounce, got "
            f"{type(bounce).__name__}"
        )
    _validate_datum(sole_width_m, entry_height_m, datum_offset_m)
    drop_m = sole_width_m * math.tan(bounce.angle_rad)
    return MarketedBounce(
        math.degrees(math.atan2(drop_m - entry_height_m, sole_width_m - datum_offset_m))
    )


def geometric_from_marketed(
    bounce: MarketedBounce,
    *,
    sole_width_m: float,
    entry_height_m: float,
    datum_offset_m: float,
) -> GeometricBounce:
    """Inverse of :func:`marketed_from_geometric`.

    Raises:
        TypeError: If ``bounce`` is not a ``MarketedBounce``.
        ValueError: If the datum geometry is not physically measurable.
    """
    if type(bounce) is not MarketedBounce:
        raise TypeError(
            f"geometric_from_marketed() needs a MarketedBounce, got "
            f"{type(bounce).__name__}"
        )
    _validate_datum(sole_width_m, entry_height_m, datum_offset_m)
    drop_m = entry_height_m + (sole_width_m - datum_offset_m) * math.tan(
        bounce.angle_rad
    )
    return GeometricBounce(math.degrees(math.atan2(drop_m, sole_width_m)))
