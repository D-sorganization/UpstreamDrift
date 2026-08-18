"""Unit convention and conversions for BunkerShot3D (issue #8608, ADR-0032).

**The convention: SI internally, always.** Metres, kilograms, seconds, pascals,
newtons, radians. Nothing inside this package stores a quantity in millimetres,
grams, degrees or miles per hour. Non-SI units exist only at the boundary --
where a patent quotes millimetres, an OEM quotes grams, or a golfer quotes
degrees and miles per hour -- and are converted on the way in and on the way
out by the helpers below.

**Every physical quantity is named with its unit.** ``sole_width_m``,
``dt_s``, ``clubhead_speed_mps``, ``theta_rad``, ``loft_deg``,
``density_kg_m3``. This is not a style preference: it is the only thing that
makes a degree/radian mix-up visible at the call site, and
:func:`si_unit_for` makes it machine-checkable. Angles are the sharp case, so
they never appear unsuffixed -- there is no ``loft``, only ``loft_deg`` or
``loft_rad``.

Degrees are kept as a first-class *tagged* unit rather than banned. Loft,
bounce, lie and attack angle are quoted in degrees by every source this package
reads (patents, OEM specs, launch monitors), and silently storing them in
radians would make every literal in the codebase unrecognisable to the people
who have to check it. The suffix carries the tag.

The conversion registry :data:`CONVERSIONS` is the single list of conversions
this package offers; ``tests/bunkershot3d/test_units_8608.py`` iterates it, so
a conversion cannot be added without acquiring a round-trip property test.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .exceptions import UnitConversionError, UnitConventionError

__all__ = [
    "CONVERSIONS",
    "MILE_M",
    "SI_SUFFIXES",
    "Conversion",
    "deg_to_rad",
    "g_to_kg",
    "hz_to_period_s",
    "kg_to_g",
    "kpa_to_pa",
    "m_to_mm",
    "m2_to_mm2",
    "mm_to_m",
    "mm2_to_m2",
    "mph_to_mps",
    "mps_to_mph",
    "pa_to_kpa",
    "period_s_to_hz",
    "rad_to_deg",
    "si_unit_for",
]

#: One international mile in metres, exactly, by the 1959 agreement.
MILE_M = 1609.344

_MM_PER_M = 1.0e3
_MM2_PER_M2 = 1.0e6
_G_PER_KG = 1.0e3
_PA_PER_KPA = 1.0e3
_SECONDS_PER_HOUR = 3600.0


def _finite(value: float, what: str) -> float:
    """Return ``value`` as a float, refusing NaN and infinities.

    Args:
        value: The quantity to convert.
        what: Description used in the error message.

    Returns:
        ``float(value)``.

    Raises:
        UnitConversionError: ``value`` is not finite, or is not a number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UnitConversionError(f"{what} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise UnitConversionError(f"{what} must be finite, got {number!r}")
    return number


def _positive(value: float, what: str) -> float:
    """Return ``value`` as a strictly positive float.

    Args:
        value: The quantity to convert.
        what: Description used in the error message.

    Returns:
        ``float(value)``.

    Raises:
        UnitConversionError: ``value`` is not finite or is not positive.
    """
    number = _finite(value, what)
    if number <= 0.0:
        raise UnitConversionError(f"{what} must be positive, got {number!r}")
    return number


# ---------------------------------------------------------------------------
# Angle
# ---------------------------------------------------------------------------


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(_finite(deg, "angle in degrees"))


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(_finite(rad, "angle in radians"))


# ---------------------------------------------------------------------------
# Length, area, mass, pressure
# ---------------------------------------------------------------------------


def mm_to_m(mm: float) -> float:
    """Convert millimetres to metres."""
    return _finite(mm, "length in millimetres") / _MM_PER_M


def m_to_mm(m: float) -> float:
    """Convert metres to millimetres."""
    return _finite(m, "length in metres") * _MM_PER_M


def mm2_to_m2(mm2: float) -> float:
    """Convert square millimetres to square metres."""
    return _finite(mm2, "area in square millimetres") / _MM2_PER_M2


def m2_to_mm2(m2: float) -> float:
    """Convert square metres to square millimetres."""
    return _finite(m2, "area in square metres") * _MM2_PER_M2


def g_to_kg(g: float) -> float:
    """Convert grams to kilograms."""
    return _finite(g, "mass in grams") / _G_PER_KG


def kg_to_g(kg: float) -> float:
    """Convert kilograms to grams."""
    return _finite(kg, "mass in kilograms") * _G_PER_KG


def kpa_to_pa(kpa: float) -> float:
    """Convert kilopascals to pascals."""
    return _finite(kpa, "pressure in kilopascals") * _PA_PER_KPA


def pa_to_kpa(pa: float) -> float:
    """Convert pascals to kilopascals."""
    return _finite(pa, "pressure in pascals") / _PA_PER_KPA


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def mph_to_mps(mph: float) -> float:
    """Convert miles per hour to metres per second."""
    return _finite(mph, "speed in miles per hour") * MILE_M / _SECONDS_PER_HOUR


def mps_to_mph(mps: float) -> float:
    """Convert metres per second to miles per hour."""
    return _finite(mps, "speed in metres per second") * _SECONDS_PER_HOUR / MILE_M


# ---------------------------------------------------------------------------
# Rate <-> period (the one reciprocal conversion)
# ---------------------------------------------------------------------------


def hz_to_period_s(hz: float) -> float:
    """Convert a rate in hertz to the corresponding period in seconds.

    Raises:
        UnitConversionError: The rate is not a positive finite number. A zero
            or negative rate has no period, and a sampling rate of zero is the
            kind of silent division that produced defect B30.
    """
    return 1.0 / _positive(hz, "rate in hertz")


def period_s_to_hz(period_s: float) -> float:
    """Convert a period in seconds to the corresponding rate in hertz.

    Raises:
        UnitConversionError: The period is not a positive finite number.
    """
    return 1.0 / _positive(period_s, "period in seconds")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Conversion:
    """One convertible quantity and its two directions.

    Attributes:
        quantity: What is being measured (``"angle"``, ``"length"``, ...).
        si_unit: Symbol of the SI unit this package stores internally.
        other_unit: Symbol of the boundary unit.
        to_si: Callable mapping ``other_unit`` to ``si_unit``.
        from_si: Callable mapping ``si_unit`` to ``other_unit``.
        positive_only: The conversion is defined only for positive values.
    """

    quantity: str
    si_unit: str
    other_unit: str
    to_si: Callable[[float], float]
    from_si: Callable[[float], float]
    positive_only: bool = False


CONVERSIONS: Mapping[str, Conversion] = MappingProxyType(
    {
        conversion.quantity: conversion
        for conversion in (
            Conversion("angle", "rad", "deg", deg_to_rad, rad_to_deg),
            Conversion("length", "m", "mm", mm_to_m, m_to_mm),
            Conversion("area", "m^2", "mm^2", mm2_to_m2, m2_to_mm2),
            Conversion("mass", "kg", "g", g_to_kg, kg_to_g),
            Conversion("pressure", "Pa", "kPa", kpa_to_pa, pa_to_kpa),
            Conversion("speed", "m/s", "mph", mph_to_mps, mps_to_mph),
            Conversion(
                "rate",
                "s",
                "Hz",
                hz_to_period_s,
                period_s_to_hz,
                positive_only=True,
            ),
        )
    }
)
"""Every conversion this package offers, keyed by quantity."""


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------

#: Recognised unit suffixes, mapped to the unit symbol they denote. Ordered
#: longest-first at lookup time so ``_kg_m3`` is not read as ``_m3``.
SI_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        "_deg": "deg",
        "_hz": "Hz",
        "_j": "J",
        "_k": "K",
        "_kg": "kg",
        "_kg_m3": "kg/m^3",
        "_m": "m",
        "_m2": "m^2",
        "_m3": "m^3",
        "_mps": "m/s",
        "_mps2": "m/s^2",
        "_n": "N",
        "_nm": "N.m",
        "_pa": "Pa",
        "_rad": "rad",
        "_rad_s": "rad/s",
        "_s": "s",
    }
)

_SUFFIXES_LONGEST_FIRST: tuple[tuple[str, str], ...] = tuple(
    sorted(SI_SUFFIXES.items(), key=lambda item: -len(item[0]))
)


def si_unit_for(name: str) -> str:
    """Return the unit symbol implied by a field or variable name's suffix.

    This is what gives the naming convention teeth: a test can walk the fields
    of a value object and require that each one either is explicitly
    dimensionless or resolves here.

    Args:
        name: A field, attribute or parameter name, e.g. ``"sole_width_m"``.

    Returns:
        The unit symbol, e.g. ``"m"``.

    Raises:
        UnitConventionError: ``name`` carries no recognised unit suffix.
    """
    text = str(name)
    for suffix, unit in _SUFFIXES_LONGEST_FIRST:
        if text.endswith(suffix) and len(text) > len(suffix):
            return unit
    raise UnitConventionError(
        f"{name!r} carries no recognised unit suffix. Physical quantities in "
        "this package are named with their unit (dt_s, v_mps, theta_rad); "
        f"known suffixes are {sorted(SI_SUFFIXES)}."
    )
