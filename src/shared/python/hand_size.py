"""Stature-tied hand length and breadth (HRG-HX3, #13760).

Hand length defaults to ``handlength / stature`` and breadth to
``handbreadth / handlength`` from the ANSUR II public summary (2012 U.S. Army
anthropometric survey; Gordon et al. 2015, NATICK/TR-15/007;
https://www.openlab.psu.edu/ansur2/). Column units are millimetres. Files
verified by sha256 against the official PSU distribution (mirror used when
PSU returns 403: ``cfgranda/ps4ds`` GitHub raw CSVs, byte-identical).

Male (n = 4082, sha256 0547aea0170e5293de519389981135e75803f02e6d40c77ace740decc0caac64):
  handlength/stature mean 0.1101, sd 0.0042, p5 0.1034, p95 0.1172.
Female (n = 1986, sha256 ed7e800aa97a4d42f8286b988be2fe1883a2e789588169db959864b707d1757a):
  handlength/stature mean 0.1112, sd 0.0047, p5 0.1039, p95 0.1193.
Sex unspecified uses the pooled mean handlength/stature 0.1104 (sd 0.0044).

Breadth/length (handbreadth/handlength) male mean 0.4572 (sd 0.0211, p5 0.4233,
p95 0.4921); female mean 0.4322 (sd 0.0210, p5 0.3977, p95 0.4671); pooled
mean 0.4490 (p5 0.4089, p95 0.4888). DbC breadth/length band uses the pooled
p5–p95 interval.

de Leva (1996) male table hand length 0.189 m at 1.754 m stature (~0.108) is a
length-only cross-check; de Leva publishes no hand breadth.
"""

from __future__ import annotations

import math
from typing import Final

HAND_LENGTH_RANGE_M: Final[tuple[float, float]] = (0.14, 0.24)

# Per-subject handlength / stature (ANSUR II public CSV, see module docstring).
HAND_LENGTH_STATURE_FRACTION_MALE: Final[float] = 0.1101
HAND_LENGTH_STATURE_FRACTION_FEMALE: Final[float] = 0.1112
HAND_LENGTH_STATURE_FRACTION_UNSPECIFIED: Final[float] = 0.1104

# Per-subject handbreadth / handlength.
BREADTH_OVER_LENGTH_MALE: Final[float] = 0.4572
BREADTH_OVER_LENGTH_FEMALE: Final[float] = 0.4322
BREADTH_OVER_LENGTH_UNSPECIFIED: Final[float] = 0.4490

# Pooled-sex p5–p95 of handbreadth / handlength (ANSUR II, n = 6068).
BREADTH_LENGTH_RATIO_RANGE: Final[tuple[float, float]] = (0.4089, 0.4888)

# Reference stature for pad/mesh scaling L0 (average adult male in physics_constants).
REFERENCE_STATURE_M: Final[float] = 1.75
REFERENCE_HAND_LENGTH_M: Final[float] = (
    REFERENCE_STATURE_M * HAND_LENGTH_STATURE_FRACTION_UNSPECIFIED
)
DEFAULT_HAND_LENGTH_M: Final[float] = REFERENCE_HAND_LENGTH_M


def hand_length_stature_fraction(sex: str) -> float:
    """Return the ANSUR II mean handlength/stature ratio for *sex*."""
    if sex == "M":
        return HAND_LENGTH_STATURE_FRACTION_MALE
    if sex == "F":
        return HAND_LENGTH_STATURE_FRACTION_FEMALE
    return HAND_LENGTH_STATURE_FRACTION_UNSPECIFIED


def breadth_over_length(sex: str) -> float:
    """Return the ANSUR II mean handbreadth/handlength ratio for *sex*."""
    if sex == "M":
        return BREADTH_OVER_LENGTH_MALE
    if sex == "F":
        return BREADTH_OVER_LENGTH_FEMALE
    return BREADTH_OVER_LENGTH_UNSPECIFIED


def validate_hand_length_m(value: float) -> float:
    """DbC: finite real hand length in ``HAND_LENGTH_RANGE_M``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("hand_length_m must be a finite real number")
    length = float(value)
    lo, hi = HAND_LENGTH_RANGE_M
    if not math.isfinite(length) or not (lo <= length <= hi):
        raise ValueError(f"hand_length_m must be in [{lo}, {hi}] m, got {length}")
    return length


def validate_breadth_length_ratio(breadth_m: float, length_m: float) -> None:
    """DbC: breadth/length lies in the documented ANSUR pooled p5–p95 band."""
    if isinstance(breadth_m, bool) or isinstance(length_m, bool):
        raise TypeError("hand breadth and length must be finite real numbers")
    if not (
        isinstance(breadth_m, int | float)
        and isinstance(length_m, int | float)
        and math.isfinite(float(breadth_m))
        and math.isfinite(float(length_m))
    ):
        raise TypeError("hand breadth and length must be finite real numbers")
    if length_m <= 0.0:
        raise ValueError("hand_length_m must be positive for breadth validation")
    ratio = float(breadth_m) / float(length_m)
    lo, hi = BREADTH_LENGTH_RATIO_RANGE
    if not (lo <= ratio <= hi):
        raise ValueError(
            f"hand_breadth_m/hand_length_m must be in [{lo}, {hi}], got {ratio}"
        )


def resolve_hand_length_m(
    stature_m: float,
    sex: str = "unspecified",
    *,
    override_m: float | None = None,
) -> float:
    """Hand length (m) from stature and sex, or an explicit override."""
    if override_m is not None:
        return validate_hand_length_m(override_m)
    if isinstance(stature_m, bool) or not isinstance(stature_m, int | float):
        raise TypeError("stature_m must be a finite positive number")
    stature = float(stature_m)
    if not math.isfinite(stature) or stature <= 0.0:
        raise ValueError(f"stature_m must be positive and finite, got {stature_m!r}")
    return validate_hand_length_m(stature * hand_length_stature_fraction(sex))


def resolve_hand_breadth_m(
    hand_length_m: float,
    sex: str = "unspecified",
    *,
    override_m: float | None = None,
) -> float:
    """Hand breadth (m) from length and sex, or an explicit override."""
    length = validate_hand_length_m(hand_length_m)
    if override_m is not None:
        breadth = float(override_m)
        if isinstance(override_m, bool) or not math.isfinite(breadth) or breadth <= 0:
            raise TypeError("hand_breadth_m must be a finite positive number")
        validate_breadth_length_ratio(breadth, length)
        return breadth
    breadth = length * breadth_over_length(sex)
    validate_breadth_length_ratio(breadth, length)
    return breadth


def hand_mesh_scale(hand_length_m: float) -> float:
    """Dimensionless visual scale ``L / L0`` for :class:`~grip_hand_mesh.HandMeshParams`."""
    length = validate_hand_length_m(hand_length_m)
    return length / REFERENCE_HAND_LENGTH_M


__all__ = [
    "BREADTH_LENGTH_RATIO_RANGE",
    "BREADTH_OVER_LENGTH_FEMALE",
    "BREADTH_OVER_LENGTH_MALE",
    "BREADTH_OVER_LENGTH_UNSPECIFIED",
    "DEFAULT_HAND_LENGTH_M",
    "HAND_LENGTH_RANGE_M",
    "HAND_LENGTH_STATURE_FRACTION_FEMALE",
    "HAND_LENGTH_STATURE_FRACTION_MALE",
    "HAND_LENGTH_STATURE_FRACTION_UNSPECIFIED",
    "REFERENCE_HAND_LENGTH_M",
    "REFERENCE_STATURE_M",
    "breadth_over_length",
    "hand_length_stature_fraction",
    "hand_mesh_scale",
    "resolve_hand_breadth_m",
    "resolve_hand_length_m",
    "validate_breadth_length_ratio",
    "validate_hand_length_m",
]
