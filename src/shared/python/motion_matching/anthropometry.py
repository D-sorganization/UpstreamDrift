"""Segment anthropometry from stature and body mass (de Leva 1996 male table).

The golfer's stature and mass are not measured, so segment lengths, masses,
centre-of-mass positions and radii of gyration are derived from a stature and
mass estimate with the adjusted Zatsiorsky-Seluyanov parameters published by
de Leva (J. Biomech. 29(9), 1996, Table 4, males; reference subject 1.741 m,
73.0 kg). Lengths scale linearly with stature, masses with body mass, and
inertia tensors follow ``m (k L)^2`` with the tabulated radii of gyration.
The table below was transcribed for this module; before any qualification
claim it must be checked line by line against the paper.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

REFERENCE_STATURE_M = 1.741
REFERENCE_MASS_KG = 73.0


@dataclass(frozen=True)
class SegmentTable:
    """One row of de Leva's male table (lengths in metres for the reference subject)."""

    length_m: float
    mass_fraction: float
    com_fraction: float  # from the proximal end, along the segment
    radii: tuple[
        float, float, float
    ]  # sagittal, transverse, longitudinal, as fractions


# de Leva 1996 Table 4 (males), transcribed. Segment ends: head = vertex to
# cervicale; trunk = cervicale to hip joint centre; upper arm = shoulder to
# elbow joint centres; forearm = elbow to wrist; hand = wrist to metacarpale
# III; thigh = hip to knee; shank = knee to ankle; foot = heel to toe tip.
DE_LEVA_MALE: Mapping[str, SegmentTable] = {
    "head": SegmentTable(0.2429, 0.0694, 0.5002, (0.303, 0.315, 0.261)),
    "trunk": SegmentTable(0.5319, 0.4346, 0.5138, (0.328, 0.306, 0.169)),
    "upper_trunk": SegmentTable(0.1707, 0.1596, 0.2999, (0.505, 0.465, 0.418)),
    "middle_trunk": SegmentTable(0.2155, 0.1633, 0.4502, (0.482, 0.383, 0.468)),
    "lower_trunk": SegmentTable(0.1457, 0.1117, 0.6115, (0.615, 0.551, 0.587)),
    "upper_arm": SegmentTable(0.2817, 0.0271, 0.5772, (0.285, 0.269, 0.158)),
    "forearm": SegmentTable(0.2689, 0.0162, 0.4574, (0.276, 0.265, 0.121)),
    "hand": SegmentTable(0.0862, 0.0061, 0.7900, (0.628, 0.513, 0.401)),
    "thigh": SegmentTable(0.4222, 0.1416, 0.4095, (0.329, 0.329, 0.149)),
    "shank": SegmentTable(0.4340, 0.0433, 0.4395, (0.251, 0.246, 0.102)),
    "foot": SegmentTable(0.2581, 0.0137, 0.4415, (0.257, 0.245, 0.124)),
}


@dataclass(frozen=True)
class SegmentParameters:
    """Derived parameters of one segment for a given stature and mass."""

    length_m: float
    mass_kg: float
    com_from_proximal_m: float
    inertia_principal_kg_m2: tuple[float, float, float]  # sagittal, transverse, long.


def _check_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def segment_parameters(
    stature_m: float, mass_kg: float, segment: str
) -> SegmentParameters:
    """Scale one table row to a subject. Precondition: known segment, positive inputs.

    Postcondition: ``mass_kg`` equals the table fraction of body mass and the
    inertia components are ``mass (k length)^2`` about the centre of mass.
    """
    _check_positive(stature_m, "stature_m")
    _check_positive(mass_kg, "mass_kg")
    if segment not in DE_LEVA_MALE:
        raise ValueError(f"Unknown segment {segment}; known: {sorted(DE_LEVA_MALE)}")
    row = DE_LEVA_MALE[segment]
    length = row.length_m * stature_m / REFERENCE_STATURE_M
    mass = row.mass_fraction * mass_kg
    inertia = tuple(float(mass * (k * length) ** 2) for k in row.radii)
    return SegmentParameters(
        length_m=length,
        mass_kg=mass,
        com_from_proximal_m=row.com_fraction * length,
        inertia_principal_kg_m2=(inertia[0], inertia[1], inertia[2]),
    )


def whole_body(stature_m: float, mass_kg: float) -> dict[str, SegmentParameters]:
    """Every table segment scaled to the subject (bilateral limbs listed once)."""
    return {name: segment_parameters(stature_m, mass_kg, name) for name in DE_LEVA_MALE}


def stature_from_segment_lengths(lengths_m: Mapping[str, float]) -> float:
    """Stature estimate as the mean of per-segment stature implied by the table.

    ``lengths_m`` maps segment names to measured joint-centre lengths. Useful
    when no upright frame exists in a capture. Precondition: at least one
    known segment with a positive length.
    """
    estimates = []
    for name, length in lengths_m.items():
        if name not in DE_LEVA_MALE:
            raise ValueError(f"Unknown segment {name}")
        _check_positive(length, f"length of {name}")
        estimates.append(REFERENCE_STATURE_M * length / DE_LEVA_MALE[name].length_m)
    if not estimates:
        raise ValueError("At least one segment length is required")
    return float(np.mean(estimates))


def mass_check(total_kg: float, stature_m: float, mass_kg: float) -> float:
    """Ratio of a model's total mass to the anthropometric body mass (1.0 ideal)."""
    _check_positive(total_kg, "total_kg")
    _check_positive(mass_kg, "mass_kg")
    _check_positive(stature_m, "stature_m")
    return total_kg / mass_kg


def inertia_about_axis(
    mass_kg: float, length_m: float, radii: tuple[float, float, float], axis: Array
) -> Array:
    """Inertia tensor ``m (k L)^2`` with the longitudinal radius along ``axis``.

    ``radii`` are (sagittal, transverse, longitudinal) fractions of the
    length; the two transverse principal axes are placed orthogonal to
    ``axis``. Precondition: positive mass and length, nonzero axis.
    Postcondition: symmetric positive definite tensor about the centre of mass.
    """
    _check_positive(mass_kg, "mass_kg")
    _check_positive(length_m, "length_m")
    a = np.asarray(axis, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12:
        raise ValueError("axis must be a finite nonzero 3-vector")
    a = a / np.linalg.norm(a)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, helper)
    x /= np.linalg.norm(x)
    y = np.cross(a, x)
    rot = np.column_stack([x, y, a])
    principal = np.diag([mass_kg * (k * length_m) ** 2 for k in radii])
    return np.asarray(rot @ principal @ rot.T, dtype=np.float64)


def de_leva_table_dict() -> dict[str, dict[str, Any]]:
    """Return DE_LEVA_MALE table as a serializable dictionary of segment parameters."""
    return {
        name: {
            "length_m": row.length_m,
            "mass_fraction": row.mass_fraction,
            "com_fraction": row.com_fraction,
            "radii": list(row.radii),
        }
        for name, row in sorted(DE_LEVA_MALE.items())
    }


def de_leva_table_sha256() -> str:
    """Canonical SHA-256 hash of the DE_LEVA_MALE table as JSON."""
    raw = json.dumps(de_leva_table_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
