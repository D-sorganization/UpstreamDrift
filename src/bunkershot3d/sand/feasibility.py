"""Grain-population feasibility precondition (issue #8610, defects B26/B29).

The canonical BunkerShot3D configuration asked for 50,000 grains of
d = 0.4 mm in a 0.4 x 0.3 x 0.1 m domain. That is::

    V_grain = (pi/6) d^3          = 3.351e-11 m^3
    V_solid = 50000 * V_grain     = 1.676e-6  m^3
    phi     = V_solid / 0.012     = 1.4e-4        (0.014 %)
    depth   = V_solid / (A * 0.6) = 2.33e-5 m     (0.023 mm)

-- a settled bed about one seventeenth of a single grain diameter deep. Every
"bunker shot" run from that configuration swung a club through an essentially
empty box, and nothing in the code said so. A real 100 mm USGA base at
phi = 0.60 needs **2.1e8** grains; even a 10 mm token bed needs 2.1e7.

This module turns that arithmetic into a precondition. Every check raises --
never asserts -- because ``python -O`` removes assertions and a guard that
vanishes under optimisation is not a guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bed import BunkerBedGeometry
from .exceptions import InfeasibleBedError

__all__ = [
    "DEFAULT_DEPTH_TOLERANCE",
    "MAX_PHYSICAL_SOLID_FRACTION",
    "BedFeasibilityReport",
    "achieved_solid_fraction",
    "evaluate_bed_feasibility",
    "grain_volume_m3",
    "require_feasible_bed",
    "required_grain_count",
    "settled_bed_depth_m",
]

MAX_PHYSICAL_SOLID_FRACTION = 0.64
"""Random close packing of equal spheres; no bed can exceed it."""

DEFAULT_DEPTH_TOLERANCE = 0.10
"""A configured bed may miss its target depth by 10% before it is refused."""


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise InfeasibleBedError(f"{name} must be finite, got {value!r}")
    return float(value)


def _require_positive_diameter(grain_diameter_m: float) -> float:
    diameter = _require_finite(grain_diameter_m, "grain diameter")
    if diameter <= 0.0:
        raise InfeasibleBedError(f"grain diameter must be positive, got {diameter!r} m")
    return diameter


def _require_physical_solid_fraction(solid_fraction: float) -> float:
    phi = _require_finite(solid_fraction, "solid fraction")
    if not 0.0 < phi <= MAX_PHYSICAL_SOLID_FRACTION:
        raise InfeasibleBedError(
            f"target solid fraction must lie in (0, {MAX_PHYSICAL_SOLID_FRACTION}], "
            f"got {phi!r}; random close packing of equal spheres is the ceiling"
        )
    return phi


def _require_positive_count(grain_count: int) -> int:
    count = int(grain_count)
    if count <= 0:
        raise InfeasibleBedError(f"grain count must be positive, got {count!r}")
    return count


def grain_volume_m3(grain_diameter_m: float) -> float:
    """Return the volume of one spherical grain.

    Raises:
        InfeasibleBedError: if the diameter is not a positive finite number.
    """
    diameter = _require_positive_diameter(grain_diameter_m)
    return (math.pi / 6.0) * diameter**3


def required_grain_count(
    bulk_volume_m3: float,
    solid_fraction: float,
    grain_diameter_m: float,
) -> int:
    """Return how many grains it takes to fill a bulk volume at a packing.

    Args:
        bulk_volume_m3: Bulk (total) volume of the bed, voids included.
        solid_fraction: Target solid volume fraction.
        grain_diameter_m: Monodisperse grain diameter.

    Returns:
        The grain count, rounded up.

    Raises:
        InfeasibleBedError: on non-physical arguments.
    """
    volume = _require_finite(bulk_volume_m3, "bulk volume")
    if volume <= 0.0:
        raise InfeasibleBedError(f"bulk volume must be positive, got {volume!r} m^3")
    phi = _require_physical_solid_fraction(solid_fraction)
    return math.ceil(volume * phi / grain_volume_m3(grain_diameter_m))


def achieved_solid_fraction(
    grain_count: int,
    grain_diameter_m: float,
    bulk_volume_m3: float,
) -> float:
    """Return the solid volume fraction a grain population actually delivers.

    Raises:
        InfeasibleBedError: on non-physical arguments.
    """
    count = _require_positive_count(grain_count)
    volume = _require_finite(bulk_volume_m3, "bulk volume")
    if volume <= 0.0:
        raise InfeasibleBedError(f"bulk volume must be positive, got {volume!r} m^3")
    return count * grain_volume_m3(grain_diameter_m) / volume


def settled_bed_depth_m(
    grain_count: int,
    grain_diameter_m: float,
    plan_area_m2: float,
    solid_fraction: float,
) -> float:
    """Return how deep a grain population settles over a given footprint.

    Raises:
        InfeasibleBedError: on non-physical arguments.
    """
    count = _require_positive_count(grain_count)
    area = _require_finite(plan_area_m2, "plan area")
    if area <= 0.0:
        raise InfeasibleBedError(f"plan area must be positive, got {area!r} m^2")
    phi = _require_physical_solid_fraction(solid_fraction)
    return count * grain_volume_m3(grain_diameter_m) / (area * phi)


@dataclass(frozen=True, slots=True)
class BedFeasibilityReport:
    """What a configured grain population actually produces."""

    grain_count: int
    grain_diameter_m: float
    target_solid_fraction: float
    target_depth_m: float
    settled_depth_m: float
    achieved_solid_fraction: float
    required_grain_count: int
    reasons: tuple[str, ...]

    @property
    def depth_ratio(self) -> float:
        """Settled depth divided by target depth."""
        return self.settled_depth_m / self.target_depth_m

    @property
    def is_feasible(self) -> bool:
        """True when no reason to refuse the configuration was found."""
        return not self.reasons

    def message(self) -> str:
        """Return an actionable refusal message."""
        header = (
            f"grain population is not a bunker bed: {self.grain_count:,} grains "
            f"of grain diameter {self.grain_diameter_m * 1e3:.4g} mm settle to a "
            f"depth of {self.settled_depth_m * 1e3:.4g} mm in a bed asked to be "
            f"{self.target_depth_m * 1e3:.4g} mm deep "
            f"({self.depth_ratio * 100:.3g}% of target), giving a solid fraction "
            f"of {self.achieved_solid_fraction:.3g} against a target of "
            f"{self.target_solid_fraction:.3g}."
        )
        remedies = (
            f"Remedies: configure {self.required_grain_count:,} grains at this "
            f"grain diameter; or coarse-grain the population (10x the grain "
            f"diameter cuts the count 1000x, but check the grain-diameter to "
            f"leading-edge-radius ratio before believing the result); or drop "
            f"to a continuum fidelity tier, which does not resolve grains at all."
        )
        return f"{header} " + " ".join(self.reasons) + f" {remedies}"


def evaluate_bed_feasibility(
    bed: BunkerBedGeometry,
    grain_count: int,
    grain_diameter_m: float,
    target_solid_fraction: float,
    depth_tolerance: float = DEFAULT_DEPTH_TOLERANCE,
    max_grain_count: int | None = None,
) -> BedFeasibilityReport:
    """Check whether a grain population can fill the requested bed.

    Args:
        bed: The bed the population is supposed to fill.
        grain_count: Configured number of grains.
        grain_diameter_m: Configured (monodisperse) grain diameter.
        target_solid_fraction: Packing the settled bed should reach.
        depth_tolerance: Allowed relative miss on the target depth.
        max_grain_count: Optional tractability ceiling. Supply the largest
            population the chosen backend can actually integrate.

    Returns:
        A report; ``is_feasible`` is False when any check failed.

    Raises:
        InfeasibleBedError: for arguments that are not merely infeasible but
            malformed -- a non-positive count or diameter, or a grain larger
            than the bed it is supposed to fill.
    """
    count = _require_positive_count(grain_count)
    diameter = _require_positive_diameter(grain_diameter_m)
    phi_target = _require_physical_solid_fraction(target_solid_fraction)
    tolerance = _require_finite(depth_tolerance, "depth tolerance")
    if not 0.0 <= tolerance < 1.0:
        raise InfeasibleBedError(
            f"depth tolerance must lie in [0, 1), got {tolerance!r}"
        )
    smallest_extent_m = min(bed.depth_m, bed.plan_length_m, bed.plan_width_m)
    if diameter >= smallest_extent_m:
        raise InfeasibleBedError(
            f"grain diameter {diameter:.4g} m is larger than the smallest bed "
            f"extent {smallest_extent_m:.4g} m; a single grain cannot fit in "
            "the domain it is supposed to fill"
        )

    settled_m = settled_bed_depth_m(
        grain_count=count,
        grain_diameter_m=diameter,
        plan_area_m2=bed.plan_area_m2,
        solid_fraction=phi_target,
    )
    phi_achieved = achieved_solid_fraction(
        grain_count=count,
        grain_diameter_m=diameter,
        bulk_volume_m3=bed.bulk_volume_m3,
    )
    needed = required_grain_count(
        bulk_volume_m3=bed.bulk_volume_m3,
        solid_fraction=phi_target,
        grain_diameter_m=diameter,
    )

    reasons: list[str] = []
    if phi_achieved > MAX_PHYSICAL_SOLID_FRACTION:
        reasons.append(
            f"The configured population exceeds random close packing: it "
            f"implies a solid fraction of {phi_achieved:.3g} against a physical "
            f"ceiling of {MAX_PHYSICAL_SOLID_FRACTION}."
        )
    elif settled_m < bed.depth_m * (1.0 - tolerance):
        reasons.append(
            f"The bed is under-filled by a factor of {bed.depth_m / settled_m:.4g}."
        )
    elif settled_m > bed.depth_m * (1.0 + tolerance):
        reasons.append(
            f"The bed is over-filled by a factor of {settled_m / bed.depth_m:.4g}."
        )
    if max_grain_count is not None and needed > max_grain_count:
        reasons.append(
            f"Filling this bed needs {needed:,} grains, beyond the tractable "
            f"ceiling of {max_grain_count:,} supplied by the caller."
        )
    return BedFeasibilityReport(
        grain_count=count,
        grain_diameter_m=diameter,
        target_solid_fraction=phi_target,
        target_depth_m=bed.depth_m,
        settled_depth_m=settled_m,
        achieved_solid_fraction=phi_achieved,
        required_grain_count=needed,
        reasons=tuple(reasons),
    )


def require_feasible_bed(
    bed: BunkerBedGeometry,
    grain_count: int,
    grain_diameter_m: float,
    target_solid_fraction: float,
    depth_tolerance: float = DEFAULT_DEPTH_TOLERANCE,
    max_grain_count: int | None = None,
) -> BedFeasibilityReport:
    """Refuse a grain population that cannot physically fill the bed.

    Returns:
        The feasibility report, when the configuration is acceptable.

    Raises:
        InfeasibleBedError: with an actionable message, when it is not.
    """
    report = evaluate_bed_feasibility(
        bed=bed,
        grain_count=grain_count,
        grain_diameter_m=grain_diameter_m,
        target_solid_fraction=target_solid_fraction,
        depth_tolerance=depth_tolerance,
        max_grain_count=max_grain_count,
    )
    if not report.is_feasible:
        raise InfeasibleBedError(report.message())
    return report
