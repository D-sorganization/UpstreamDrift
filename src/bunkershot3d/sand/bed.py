"""Bunker bed geometry (issue #8610).

USGA guidance is a sand depth of 4-6 in (100-150 mm) on bunker floors and
2-3 in (50-75 mm) on faces and slopes -- deliberately non-uniform, because a
deep face buries the ball. The trend with firmer, more angular sands is toward
8-10 in, so depth outside the recommended band is reported as an advisory
rather than rejected.

The surface is modelled as a plane tilted by ``surface_slope_rad`` about the
horizontal direction given by ``surface_azimuth_rad``. The stance slope is
carried separately: the ball can sit on a bank while the player stands
somewhere else entirely.

All lengths are metres and all angles radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .exceptions import BedGeometryError

__all__ = [
    "MAX_STABLE_SLOPE_RAD",
    "USGA_FACE_DEPTH_RANGE_M",
    "USGA_FLOOR_DEPTH_RANGE_M",
    "BedZone",
    "BunkerBedGeometry",
]

USGA_FLOOR_DEPTH_RANGE_M = (0.100, 0.150)
"""Recommended sand depth on a bunker floor (4-6 in)."""

USGA_FACE_DEPTH_RANGE_M = (0.050, 0.075)
"""Recommended sand depth on a bunker face or slope (2-3 in)."""

MAX_STABLE_SLOPE_RAD = math.radians(60.0)
"""Beyond this the sand cannot stand; a steeper face is a modelling error."""


class BedZone(StrEnum):
    """Which part of the bunker a bed patch represents."""

    FLOOR = "floor"
    FACE = "face"

    @property
    def usga_depth_range_m(self) -> tuple[float, float]:
        """Recommended sand-depth band for this zone."""
        if self is BedZone.FLOOR:
            return USGA_FLOOR_DEPTH_RANGE_M
        return USGA_FACE_DEPTH_RANGE_M


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise BedGeometryError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class BunkerBedGeometry:
    """A rectangular patch of bunker sand.

    Attributes:
        depth_m: Sand depth measured vertically.
        plan_length_m: Extent along x.
        plan_width_m: Extent along y.
        zone: Floor or face, which selects the USGA depth band.
        surface_slope_rad: Tilt of the sand surface from horizontal.
        surface_azimuth_rad: Horizontal direction of steepest ascent.
        stance_slope_rad: Slope the player stands on; signed, and independent
            of the surface slope at the ball.
    """

    depth_m: float
    plan_length_m: float
    plan_width_m: float
    zone: BedZone = BedZone.FLOOR
    surface_slope_rad: float = 0.0
    surface_azimuth_rad: float = 0.0
    stance_slope_rad: float = 0.0

    def __post_init__(self) -> None:
        depth = _require_finite(self.depth_m, "bed depth_m")
        length = _require_finite(self.plan_length_m, "plan_length_m")
        width = _require_finite(self.plan_width_m, "plan_width_m")
        surface = _require_finite(self.surface_slope_rad, "surface slope")
        azimuth = _require_finite(self.surface_azimuth_rad, "surface azimuth")
        stance = _require_finite(self.stance_slope_rad, "stance slope")
        if depth <= 0.0:
            raise BedGeometryError(f"bed depth_m must be positive, got {depth!r} m")
        if length <= 0.0 or width <= 0.0:
            raise BedGeometryError(
                "plan_length_m and plan_width_m must be positive, got "
                f"{length!r} m and {width!r} m"
            )
        for name, angle in (
            ("surface slope", surface),
            ("stance slope", stance),
        ):
            if abs(angle) > MAX_STABLE_SLOPE_RAD:
                raise BedGeometryError(
                    f"{name} of {math.degrees(angle):.1f} deg exceeds the "
                    f"{math.degrees(MAX_STABLE_SLOPE_RAD):.0f} deg at which "
                    "sand can still stand"
                )
        if abs(azimuth) > 2.0 * math.pi:
            raise BedGeometryError(
                f"surface azimuth must lie within +/- 2 pi rad, got {azimuth!r}"
            )

    # ------------------------------------------------------------ extents

    @property
    def plan_area_m2(self) -> float:
        """Footprint area of the modelled patch."""
        return self.plan_length_m * self.plan_width_m

    @property
    def bulk_volume_m3(self) -> float:
        """Bulk (total) sand volume of the patch, voids included."""
        return self.plan_area_m2 * self.depth_m

    @property
    def surface_slope_deg(self) -> float:
        """Surface slope in degrees."""
        return math.degrees(self.surface_slope_rad)

    @property
    def stance_slope_deg(self) -> float:
        """Stance slope in degrees."""
        return math.degrees(self.stance_slope_rad)

    # ---------------------------------------------------------- profiling

    def surface_height_m(self, x_m: Any, y_m: Any) -> Any:
        """Return the sand-surface height above the patch datum.

        Args:
            x_m: Position along x, scalar or array.
            y_m: Position along y, scalar or array.

        Returns:
            The surface height, matching the broadcast shape of the inputs.
        """
        gradient = math.tan(self.surface_slope_rad)
        along = np.cos(self.surface_azimuth_rad) * np.asarray(
            x_m, dtype=float
        ) + np.sin(self.surface_azimuth_rad) * np.asarray(y_m, dtype=float)
        height = gradient * along
        if np.ndim(height) == 0:
            return float(height)
        return height

    # ----------------------------------------------------------- advisory

    @property
    def usga_depth_range_m(self) -> tuple[float, float]:
        """Recommended sand-depth band for this bed's zone."""
        return self.zone.usga_depth_range_m

    @property
    def is_within_usga_depth(self) -> bool:
        """True when the depth sits inside the recommended band for the zone."""
        low, high = self.usga_depth_range_m
        return low <= self.depth_m <= high

    def depth_advisory(self) -> str:
        """Return a human-readable statement about the configured depth."""
        low, high = self.usga_depth_range_m
        band = f"{low * 1e3:.0f}-{high * 1e3:.0f} mm"
        actual = f"{self.depth_m * 1e3:.1f} mm"
        if self.is_within_usga_depth:
            return (
                f"{actual} of sand on a bunker {self.zone.value} is inside the "
                f"USGA recommended band of {band}."
            )
        direction = "shallower" if self.depth_m < low else "deeper"
        return (
            f"{actual} of sand on a bunker {self.zone.value} is {direction} "
            f"than the USGA recommended band of {band}. This is an advisory: "
            "the trend with firmer, more angular sands is toward 200-250 mm."
        )
