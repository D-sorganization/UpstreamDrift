"""Cylinder or tapered-cone grip for the shared poser.

The appearance frame puts the shaft on +Z (butt toward the head). A tapered
preset uses the same station law as ``GripGeometry.radius_at``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.grip_contact.grip_geometry import (
    STATION_LOWER_M,
    STATION_UPPER_M,
    GripGeometry,
    get_grip_preset,
)


@dataclass(frozen=True)
class GripSolid:
    """Right circular grip: ``radius_at(z) = radius_m + slope_per_m * (z - z0_m)``.

    Preconditions: ``radius_m`` and ``length_m`` are finite and positive.
    ``radius_at`` may be non-positive outside a taper's measured stations;
    the poser rejects that instead of curling onto a vanished shaft.
    """

    radius_m: float
    length_m: float = 0.27
    z0_m: float = 0.0
    slope_per_m: float = 0.0

    def __post_init__(self) -> None:
        for name in ("radius_m", "length_m", "z0_m", "slope_per_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        if self.length_m <= 0.0:
            raise ValueError(f"length_m must be positive, got {self.length_m}")

    def radius_at(self, z_m: float) -> float:
        """Grip radius at axial station ``z_m`` (metres, butt toward the head)."""
        if isinstance(z_m, bool) or not isinstance(z_m, (int, float)):
            raise TypeError(f"z_m must be a number, got {z_m!r}")
        if not math.isfinite(float(z_m)):
            raise ValueError(f"z_m must be finite, got {z_m!r}")
        return float(self.radius_m + self.slope_per_m * (float(z_m) - self.z0_m))

    @classmethod
    def cylinder(cls, radius_m: float, length_m: float = 0.27) -> GripSolid:
        """Zero-taper cylinder of the given radius."""
        return cls(radius_m=radius_m, length_m=length_m)

    @classmethod
    def from_geometry(cls, geom: GripGeometry) -> GripSolid:
        """Match ``GripGeometry.radius_at``, including buildup and taper."""
        if not isinstance(geom, GripGeometry):
            raise TypeError(f"geom must be a GripGeometry, got {type(geom).__name__}")
        radius = 0.5 * geom.effective_d_upper_m
        if geom.taper == "zero":
            return cls.cylinder(radius, geom.length_m)
        lower = 0.5 * geom.effective_d_lower_m
        span = STATION_LOWER_M - STATION_UPPER_M
        return cls(
            radius_m=radius,
            length_m=geom.length_m,
            z0_m=STATION_UPPER_M,
            slope_per_m=(lower - radius) / span,
        )


def default_grip() -> GripSolid:
    """The legacy zero-taper cylinder (diameter 0.0254 m)."""
    return GripSolid.from_geometry(get_grip_preset())


__all__ = ["GripSolid", "default_grip"]
