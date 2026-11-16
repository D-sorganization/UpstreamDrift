"""Tapered club grip signed-distance field (HRG-GB2)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_build.geometry import closest_points_segments
from src.shared.python.grip_contact.grip_geometry import GripGeometry

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ClubSDF:
    """Analytic tapered-grip distance field in the hand-grip frame (axis = +z)."""

    geometry: GripGeometry
    axial_offset_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.axial_offset_m):
            raise ValueError("axial_offset_m must be finite")

    def radius_at_z(self, z_m: float) -> float:
        """Grip radius at axial coordinate *z_m* (metres from hand frame origin)."""
        return self.geometry.radius_at(z_m + self.axial_offset_m)

    def signed_distance_point(self, point: np.ndarray) -> float:
        """Signed distance from a point to the grip solid (negative = inside)."""
        if not isinstance(point, np.ndarray) or point.shape != (3,):
            raise TypeError("point must be a (3,) ndarray")
        x, y, z = (float(point[0]), float(point[1]), float(point[2]))
        radial = math.hypot(x, y)
        return radial - self.radius_at_z(z)

    def signed_distance_capsule(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        radius_m: float,
        *,
        samples: int = 12,
    ) -> float:
        """Minimum signed distance from a capsule solid to the club grip."""
        if samples < 4:
            raise ValueError("samples must be >= 4")
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        ts = np.linspace(0.0, 1.0, samples)
        best = -math.inf
        for t in ts:
            centre = p0 + t * (p1 - p0)
            d = self.signed_distance_point(centre) - radius_m
            best = max(best, d)
        return best

    def axis_segment(self) -> tuple[np.ndarray, np.ndarray]:
        """Finite club axis segment used for flat-start alignment checks."""
        z0 = -0.05
        z1 = 0.12
        return (
            np.array([0.0, 0.0, z0], dtype=np.float64),
            np.array([0.0, 0.0, z1], dtype=np.float64),
        )

    def segment_clearance_capsule(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        radius_m: float,
        axis_p0: np.ndarray,
        axis_p1: np.ndarray,
        axis_radius_m: float,
    ) -> float:
        """Capsule-to-axis-cylinder clearance via segment closest points."""
        _ = axis_radius_m
        pa, pb = closest_points_segments(p0, p1, axis_p0, axis_p1)
        radial = float(np.linalg.norm(pa[:2] - pb[:2]))
        return radial - radius_m - self.radius_at_z(float(pa[2]))
