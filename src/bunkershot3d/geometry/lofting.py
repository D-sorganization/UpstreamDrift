"""Loft a verified watertight wedge mesh from the design vector (#8609).

Cross-sections are built in the Acushnet measurement plane at a series
of spanwise stations from heel to toe, then stitched into a closed
triangle mesh:

* **heel/toe relief** narrows the sole toward the ends, moving the
  trailing contact point forward - so each station re-solves its own
  sole profile;
* **heel-toe rocker** lifts the sole away from the ground toward the
  ends, integrated from a curvature field that matches the centre,
  heel and toe radii;
* **trailing relief** chamfers the rear corner of the flange;
* **face progression** shifts the whole head along the target axis.

Head frame: origin at the leading-edge point of the centre section
before progression, ``+x`` rearward, ``+y`` heel to toe, ``+z`` up.

The result is checked - manifold, closed, consistently wound outward,
Euler characteristic 2 - before it is returned, so downstream solvers
receive a mesh whose validity is a fact rather than an assumption.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .mass_properties import MassProperties, compute_mass_properties
from .mesh import TriangleMesh, require_watertight, signed_volume_m3
from .profile import (
    build_section_polygon,
    constructible_camber_range_m2,
    polygon_area_m2,
)
from .wedge import WedgeGeometry

__all__ = [
    "build_wedge_mesh",
    "rocker_offsets_m",
    "shaft_axis",
    "wedge_mass_properties",
]

_MIN_STATIONS = 5
_RELIEF_EXPONENT = 3.0
_MIN_TRAILING_RELIEF = 1e-6


def _relief_fractions(
    geometry: WedgeGeometry, span: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Sole-width relief at each station, 0 at the centre."""
    half = geometry.blade_length_m / 2.0
    normalised = span / half
    heel = geometry.heel_relief_fraction * np.clip(-normalised, 0.0, 1.0) ** 2
    toe = geometry.toe_relief_fraction * np.clip(normalised, 0.0, 1.0) ** 2
    return heel + toe


def rocker_offsets_m(
    geometry: WedgeGeometry, span: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Heel-toe rocker lift at each spanwise station.

    The rocker is integrated twice from a curvature field that equals
    ``1/centre_rocker_radius`` at the centre and blends to the heel and
    toe curvatures toward the ends, so the sole is tangent to the ground
    at the centre station and rises monotonically outboard.

    Note on the patent's numbers: US10143900B2 quotes heel < 30 mm, toe
    < 40 mm and centre > 70 mm, but those are *local* curvature
    measurements over a short arc.  Applied across a full 78 mm blade a
    75 mm centre radius would lift the ends by 11 mm, which no wedge
    does, so blade-scale radii belong in this field.
    """
    half = geometry.blade_length_m / 2.0
    fine = np.linspace(-half, half, 401)
    normalised = fine / half
    centre = 1.0 / geometry.centre_rocker_radius_m
    heel_extra = (1.0 / geometry.heel_rocker_radius_m - centre) * np.clip(
        -normalised, 0.0, 1.0
    ) ** _RELIEF_EXPONENT
    toe_extra = (1.0 / geometry.toe_rocker_radius_m - centre) * np.clip(
        normalised, 0.0, 1.0
    ) ** _RELIEF_EXPONENT
    curvature = centre + heel_extra + toe_extra

    slope = np.zeros_like(fine)
    height = np.zeros_like(fine)
    middle = len(fine) // 2
    step = fine[1] - fine[0]
    for index in range(middle + 1, len(fine)):
        slope[index] = slope[index - 1] + curvature[index] * step
        height[index] = height[index - 1] + slope[index] * step
    for index in range(middle - 1, -1, -1):
        slope[index] = slope[index + 1] - curvature[index] * step
        height[index] = height[index + 1] - slope[index] * step
    return np.interp(span, fine, height)


def _station_polygon(
    geometry: WedgeGeometry,
    station_m: float,
    relief: float,
    n_profile_points: int,
) -> NDArray[np.float64]:
    """Cross-section at one spanwise station, with relief applied."""
    width_m = geometry.sole_width_m * (1.0 - relief)
    scaled_camber = geometry.sole_camber_area_m2 * (1.0 - relief) ** 2
    low, high = constructible_camber_range_m2(
        geometry, sole_width_m=width_m, n_points=n_profile_points
    )
    # A narrower sole admits a narrower camber band; clamping keeps the
    # relieved section as close to the declared camber as its width can
    # carry, rather than silently changing the declared design.
    camber_m2 = min(max(scaled_camber, low), high)
    return build_section_polygon(
        geometry,
        n_points=n_profile_points,
        sole_width_m=width_m,
        camber_area_m2=camber_m2,
    )


def _chamfer_rear_corner(
    polygon: NDArray[np.float64], fraction: float
) -> NDArray[np.float64]:
    """Grind the rear corner of the flange (trailing-edge relief)."""
    if fraction <= _MIN_TRAILING_RELIEF:
        return polygon
    corner = polygon[-2]
    towards_sole = polygon[-3]
    towards_face = polygon[-1]
    first = corner + 0.5 * fraction * (towards_sole - corner)
    second = corner + 0.5 * fraction * (towards_face - corner)
    return np.vstack([polygon[:-2], first, second, polygon[-1:]])


def _apply_rocker(
    polygon: NDArray[np.float64], lift_m: float, top_z_m: float
) -> NDArray[np.float64]:
    """Lift the sole by ``lift_m``, leaving the topline where it is."""
    shifted = polygon.copy()
    weights = np.where(
        polygon[:, 1] <= 0.0,
        1.0,
        np.clip(1.0 - polygon[:, 1] / top_z_m, 0.0, 1.0),
    )
    shifted[:, 1] = polygon[:, 1] + lift_m * weights
    return shifted


def _stitch(n_stations: int, n_points: int) -> NDArray[np.int64]:
    """Triangulate the tube plus both end caps."""
    faces: list[tuple[int, int, int]] = []
    for station in range(n_stations - 1):
        base = station * n_points
        nxt = base + n_points
        for index in range(n_points):
            following = (index + 1) % n_points
            faces.append((base + index, nxt + index, nxt + following))
            faces.append((base + index, nxt + following, base + following))
    heel_centre = n_stations * n_points
    toe_centre = heel_centre + 1
    last = (n_stations - 1) * n_points
    for index in range(n_points):
        following = (index + 1) % n_points
        faces.append((heel_centre, index, following))
        faces.append((toe_centre, last + following, last + index))
    return np.asarray(faces, dtype=np.int64)


def build_wedge_mesh(
    geometry: WedgeGeometry,
    *,
    n_profile_points: int = 40,
    n_stations: int = 17,
) -> TriangleMesh:
    """Loft a watertight triangle mesh for a wedge head.

    Args:
        geometry: The wedge design vector.
        n_profile_points: Sole samples per cross-section.
        n_stations: Cross-sections from heel to toe.

    Returns:
        A mesh that has passed :func:`~.mesh.require_watertight`.

    Raises:
        ValueError: If the resolution is too coarse to represent the
            sole, or a station's geometry is not constructible.
        MeshValidationError: If the lofted mesh fails a solid-mesh
            precondition - which is a bug in this module, not a user
            error, and is surfaced rather than swallowed.
    """
    if not isinstance(geometry, WedgeGeometry):
        raise TypeError(f"expected a WedgeGeometry, got {type(geometry).__name__}")
    if n_stations < _MIN_STATIONS:
        raise ValueError(
            f"n_stations must be at least {_MIN_STATIONS}, got {n_stations}"
        )

    half = geometry.blade_length_m / 2.0
    span = np.linspace(-half, half, n_stations, dtype=np.float64)
    reliefs = _relief_fractions(geometry, span)
    lifts = rocker_offsets_m(geometry, span)
    top_z_m = geometry.face_height_m * math.cos(geometry.loft_rad)

    sections: list[NDArray[np.float64]] = []
    for station, relief, lift in zip(span, reliefs, lifts, strict=True):
        polygon = _station_polygon(
            geometry, float(station), float(relief), n_profile_points
        )
        polygon = _chamfer_rear_corner(polygon, geometry.trailing_relief_fraction)
        sections.append(_apply_rocker(polygon, float(lift), top_z_m))

    n_points = sections[0].shape[0]
    vertices = np.zeros((n_stations * n_points + 2, 3))
    for index, (station, polygon) in enumerate(zip(span, sections, strict=True)):
        block = slice(index * n_points, (index + 1) * n_points)
        vertices[block, 0] = polygon[:, 0] + geometry.face_progression_m
        vertices[block, 1] = station
        vertices[block, 2] = polygon[:, 1]
    vertices[-2] = (
        float(sections[0][:, 0].mean()) + geometry.face_progression_m,
        -half,
        float(sections[0][:, 1].mean()),
    )
    vertices[-1] = (
        float(sections[-1][:, 0].mean()) + geometry.face_progression_m,
        half,
        float(sections[-1][:, 1].mean()),
    )

    faces = _stitch(n_stations, n_points)
    mesh = TriangleMesh(vertices, faces)
    if signed_volume_m3(mesh) < 0.0:
        mesh = TriangleMesh(vertices, faces[:, ::-1])
    require_watertight(mesh, context="lofted wedge head")
    return mesh


def shaft_axis(
    geometry: WedgeGeometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The shaft axis in head coordinates as ``(point, unit direction)``.

    The axis passes through the heel end of the leading edge and rises at
    the design lie angle.  Moment of inertia about this axis is the
    quantity that resists the sand torquing the face open between entry
    and the ball.
    """
    point = np.array([geometry.face_progression_m, -geometry.blade_length_m / 2.0, 0.0])
    direction = np.array([0.0, -math.cos(geometry.lie_rad), math.sin(geometry.lie_rad)])
    return point, direction / float(np.linalg.norm(direction))


def wedge_mass_properties(
    geometry: WedgeGeometry,
    *,
    n_profile_points: int = 40,
    n_stations: int = 17,
) -> MassProperties:
    """Mass properties of the lofted head at its declared head mass."""
    mesh = build_wedge_mesh(
        geometry, n_profile_points=n_profile_points, n_stations=n_stations
    )
    return compute_mass_properties(mesh, mass_kg=geometry.head_mass_kg)


def section_area_m2(geometry: WedgeGeometry, *, n_profile_points: int = 40) -> float:
    """Cross-sectional area of the centre station, for quick sanity checks."""
    return polygon_area_m2(build_section_polygon(geometry, n_points=n_profile_points))
