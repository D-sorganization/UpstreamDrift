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

A narrow sole geometrically cannot host an arbitrarily large camber
area, so a station's camber is fitted to the band its width admits.
That fit is correct, but before issue #8698 it was invisible: a caller
declared 48 mm^2, received something else, and had no way to find out.
:func:`loft_wedge` therefore returns a :class:`LoftedWedge` carrying the
effective area and a per-station account, and :class:`CamberFit` decides
whether an inconstructible *declaration* is refused (the default) or
fitted on request.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .delivery import TRAVEL_AXIS_BODY
from .mass_properties import MassProperties, compute_mass_properties
from .mesh import (
    MeshValidationError,
    TriangleMesh,
    require_watertight,
    signed_volume_m3,
)
from .profile import (
    InconstructibleCamberError,
    build_section_polygon,
    constructible_camber_range_m2,
    polygon_area_m2,
)
from .wedge import WedgeGeometry

__all__ = [
    "CamberFit",
    "LoftedWedge",
    "StationCamber",
    "build_wedge_mesh",
    "loft_wedge",
    "require_travel_frame",
    "rocker_offsets_m",
    "shaft_axis",
    "wedge_mass_properties",
]

_MIN_STATIONS = 5
_RELIEF_EXPONENT = 3.0
_MIN_TRAILING_RELIEF = 1e-6
_SOLE_SPAN_SLACK = 0.05
"""Slack on the sole span, for the rocker lift and the trailing chamfer."""


class CamberFit(Enum):
    """What to do with a declared camber area the sole cannot carry.

    Attributes:
        STRICT: Refuse - raise :class:`~.profile.InconstructibleCamberError`.
            The default, matching :func:`~.profile.build_sole_profile` and
            :meth:`bunkershot3d.study.design_space.DesignSpace.sample`, both
            of which refuse rather than quietly deliver something adjacent.
        NEAREST: Substitute the nearest constructible area, and record the
            substitution on the returned :class:`LoftedWedge`.  This is what
            a design sweep wants - it keeps every point evaluable - but it
            has to be asked for, and the effective value comes back with it.
    """

    STRICT = "strict"
    NEAREST = "nearest"


@dataclass(frozen=True, slots=True)
class StationCamber:
    """The camber area one spanwise station asked for, and the one it got.

    Heel and toe relief narrow the sole toward the ends, and a narrower sole
    admits a narrower camber band, so the relief-scaled request can fall
    outside what that station can carry.  That request is *derived* by this
    module rather than declared by the caller, so it is always fitted to the
    nearest constructible area - but never silently: it is recorded here.

    Attributes:
        station_m: Spanwise position, heel negative, toe positive.
        sole_width_m: The relieved sole width at this station.
        requested_camber_area_m2: The relief-scaled request.
        effective_camber_area_m2: What the section was actually built with.
        constructible_camber_range_m2: ``(low, high)`` this width admits.
    """

    station_m: float
    sole_width_m: float
    requested_camber_area_m2: float
    effective_camber_area_m2: float
    constructible_camber_range_m2: tuple[float, float]

    @property
    def was_clamped(self) -> bool:
        """Whether this station got something other than it asked for."""
        return self.effective_camber_area_m2 != self.requested_camber_area_m2

    @property
    def camber_substitution_m2(self) -> float:
        """Effective minus requested; zero when nothing was substituted."""
        return self.effective_camber_area_m2 - self.requested_camber_area_m2


@dataclass(frozen=True, eq=False)
class LoftedWedge:
    """A lofted head, plus what its sole actually realised (issue #8698).

    :func:`build_wedge_mesh` returns only the mesh, so a caller that declared
    a camber area had no way to learn that a different one was built.  This
    is the object that closes that gap: :attr:`effective_camber_area_m2` is
    the number a design study should log next to the declared one.

    **Substitution happens at two scopes, and they do not imply each other.**
    The *aggregate* scope is the declared camber against the band the declared
    (unrelieved) sole width admits.  The *station* scope is each spanwise
    section's relief-scaled request against the band its own narrowed width
    admits.  A declaration can sit comfortably inside the aggregate band while
    heel and toe stations - which are narrower, and so admit a narrower band -
    are refitted.  The shipped ``sm9_58_m`` preset is exactly that case: 42.00
    mm^2 declared, 42.00 mm^2 effective, inside a (38.70, 42.44) mm^2 band,
    and three of its seventeen stations refitted anyway.

    So there is no single "was it clamped" answer, and this class refuses to
    pretend there is.  Read :attr:`any_camber_was_clamped` for "did the lofter
    build something other than what was asked for, anywhere", which is what a
    caller checking one boolean almost always means.  Read
    :attr:`aggregate_camber_was_clamped` only for the narrow declared-versus-
    effective question, and :attr:`clamped_stations` for the detail.

    Attributes:
        mesh: The verified watertight head.
        geometry: The design vector it was lofted from.
        camber_fit: The policy that was in force.
        constructible_camber_range_m2: ``(low, high)`` the *declared* sole
            width admits, at the resolution the head was lofted at.
        effective_camber_area_m2: The camber area at the declared sole
            width - equal to the declared value unless it was substituted.
        stations: Per-station records, heel to toe.
    """

    mesh: TriangleMesh
    geometry: WedgeGeometry
    camber_fit: CamberFit
    constructible_camber_range_m2: tuple[float, float]
    effective_camber_area_m2: float
    stations: tuple[StationCamber, ...]

    @property
    def declared_camber_area_m2(self) -> float:
        """The camber area the design vector declared."""
        return self.geometry.sole_camber_area_m2

    @property
    def aggregate_camber_was_clamped(self) -> bool:
        """Whether the *declared* camber area itself was substituted.

        Scoped deliberately narrowly: this compares
        :attr:`effective_camber_area_m2` against
        :attr:`declared_camber_area_m2` at the declared sole width, and says
        nothing about the relieved stations.  It can be ``False`` while
        :attr:`clamped_stations` is non-empty.

        Use this only when the declared number is the subject - reporting
        "you asked for X and got Y", for instance.  For "was anything
        substituted at all", use :attr:`any_camber_was_clamped`.
        """
        return self.effective_camber_area_m2 != self.declared_camber_area_m2

    @property
    def any_camber_was_clamped(self) -> bool:
        """Whether *any* camber substitution occurred, at any scope.

        True when the declared area was substituted, or when any spanwise
        station was refitted to its own constructible band.  This is the
        honest answer to "is the sole I got the sole I asked for", and it is
        the flag a caller should reach for by default: it cannot read
        ``False`` while :attr:`clamped_stations` holds anything.
        """
        return self.aggregate_camber_was_clamped or bool(self.clamped_stations)

    @property
    def aggregate_camber_substitution_m2(self) -> float:
        """Effective minus declared; zero when the declaration was honoured.

        Aggregate-scoped, matching :attr:`aggregate_camber_was_clamped` - a
        station's own substitution is
        :attr:`StationCamber.camber_substitution_m2`.
        """
        return self.effective_camber_area_m2 - self.declared_camber_area_m2

    @property
    def clamped_stations(self) -> tuple[StationCamber, ...]:
        """Every station that could not carry what it asked for."""
        return tuple(station for station in self.stations if station.was_clamped)


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


def _fit_station_camber(
    geometry: WedgeGeometry,
    *,
    station_m: float,
    relief: float,
    n_profile_points: int,
    declared_range_m2: tuple[float, float],
) -> StationCamber:
    """Decide the camber area one spanwise station will actually be built with.

    A narrower sole admits a narrower camber band; fitting keeps the relieved
    section as close to the declared camber as its width can carry, rather
    than emitting an inconstructible section.  The substitution is *recorded*
    on the returned record rather than discarded, which is the whole point of
    issue #8698.

    Args:
        geometry: The design vector.
        station_m: Spanwise position of this station.
        relief: Sole-width relief fraction at this station, 0 at the centre.
        n_profile_points: Sole samples per cross-section.
        declared_range_m2: The band the unrelieved sole width admits, already
            computed by the caller and reused when ``relief`` is zero.

    Returns:
        The station's request, the band, and what it will be built with.
    """
    width_m = geometry.sole_width_m * (1.0 - relief)
    requested_m2 = geometry.sole_camber_area_m2 * (1.0 - relief) ** 2
    low, high = (
        declared_range_m2
        if relief == 0.0
        else constructible_camber_range_m2(
            geometry, sole_width_m=width_m, n_points=n_profile_points
        )
    )
    return StationCamber(
        station_m=station_m,
        sole_width_m=width_m,
        requested_camber_area_m2=requested_m2,
        effective_camber_area_m2=min(max(requested_m2, low), high),
        constructible_camber_range_m2=(low, high),
    )


def _station_polygon(
    geometry: WedgeGeometry, fit: StationCamber, n_profile_points: int
) -> NDArray[np.float64]:
    """Cross-section at one spanwise station, with relief applied."""
    return build_section_polygon(
        geometry,
        n_points=n_profile_points,
        sole_width_m=fit.sole_width_m,
        camber_area_m2=fit.effective_camber_area_m2,
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


def _require_constructible_declaration(
    geometry: WedgeGeometry,
    *,
    camber_fit: CamberFit,
    n_profile_points: int,
) -> tuple[tuple[float, float], float]:
    """Resolve the declared camber against the band its sole width admits.

    Args:
        geometry: The wedge design vector.
        camber_fit: The policy in force.
        n_profile_points: Sole samples per cross-section.

    Returns:
        The band ``(low, high)`` and the effective camber area.

    Raises:
        InconstructibleCamberError: Under :attr:`CamberFit.STRICT`, when the
            declared area falls outside the band.
    """
    band = constructible_camber_range_m2(geometry, n_points=n_profile_points)
    low, high = band
    declared_m2 = geometry.sole_camber_area_m2
    effective_m2 = min(max(declared_m2, low), high)
    if camber_fit is CamberFit.NEAREST or effective_m2 == declared_m2:
        return band, effective_m2
    raise InconstructibleCamberError(
        f"the declared sole camber area of {declared_m2 * 1e6:.4g} mm^2 is "
        f"outside the {low * 1e6:.4g} to {high * 1e6:.4g} mm^2 band a convex "
        f"monotone {geometry.sole_width_m * 1e3:.4g} mm sole admits at "
        f"{geometry.geometric_bounce.angle_deg:.2f} deg of geometric bounce, "
        f"so the lofted head would carry {effective_m2 * 1e6:.4g} mm^2 "
        "instead. Change the declared camber, widen the sole, or pass "
        "camber_fit=CamberFit.NEAREST to accept the nearest constructible "
        "area - which is then reported as LoftedWedge.effective_camber_area_m2",
        requested_camber_area_m2=declared_m2,
        sole_width_m=geometry.sole_width_m,
        constructible_range_m2=band,
    )


def _stitch_sections(
    geometry: WedgeGeometry,
    span: NDArray[np.float64],
    sections: list[NDArray[np.float64]],
) -> TriangleMesh:
    """Place the cross-sections in the head frame and close them into a solid.

    Args:
        geometry: The wedge design vector, supplying the face progression.
        span: Spanwise station positions, heel to toe.
        sections: One planar polygon per station, all the same length.

    Returns:
        A mesh that has passed :func:`~.mesh.require_watertight`.

    Raises:
        MeshValidationError: If the stitched mesh is not a valid solid.
    """
    half = float(span[-1])
    n_stations = len(sections)
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


def loft_wedge(
    geometry: WedgeGeometry,
    *,
    n_profile_points: int = 40,
    n_stations: int = 17,
    camber_fit: CamberFit = CamberFit.STRICT,
) -> LoftedWedge:
    """Loft a wedge head and report what its sole actually realised.

    This is :func:`build_wedge_mesh` with the bookkeeping kept: the mesh is
    identical, and the returned record additionally carries the effective
    camber area, the constructible band, and a per-station account of every
    substitution the relief forced (issue #8698).

    Args:
        geometry: The wedge design vector.
        n_profile_points: Sole samples per cross-section.
        n_stations: Cross-sections from heel to toe.
        camber_fit: What to do when the declared camber area is outside the
            band the declared sole width admits.  Defaults to
            :attr:`CamberFit.STRICT`, which refuses.

    Returns:
        The lofted head and its camber account.

    Raises:
        TypeError: If ``geometry`` or ``camber_fit`` is the wrong type.
        ValueError: If the resolution is too coarse to represent the sole,
            or a station's geometry is not constructible.
        InconstructibleCamberError: Under :attr:`CamberFit.STRICT`, if the
            declared camber area is outside the constructible band.
        MeshValidationError: If the lofted mesh fails a solid-mesh
            precondition - which is a bug in this module, not a user
            error, and is surfaced rather than swallowed.
    """
    if not isinstance(geometry, WedgeGeometry):
        raise TypeError(f"expected a WedgeGeometry, got {type(geometry).__name__}")
    if not isinstance(camber_fit, CamberFit):
        raise TypeError(
            f"camber_fit must be a CamberFit, got {type(camber_fit).__name__}"
        )
    if n_stations < _MIN_STATIONS:
        raise ValueError(
            f"n_stations must be at least {_MIN_STATIONS}, got {n_stations}"
        )

    band_m2, effective_m2 = _require_constructible_declaration(
        geometry, camber_fit=camber_fit, n_profile_points=n_profile_points
    )

    half = geometry.blade_length_m / 2.0
    span = np.linspace(-half, half, n_stations, dtype=np.float64)
    reliefs = _relief_fractions(geometry, span)
    lifts = rocker_offsets_m(geometry, span)
    top_z_m = geometry.face_height_m * math.cos(geometry.loft_rad)

    sections: list[NDArray[np.float64]] = []
    fits: list[StationCamber] = []
    for station, relief, lift in zip(span, reliefs, lifts, strict=True):
        fit = _fit_station_camber(
            geometry,
            station_m=float(station),
            relief=float(relief),
            n_profile_points=n_profile_points,
            declared_range_m2=band_m2,
        )
        fits.append(fit)
        polygon = _station_polygon(geometry, fit, n_profile_points)
        polygon = _chamfer_rear_corner(polygon, geometry.trailing_relief_fraction)
        sections.append(_apply_rocker(polygon, float(lift), top_z_m))

    mesh = _stitch_sections(geometry, span, sections)
    require_travel_frame(mesh, geometry)
    return LoftedWedge(
        mesh=mesh,
        geometry=geometry,
        camber_fit=camber_fit,
        constructible_camber_range_m2=band_m2,
        effective_camber_area_m2=effective_m2,
        stations=tuple(fits),
    )


def require_travel_frame(mesh: TriangleMesh, geometry: WedgeGeometry) -> None:
    """Refuse a mesh that is mirrored against the travel axis.

    The head frame and :data:`~.delivery.TRAVEL_AXIS_BODY` are two halves of
    one convention, and until issue #9247 nothing held them together.  The
    sections are built with the leading-edge point at ``x =
    face_progression_m`` and the sole descending rearward from it to the
    trailing contact, which is precisely what makes ``-x`` the direction a
    wedge travels.  Reverse the sections and that constant is silently wrong:
    the head is then driven trailing edge first down a sole descending into
    the bed, on which more bounce is a steeper ramp and digs *deeper*.  That
    inverted the tool's primary design variable, and it survived a suite of
    2,000 tests because the two conventions were stated only in prose, in two
    docstrings, that disagreed.

    So the contract is checked on the artefact: the lowest point of a lofted
    head lies on the sole, rearward of the leading edge and no further back
    than the sole is wide.  Note that comparing the mesh's own extremes would
    not do -- an ``x`` mirror maps extremes to extremes, and a check on those
    passes either way round.  It is the leading edge's *forward-referenced*
    station that breaks the symmetry.

    Checked rather than asserted: ``python -O`` strips assertions, and every
    delivered pose and entry velocity in the package is expressed against
    this frame.

    Args:
        mesh: The lofted head.
        geometry: The design vector it was lofted from, supplying the leading
            edge's station and the sole width.

    Raises:
        MeshValidationError: If the sole's lowest point is not rearward of the
            leading edge and within the sole width of it.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lowest_x = float(vertices[int(np.argmin(vertices[:, 2])), 0])
    rearward = -float(TRAVEL_AXIS_BODY[0])
    along_m = (lowest_x - geometry.face_progression_m) * rearward
    if not 0.0 <= along_m <= geometry.sole_width_m * (1.0 + _SOLE_SPAN_SLACK):
        raise MeshValidationError(
            "the lofted sole must descend rearward from the leading edge: the "
            f"leading edge is at x = {geometry.face_progression_m * 1e3:.4g} mm "
            f"and the lowest point at x = {lowest_x * 1e3:.4g} mm, which is "
            f"{along_m * 1e3:.4g} mm rearward against a {geometry.sole_width_m * 1e3:.4g} "
            f"mm sole. With a travel axis of {TRAVEL_AXIS_BODY} the head would "
            "be driven trailing edge first down a sole descending into the "
            "bed. Either the sections are mirrored or TRAVEL_AXIS_BODY is "
            "(issue #9247)."
        )


def build_wedge_mesh(
    geometry: WedgeGeometry,
    *,
    n_profile_points: int = 40,
    n_stations: int = 17,
    camber_fit: CamberFit = CamberFit.STRICT,
) -> TriangleMesh:
    """Loft a watertight triangle mesh for a wedge head.

    A convenience wrapper over :func:`loft_wedge` for callers that want only
    the solid.  Anything that needs to know whether the sole it got is the
    sole it declared should call :func:`loft_wedge` instead and read
    :attr:`LoftedWedge.effective_camber_area_m2`.

    Args:
        geometry: The wedge design vector.
        n_profile_points: Sole samples per cross-section.
        n_stations: Cross-sections from heel to toe.
        camber_fit: What to do when the declared camber area is outside the
            constructible band; see :class:`CamberFit`.

    Returns:
        A mesh that has passed :func:`~.mesh.require_watertight`.

    Raises:
        ValueError: If the resolution is too coarse to represent the
            sole, or a station's geometry is not constructible.
        InconstructibleCamberError: Under :attr:`CamberFit.STRICT`, if the
            declared camber area is outside the constructible band.
        MeshValidationError: If the lofted mesh fails a solid-mesh
            precondition - which is a bug in this module, not a user
            error, and is surfaced rather than swallowed.
    """
    return loft_wedge(
        geometry,
        n_profile_points=n_profile_points,
        n_stations=n_stations,
        camber_fit=camber_fit,
    ).mesh


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
    camber_fit: CamberFit = CamberFit.STRICT,
) -> MassProperties:
    """Mass properties of the lofted head at its declared head mass."""
    mesh = build_wedge_mesh(
        geometry,
        n_profile_points=n_profile_points,
        n_stations=n_stations,
        camber_fit=camber_fit,
    )
    return compute_mass_properties(mesh, mass_kg=geometry.head_mass_kg)


def section_area_m2(geometry: WedgeGeometry, *, n_profile_points: int = 40) -> float:
    """Cross-sectional area of the centre station, for quick sanity checks."""
    return polygon_area_m2(build_section_polygon(geometry, n_points=n_profile_points))
