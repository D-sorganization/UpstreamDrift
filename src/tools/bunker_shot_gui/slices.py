"""Cutting planes through the impact zone (issue #8711).

Headless. This module imports numpy and nothing else from the drawing or
GUI stacks, so a slice can be taken in a test, in a batch sweep, or by
the Qt workbench.

The question this answers
-------------------------

*Does the velocity of the sand near the face change through the impact
zone, and how does flow ahead of the leading edge differ from flow along
the face?*  A cut through the sand with the velocity on it is the direct
answer, provided two things are true of the picture:

* the **direction** is drawn, not only the magnitude.  Sand pushed ahead
  of the sole and sand riding up the face can carry identical speeds; a
  magnitude-only heatmap hides exactly the distinction the question is
  about;
* the **club is in the frame**, so "ahead of the sole" and "up the face"
  are locations rather than guesses.  That is why the field carries
  :attr:`~bunkershot3d.fields.schema.SandFieldSeries.body_outline_m`.

What a cut through a plane-strain field actually is
---------------------------------------------------

F1 solves one plane.  It has no heel-to-toe direction at all -- that is
:attr:`~bunkershot3d.solvers.envelope.Caveat.PLANE_STRAIN_NO_OUT_OF_PLANE`
and :attr:`~bunkershot3d.solvers.mpm.envelope.RefusedQuantity.OUT_OF_PLANE`.
So the heel-to-toe series that issue #8711 asks for cannot be a series of
independent solves, and pretending otherwise would be the fabrication the
epic was written against.

:class:`SliceFidelity` makes the difference visible instead of hiding it:

* ``SOLVED`` -- the cut is the plane the tier solved.  The picture is the
  solution.
* ``EXTRUDED`` -- the cut is parallel to it but offset out of plane.  The
  numbers are identical **by assumption**, not by result, and the frames
  say so.  A test asserts that two heel-to-toe stations of an F1 field
  are bit-for-bit equal, because that is what plane strain means.
* ``PROJECTED`` -- the cut is oblique.  Samples are taken by projecting
  onto the solved plane, so the along-cut axis is compressed by
  ``cos(obliquity)``, and the component through the cut is in-plane
  velocity resolved rather than measured out-of-plane flow.

The offset is bounded by the solver's declared ``effective_width_m``: a
station further out than the slab F1 declares is refused rather than
extruded into territory the tier never claimed.

Colour scaling
--------------

:class:`SliceScale` is built by :meth:`SliceScale.covering` over every
frame of every field being compared, and injected into the renderer.
Nothing in this package infers a limit from the frame in front of it --
issue #8728 fixed exactly that bug for the sole-load ramp, and a
cross-section view that auto-scaled per frame would reintroduce it in the
place where it matters most, since the whole point is watching a
magnitude change through impact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from bunkershot3d.fields.schema import (
    DENSITY_UNIT,
    SHEAR_RATE_UNIT,
    VELOCITY_UNIT,
    FieldLayout,
    SandFieldSeries,
)

__all__ = [
    "DENSITY_COLORMAP",
    "SHEAR_COLORMAP",
    "SPEED_COLORMAP",
    "CursorMap",
    "CuttingPlane",
    "PlanePreset",
    "SliceFidelity",
    "SliceSample",
    "SliceScale",
    "body_focus_bounds_m",
    "face_normal_plane",
    "heel_to_toe_series",
    "preset_planes",
    "sample_plane",
    "slice_scale",
    "swing_plane",
]

SPEED_COLORMAP = "magma"
DENSITY_COLORMAP = "YlOrBr"
SHEAR_COLORMAP = "cividis"
"""One ramp per quantity, fixed, so a colour means a quantity.

``YlOrBr`` is the sand ramp the rest of the package already uses for
material; ``magma`` and ``cividis`` are perceptually uniform and
colour-vision-deficiency safe, which matters because these three panels
sit side by side."""

_TOL = 1.0e-9
_MIN_OBLIQUITY_COS = 1.0e-3
_DEFAULT_SAMPLES = 160
_UP = np.array([0.0, 0.0, 1.0])
_ALONG_PATH = np.array([1.0, 0.0, 0.0])
_HEEL_TO_TOE = np.array([0.0, 1.0, 0.0])


class SliceFidelity(StrEnum):
    """What a cut through this tier's field actually is."""

    SOLVED = "solved"
    """The cut plane is the plane the tier solved. The picture is it."""

    EXTRUDED = "extruded"
    """Parallel to the solved plane, offset out of it.

    The numbers are the solved plane's, repeated. For a plane-strain
    tier that repetition is the model's assumption, not a result."""

    PROJECTED = "projected"
    """Oblique to the solved plane.

    Samples are taken by projecting the cut onto the solved plane, so
    the along-cut axis is compressed by ``cos(obliquity)``."""

    @property
    def label(self) -> str:
        """Short human label."""
        return _FIDELITY_LABELS[self]


_FIDELITY_LABELS = {
    SliceFidelity.SOLVED: "solved plane",
    SliceFidelity.EXTRUDED: "extruded from the solved plane",
    SliceFidelity.PROJECTED: "projected onto the solved plane",
}


class PlanePreset(StrEnum):
    """The cuts a wedge shot is actually discussed in terms of."""

    SWING_PLANE = "swing_plane"
    """The plane containing the club path and the vertical.

    For F1 this *is* the solved plane, which is why it is the default."""

    FACE_NORMAL = "face_normal"
    """The plane containing the face normal and the vertical.

    Coincides with the swing plane for a square face; an open face
    rotates it about the vertical by the face-open angle, which makes it
    oblique to the solved plane and therefore ``PROJECTED``."""

    HEEL_TO_TOE = "heel_to_toe"
    """A series of planes parallel to the swing plane, heel to toe."""

    @property
    def label(self) -> str:
        """Short human label."""
        return _PRESET_LABELS[self]

    @property
    def description(self) -> str:
        """One line on what the cut shows."""
        return _PRESET_DESCRIPTIONS[self]


_PRESET_LABELS = {
    PlanePreset.SWING_PLANE: "Swing plane",
    PlanePreset.FACE_NORMAL: "Face-normal plane",
    PlanePreset.HEEL_TO_TOE: "Heel-to-toe series",
}

_PRESET_DESCRIPTIONS = {
    PlanePreset.SWING_PLANE: (
        "along the club path, vertical: sand ahead of the leading edge and "
        "sand riding up the face"
    ),
    PlanePreset.FACE_NORMAL: (
        "normal to the face: what the face itself is pushing, which parts "
        "from the swing plane once the face is open"
    ),
    PlanePreset.HEEL_TO_TOE: (
        "stations across the sole; a plane-strain tier repeats one solution "
        "at every station rather than solving them"
    ),
}


@dataclass(frozen=True)
class CuttingPlane:
    """A plane through the impact zone, and its own 2-D frame.

    A point of the cut at slice coordinates ``(s, h)`` is the world point
    ``origin_m + s * along + h * up``.  Storing the frame rather than
    only a normal means the picture's axes have a stated meaning: ``s``
    runs along the cut and ``h`` runs up it, in metres, always.

    Attributes:
        name: What to call it in a frame stamp.
        origin_m: ``(3,)`` world point at ``(s, h) = (0, 0)``.
        along: ``(3,)`` unit vector, the in-cut horizontal axis.
        up: ``(3,)`` unit vector, the in-cut vertical axis.
        preset: Which named preset produced it, or ``None`` for an
            arbitrary placement.
    """

    name: str
    origin_m: NDArray[np.float64]
    along: NDArray[np.float64]
    up: NDArray[np.float64]
    preset: PlanePreset | None = None

    def __post_init__(self) -> None:
        origin = _vector(self.origin_m, "origin_m")
        along = _unit(self.along, "along")
        up = _unit(self.up, "up")
        if abs(float(along @ up)) > 1.0e-6:
            raise ValueError(
                f"along and up must be perpendicular, got a dot product of "
                f"{float(along @ up):.3g}"
            )
        if not str(self.name).strip():
            raise ValueError("a cutting plane must be named for its frame stamp")
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "along", along)
        object.__setattr__(self, "up", up)

    @property
    def normal(self) -> NDArray[np.float64]:
        """``(3,)`` unit normal of the cut, ``along x up``."""
        return np.cross(self.along, self.up).astype(np.float64)

    @property
    def obliquity_deg(self) -> float:
        """Angle between the cut's along-axis and the club path.

        Zero means the cut is parallel to the swing plane; anything else
        means a sample of a plane-strain field is a projection.
        """
        cosine = float(np.clip(abs(self.along @ _ALONG_PATH), 0.0, 1.0))
        return math.degrees(math.acos(cosine))

    @property
    def offset_m(self) -> float:
        """Heel-to-toe offset of the cut's origin from the solved plane."""
        return float(self.origin_m @ _HEEL_TO_TOE)

    def world_m(
        self, along_m: NDArray[np.float64], up_m: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """World points for a grid of slice coordinates.

        Args:
            along_m: ``(ns,)`` coordinates along the cut.
            up_m: ``(nh,)`` coordinates up the cut.

        Returns:
            ``(ns, nh, 3)`` world positions.
        """
        s = np.asarray(along_m, dtype=np.float64)[:, None, None]
        h = np.asarray(up_m, dtype=np.float64)[None, :, None]
        return self.origin_m[None, None, :] + s * self.along + h * self.up

    def describe(self) -> str:
        """One line naming the placement in metres and degrees."""
        return (
            f"{self.name}: offset {self.offset_m * 1e3:+.4g} mm heel-to-toe, "
            f"obliquity {self.obliquity_deg:.4g} deg"
        )


def swing_plane(*, offset_m: float = 0.0, height_m: float = 0.0) -> CuttingPlane:
    """The plane along the club path, at a heel-to-toe offset.

    Args:
        offset_m: Heel-to-toe offset from the solved plane.
        height_m: World height of the cut's ``h = 0`` line, normally the
            free surface.

    Returns:
        The plane.
    """
    return CuttingPlane(
        name=PlanePreset.SWING_PLANE.label,
        origin_m=np.array([0.0, float(offset_m), float(height_m)]),
        along=_ALONG_PATH,
        up=_UP,
        preset=PlanePreset.SWING_PLANE,
    )


def face_normal_plane(
    *, face_open_deg: float = 0.0, offset_m: float = 0.0, height_m: float = 0.0
) -> CuttingPlane:
    """The plane containing the face normal and the vertical.

    Args:
        face_open_deg: How far the face is open, about the vertical. A
            square face gives the swing plane back.
        offset_m: Heel-to-toe offset from the solved plane.
        height_m: World height of the cut's ``h = 0`` line.

    Returns:
        The plane.
    """
    angle = math.radians(float(face_open_deg))
    return CuttingPlane(
        name=(
            f"{PlanePreset.FACE_NORMAL.label} ({float(face_open_deg):+.3g} deg open)"
        ),
        origin_m=np.array([0.0, float(offset_m), float(height_m)]),
        along=np.array([math.cos(angle), math.sin(angle), 0.0]),
        up=_UP,
        preset=PlanePreset.FACE_NORMAL,
    )


def heel_to_toe_series(
    *, width_m: float, n_stations: int = 5, height_m: float = 0.0
) -> tuple[CuttingPlane, ...]:
    """Planes stepping across the sole, heel to toe.

    Args:
        width_m: The declared out-of-plane width the stations span. For
            F1 this is the solver's ``effective_width_m``, which is a
            stated assumption rather than a result -- so the stations
            span exactly what the tier claims and no further.
        n_stations: How many stations, at least one.
        height_m: World height of each cut's ``h = 0`` line.

    Returns:
        The planes, heel (negative offset) to toe.

    Raises:
        ValueError: If the width or the station count is unusable.
    """
    width = float(width_m)
    count = int(n_stations)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"width_m must be positive, got {width_m!r}")
    if count < 1:
        raise ValueError(f"n_stations must be at least 1, got {n_stations!r}")
    offsets = (
        np.zeros(1) if count == 1 else np.linspace(-0.5 * width, 0.5 * width, count)
    )
    planes = []
    for index, offset in enumerate(offsets):
        planes.append(
            CuttingPlane(
                name=(
                    f"Heel-to-toe station {index + 1}/{count} "
                    f"({float(offset) * 1e3:+.4g} mm)"
                ),
                origin_m=np.array([0.0, float(offset), float(height_m)]),
                along=_ALONG_PATH,
                up=_UP,
                preset=PlanePreset.HEEL_TO_TOE,
            )
        )
    return tuple(planes)


def body_focus_bounds_m(
    series: SandFieldSeries, plane: CuttingPlane, *, margin_m: float = 0.030
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Cut bounds framing the intruder's travel, at the field's own depth.

    Horizontally only.  An F1 bed runs in and out well beyond the impact
    zone so the club has somewhere to arrive from -- on the 2 mm
    reference capture the bed is 195 mm wide for a 40 mm sole -- and a
    cut over the whole of it puts the part anybody cares about in the
    middle third of the picture.  Vertically the field's own extent is
    already the right window: it is the bed and a little air.

    Framing is not cropping.  The axes stay labelled in millimetres, so a
    reader can see exactly what window they have.

    Args:
        series: The field.
        plane: The cut the bounds are expressed in.
        margin_m: Horizontal margin added on each side.

    Returns:
        ``(along_bounds, up_bounds)``, or ``None`` when the field carries
        no intruder outline to frame.
    """
    outline = series.body_outline_m
    geometry = series.geometry
    if outline is None or geometry is None:
        return None
    cosine = float(plane.along @ _ALONG_PATH)
    along = (outline[..., 0] - plane.origin_m[0]) / cosine
    margin = float(margin_m)
    lower, upper = geometry.bounds_m()
    return (
        (float(along.min()) - margin, float(along.max()) + margin),
        (float(lower[1] - plane.origin_m[2]), float(upper[1] - plane.origin_m[2])),
    )


def preset_planes(
    series: SandFieldSeries,
    *,
    face_open_deg: float = 0.0,
    n_stations: int = 5,
    height_m: float = 0.0,
) -> tuple[CuttingPlane, ...]:
    """Every named preset for one field, sized from its own settings.

    The heel-to-toe span comes from the field's recorded
    ``effective_width_m`` rather than from a caller's guess, so the
    stations cannot step outside the slab the tier declared.

    Args:
        series: The field the planes will cut.
        face_open_deg: Face-open angle for the face-normal preset.
        n_stations: Stations in the heel-to-toe series.
        height_m: World height of each cut's ``h = 0`` line.

    Returns:
        Swing plane, face-normal plane, then the heel-to-toe series.
    """
    settings = series.provenance.settings
    width = float(settings.get("effective_width_m", 0.03))
    return (
        swing_plane(height_m=height_m),
        face_normal_plane(face_open_deg=face_open_deg, height_m=height_m),
        *heel_to_toe_series(width_m=width, n_stations=n_stations, height_m=height_m),
    )


@dataclass(frozen=True)
class SliceSample:
    """One frame of one cut, resampled onto the cut's own axes.

    Attributes:
        plane: The cut this came from.
        frame: Frame index within the field.
        time_s: Simulation time of the frame.
        along_m: ``(ns,)`` coordinates along the cut.
        up_m: ``(nh,)`` coordinates up the cut.
        velocity_along_m_s: ``(ns, nh)`` in-cut horizontal velocity.
        velocity_up_m_s: ``(ns, nh)`` in-cut vertical velocity.
        velocity_through_m_s: ``(ns, nh)`` velocity through the cut, or
            ``None`` when it would be identically zero by the model's
            construction rather than by measurement. See
            :attr:`through_plane_note`.
        density_kg_m3: ``(ns, nh)`` density.
        shear_rate_1_s: ``(ns, nh)`` shear rate, or ``None``.
        occupied: ``(ns, nh)`` mask of samples holding reportable sand.
        body_outline_m: ``(V, 2)`` intruder outline in cut coordinates,
            or ``None`` when the field carried none.
        fidelity: What this cut is, relative to the solved plane.
        through_plane_note: Why the through-cut component is or is not
            reported, in words, for the frame stamp.
    """

    plane: CuttingPlane
    frame: int
    time_s: float
    along_m: NDArray[np.float64]
    up_m: NDArray[np.float64]
    velocity_along_m_s: NDArray[np.float64]
    velocity_up_m_s: NDArray[np.float64]
    velocity_through_m_s: NDArray[np.float64] | None
    density_kg_m3: NDArray[np.float64]
    shear_rate_1_s: NDArray[np.float64] | None
    occupied: NDArray[np.bool_]
    body_outline_m: NDArray[np.float64] | None
    fidelity: SliceFidelity
    through_plane_note: str

    @property
    def speed_m_s(self) -> NDArray[np.float64]:
        """``(ns, nh)`` in-cut speed, ``nan`` where there is no sand."""
        magnitude = np.hypot(self.velocity_along_m_s, self.velocity_up_m_s)
        return np.where(self.occupied, magnitude, np.nan)

    @property
    def masked_density_kg_m3(self) -> NDArray[np.float64]:
        """``(ns, nh)`` density, ``nan`` where there is no sand."""
        return np.where(self.occupied, self.density_kg_m3, np.nan)

    @property
    def masked_shear_rate_1_s(self) -> NDArray[np.float64] | None:
        """``(ns, nh)`` shear rate, ``nan`` where there is no sand."""
        if self.shear_rate_1_s is None:
            return None
        return np.where(self.occupied, self.shear_rate_1_s, np.nan)

    @property
    def peak_speed_m_s(self) -> float:
        """The fastest sand on this cut, or 0 where the cut is empty."""
        speeds = self.speed_m_s
        if not bool(np.isfinite(speeds).any()):
            return 0.0
        return float(np.nanmax(speeds))

    def describe(self) -> str:
        """One line: the cut, what it is, and when."""
        return (
            f"{self.plane.describe()}; {self.fidelity.label}; "
            f"t = {self.time_s * 1e3:.4g} ms"
        )


def sample_plane(
    series: SandFieldSeries,
    frame: int,
    plane: CuttingPlane,
    *,
    n_along: int = _DEFAULT_SAMPLES,
    n_up: int = _DEFAULT_SAMPLES,
    along_bounds_m: tuple[float, float] | None = None,
    up_bounds_m: tuple[float, float] | None = None,
) -> SliceSample:
    """Resample one frame of a field onto one cutting plane.

    Args:
        series: The field. Must be a ``GRID`` layout, since a cut needs
            sample points at known places.
        frame: Frame index.
        plane: The cut.
        n_along: Samples along the cut.
        n_up: Samples up the cut.
        along_bounds_m: ``(lower, upper)`` extent along the cut.
            Defaults to the field's own extent, mapped onto the cut.
        up_bounds_m: ``(lower, upper)`` extent up the cut. Defaults to
            the field's own vertical extent.

    Returns:
        The sample.

    Raises:
        ValueError: If the field is not a sliceable ``GRID``, if the
            frame is outside it, if the cut is edge-on to the solved
            plane, or if the cut is offset further out than the tier's
            declared width.
    """
    if series.layout is not FieldLayout.GRID:
        raise ValueError(
            "only a GRID field can be sliced on a plane; a PARTICLE field has "
            "no lattice to interpolate from, and scattering its samples onto "
            "one here would invent a field the tier did not produce"
        )
    geometry = series.geometry
    if geometry is None:  # pragma: no cover - GRID always carries geometry
        raise ValueError("a GRID field must carry its geometry")
    if geometry.dimension != 2:
        raise ValueError(
            f"this cut is written for a 2-D plane-strain field; the field is "
            f"{geometry.dimension}-D"
        )
    series.require_frame(frame)
    fidelity, note = _classify(series, plane)

    lower, upper = geometry.bounds_m()
    along_axis, up_axis = _axes(
        plane, lower, upper, n_along, n_up, along_bounds_m, up_bounds_m
    )
    world = plane.world_m(along_axis, up_axis)
    # The model knows (x, z) only, so a cut point maps onto the solved plane
    # by dropping its heel-to-toe coordinate. That drop IS the plane-strain
    # assumption, and it is the reason `fidelity` is not always SOLVED.
    query = np.stack([world[..., 0], world[..., 2]], axis=-1)

    velocity = _bilinear(geometry, series.velocity_m_s[int(frame)], query, width=2)
    density = _bilinear(geometry, series.density_kg_m3[int(frame)], query, width=None)
    shear = (
        None
        if series.shear_rate_1_s is None
        else _bilinear(geometry, series.shear_rate_1_s[int(frame)], query, width=None)
    )
    occupied = np.asarray(series.occupancy.occupied(np.nan_to_num(density, nan=0.0)))

    world_velocity = np.stack(
        [velocity[..., 0], np.zeros_like(velocity[..., 0]), velocity[..., 1]], axis=-1
    )
    through = None
    if fidelity is SliceFidelity.PROJECTED:
        through = world_velocity @ plane.normal

    return SliceSample(
        plane=plane,
        frame=int(frame),
        time_s=float(series.time_s[int(frame)]),
        along_m=along_axis,
        up_m=up_axis,
        velocity_along_m_s=world_velocity @ plane.along,
        velocity_up_m_s=world_velocity @ plane.up,
        velocity_through_m_s=through,
        density_kg_m3=density,
        shear_rate_1_s=shear,
        occupied=occupied,
        body_outline_m=_outline_in_cut(series, frame, plane),
        fidelity=fidelity,
        through_plane_note=note,
    )


def _classify(
    series: SandFieldSeries, plane: CuttingPlane
) -> tuple[SliceFidelity, str]:
    """What this cut is, and what may be said about flow through it."""
    obliquity = plane.obliquity_deg
    offset = plane.offset_m
    settings = series.provenance.settings
    width = float(settings.get("effective_width_m", 0.0))
    if width > 0.0 and abs(offset) > 0.5 * width + _TOL:
        raise ValueError(
            f"the cut is {offset * 1e3:+.4g} mm heel-to-toe, outside the "
            f"{width * 1e3:.4g} mm effective width this field declares. That "
            "width is a stated assumption, not a result, so there is nothing "
            "here to extrude into."
        )
    if abs(float(plane.along @ _ALONG_PATH)) < _MIN_OBLIQUITY_COS:
        raise ValueError(
            "the cut is edge-on to the solved plane, so it meets it in a line "
            "rather than an area. A plane-strain field has no picture to show "
            "on that cut."
        )
    if obliquity > _TOL:
        return (
            SliceFidelity.PROJECTED,
            (
                f"through-cut velocity is in-plane flow resolved at "
                f"{obliquity:.4g} deg, NOT measured heel-to-toe flow -- "
                "this tier has none"
            ),
        )
    return (
        SliceFidelity.SOLVED if abs(offset) <= _TOL else SliceFidelity.EXTRUDED,
        (
            "no through-cut velocity: plane strain has no heel-to-toe flow, "
            "so it is absent here rather than measured as zero"
        ),
    )


def _axes(
    plane: CuttingPlane,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    n_along: int,
    n_up: int,
    along_bounds_m: tuple[float, float] | None,
    up_bounds_m: tuple[float, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The cut's own sample axes, defaulting to the field's extent."""
    if int(n_along) < 2 or int(n_up) < 2:
        raise ValueError(
            f"a cut needs at least 2 samples on each axis, got {n_along!r} x {n_up!r}"
        )
    cosine = float(plane.along @ _ALONG_PATH)
    if along_bounds_m is None:
        # The along-axis is compressed by cos(obliquity) when it maps onto the
        # solved plane, so covering the field's x extent needs a longer cut.
        span = (
            (lower[0] - plane.origin_m[0]) / cosine,
            (upper[0] - plane.origin_m[0]) / cosine,
        )
        along_bounds_m = (min(span), max(span))
    if up_bounds_m is None:
        up_bounds_m = (
            float(lower[1] - plane.origin_m[2]),
            float(upper[1] - plane.origin_m[2]),
        )
    for name, bounds in (
        ("along_bounds_m", along_bounds_m),
        ("up_bounds_m", up_bounds_m),
    ):
        if bounds[1] <= bounds[0]:
            raise ValueError(f"{name} must be increasing, got {bounds!r}")
    return (
        np.linspace(
            along_bounds_m[0], along_bounds_m[1], int(n_along), dtype=np.float64
        ),
        np.linspace(up_bounds_m[0], up_bounds_m[1], int(n_up), dtype=np.float64),
    )


def _bilinear(
    geometry: object,
    values: NDArray[np.float64],
    query_m: NDArray[np.float64],
    *,
    width: int | None,
) -> NDArray[np.float64]:
    """Bilinear interpolation from a uniform lattice, ``nan`` outside it.

    Outside is ``nan`` rather than clamped for the same reason
    :func:`~bunkershot3d.solvers.mpm.state.surface_profile_m` reports an
    emptied bin as ``nan``: extending the edge value outward would draw
    sand where the solve had none.
    """
    shape = geometry.shape  # type: ignore[attr-defined]
    origin = geometry.origin_m  # type: ignore[attr-defined]
    size = float(geometry.cell_size_m)  # type: ignore[attr-defined]
    grid = np.asarray(values).reshape((*shape, width) if width is not None else shape)

    local = (np.asarray(query_m, dtype=np.float64) - origin) / size
    # Membership is decided on the continuous coordinate, not on the floored
    # cell index: a query landing exactly on the far edge of the lattice is
    # inside it, and testing the index alone would drop that whole last row
    # and column -- which is precisely the free surface on a bed-height cut.
    inside = np.ones(local.shape[:-1], dtype=bool)
    for axis, count in enumerate(shape):
        inside &= (local[..., axis] >= -_TOL) & (local[..., axis] <= count - 1 + _TOL)
    clipped = np.stack(
        [
            np.clip(np.floor(local[..., axis]), 0, shape[axis] - 2)
            for axis in range(len(shape))
        ],
        axis=-1,
    ).astype(np.int64)
    frac = local - clipped
    i, j = clipped[..., 0], clipped[..., 1]
    u, v = frac[..., 0], frac[..., 1]
    if width is not None:
        u = u[..., None]
        v = v[..., None]
    blended = (
        grid[i, j] * (1.0 - u) * (1.0 - v)
        + grid[i + 1, j] * u * (1.0 - v)
        + grid[i, j + 1] * (1.0 - u) * v
        + grid[i + 1, j + 1] * u * v
    )
    mask = inside if width is None else inside[..., None]
    return np.where(mask, blended, np.nan)


def _outline_in_cut(
    series: SandFieldSeries, frame: int, plane: CuttingPlane
) -> NDArray[np.float64] | None:
    """The intruder outline in the cut's own ``(s, h)`` coordinates."""
    outline = series.body_outline_m
    if outline is None:
        return None
    section = np.asarray(outline[int(frame)])
    cosine = float(plane.along @ _ALONG_PATH)
    along = (section[:, 0] - plane.origin_m[0]) / cosine
    up = section[:, 1] - plane.origin_m[2]
    return np.stack([along, up], axis=1)


@dataclass(frozen=True)
class SliceScale:
    """Colour limits held fixed across frames and across designs.

    Built by :meth:`covering` over every frame of every field being
    compared and injected into the renderer.  Issue #8728 fixed a real
    bug where per-grid auto-scaling made two designs incomparable; a
    cross-section view whose ramp moved between frames would be the same
    bug in the place it does the most damage, because the whole question
    is whether a magnitude *changes* through impact.

    Attributes:
        speed_m_s: ``(0, peak)`` speed limits.
        density_kg_m3: ``(lower, upper)`` density limits.
        shear_rate_1_s: ``(0, peak)`` shear limits, or ``None`` when no
            compared field carried a shear rate.
    """

    speed_m_s: tuple[float, float]
    density_kg_m3: tuple[float, float]
    shear_rate_1_s: tuple[float, float] | None

    def __post_init__(self) -> None:
        for name in ("speed_m_s", "density_kg_m3", "shear_rate_1_s"):
            limits = getattr(self, name)
            if limits is None:
                continue
            lower, upper = (float(value) for value in limits)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"{name} limits must be finite, got {limits!r}")
            if upper < lower:
                raise ValueError(f"{name} limits must not decrease, got {limits!r}")
            object.__setattr__(self, name, (lower, upper))

    @property
    def speed_unit(self) -> str:
        """Unit of the speed ramp."""
        return VELOCITY_UNIT

    @property
    def density_unit(self) -> str:
        """Unit of the density ramp."""
        return DENSITY_UNIT

    @property
    def shear_unit(self) -> str:
        """Unit of the shear ramp."""
        return SHEAR_RATE_UNIT

    def merged(self, other: SliceScale) -> SliceScale:
        """The scale covering both, so two designs share one ramp."""
        return SliceScale(
            speed_m_s=_union(self.speed_m_s, other.speed_m_s),
            density_kg_m3=_union(self.density_kg_m3, other.density_kg_m3),
            shear_rate_1_s=(
                None
                if self.shear_rate_1_s is None and other.shear_rate_1_s is None
                else _union(
                    self.shear_rate_1_s or (0.0, 0.0),
                    other.shear_rate_1_s or (0.0, 0.0),
                )
            ),
        )

    @classmethod
    def covering(cls, fields: Sequence[SandFieldSeries]) -> SliceScale:
        """The scale covering every frame of every given field.

        Limits are taken from the **occupied** samples only. A stencil
        tail divides round-off by a millionth of a cell's sand and would
        otherwise set a ramp nothing else on the picture could reach.

        Args:
            fields: The fields to compare. At least one.

        Returns:
            The shared scale.

        Raises:
            ValueError: If no field is given -- an empty covering set has
                no limits, and inventing some would be the auto-scaling
                bug wearing a different hat.
        """
        if not fields:
            raise ValueError(
                "a colour scale needs at least one field to cover; scaling to "
                "nothing means scaling to whatever arrives next"
            )
        scale: SliceScale | None = None
        for series in fields:
            speeds = series.occupied_speed_m_s()
            occupied = series.occupied()
            densities = np.where(occupied, series.density_kg_m3, np.nan)
            shear = (
                None
                if series.shear_rate_1_s is None
                else np.where(occupied, series.shear_rate_1_s, np.nan)
            )
            candidate = cls(
                speed_m_s=(0.0, _peak(speeds)),
                density_kg_m3=(0.0, _peak(densities)),
                shear_rate_1_s=None if shear is None else (0.0, _peak(shear)),
            )
            scale = candidate if scale is None else scale.merged(candidate)
        if scale is None:  # pragma: no cover - guarded above
            raise ValueError("a colour scale needs at least one field to cover")
        return scale


def slice_scale(fields: Sequence[SandFieldSeries]) -> SliceScale:
    """Shorthand for :meth:`SliceScale.covering`."""
    return SliceScale.covering(fields)


@dataclass(frozen=True)
class CursorMap:
    """How somebody else's frame index becomes this field's frame index.

    The workbench has one transport, owned by the sole-load field view,
    and issue #8711 says to reuse it rather than grow a second slider.
    But an F1 field is not sampled on the F0 shot's clock: it is a
    strided march of a *declared approach*, with its own step count and
    its own time base.

    So the mapping is by fractional progress, and the frame says so.
    Pretending the two clocks were the same would be a quieter lie than
    a second slider.

    Attributes:
        n_transport: Frames the transport owns.
        n_field: Frames the field has.
    """

    n_transport: int
    n_field: int

    def __post_init__(self) -> None:
        for name in ("n_transport", "n_field"):
            if int(getattr(self, name)) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)!r}"
                )
        object.__setattr__(self, "n_transport", int(self.n_transport))
        object.__setattr__(self, "n_field", int(self.n_field))

    @property
    def is_one_to_one(self) -> bool:
        """Whether the two records happen to have the same length."""
        return self.n_transport == self.n_field

    def field_frame(self, transport_frame: int) -> int:
        """The field frame a transport frame maps onto.

        Args:
            transport_frame: Index in the transport's record.

        Returns:
            The field frame index.

        Raises:
            ValueError: If ``transport_frame`` is outside the transport,
                which is the same refusal the other linked views make
                rather than clamping to an end.
        """
        index = int(transport_frame)
        if not 0 <= index < self.n_transport:
            raise ValueError(
                f"frame {transport_frame} is outside the shot, which has "
                f"{self.n_transport} samples"
            )
        if self.n_transport == 1 or self.n_field == 1:
            return 0
        progress = index / (self.n_transport - 1)
        return int(round(progress * (self.n_field - 1)))

    def describe(self) -> str:
        """One line for the frame stamp, naming the mapping."""
        if self.is_one_to_one:
            return f"cursor 1:1 with the field ({self.n_field} frames)"
        return (
            f"cursor mapped by progress: {self.n_transport} shot samples -> "
            f"{self.n_field} field frames (different time bases)"
        )


def _union(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    """The interval covering both."""
    return (min(left[0], right[0]), max(left[1], right[1]))


def _peak(values: NDArray[np.float64]) -> float:
    """The largest finite value, or 0 when there is none."""
    if not bool(np.isfinite(values).any()):
        return 0.0
    return float(np.nanmax(values))


def _vector(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """A finite immutable 3-vector."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,) or not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite 3-vector, got {value!r}")
    array = array.copy()
    array.flags.writeable = False
    return array


def _unit(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """A finite unit 3-vector."""
    array = np.asarray(_vector(value, name), dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= _TOL:
        raise ValueError(f"{name} must have a direction, got {value!r}")
    unit = array / norm
    unit.flags.writeable = False
    return unit
