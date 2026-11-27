"""Engine-agnostic smooth body-segment meshes (pure numpy).

``lofted_segment`` builds a closed spindle (or a flat-capped garment band)
along a segment; ``ellipsoid_mesh`` builds an oriented ellipsoid (club head,
shoe toe). Meshes carry vertices and triangle faces in the caller's frame and
are visual only: they are never used for collision, mass or inertia.

A non-round cross-section (``aspect != 1``) needs an explicit width axis: the
direction that carries the full radius. ``section_width_axis`` is the one rule
that turns a body's depth axis into that width axis per segment (R1, #12680).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.contracts import ensure, require

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Mesh:
    vertices: Array  # (N, 3) float
    faces: NDArray[np.int64]  # (M, 3) outward-wound triangles

    def volume(self) -> float:
        """Signed enclosed volume; positive for outward winding."""
        tri = self.vertices[self.faces]
        return float(
            np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        )


def concat_meshes(*meshes: Mesh) -> Mesh:
    """One mesh holding every input mesh as a separate (closed) shell.

    Precondition: at least one mesh. Postcondition: faces are re-indexed into
    the stacked vertex array, so a set of watertight shells stays watertight.
    """
    if not meshes:
        raise ValueError("concat_meshes needs at least one mesh")
    verts, faces, base = [], [], 0
    for mesh in meshes:
        verts.append(mesh.vertices)
        faces.append(mesh.faces + base)
        base += len(mesh.vertices)
    return Mesh(np.vstack(verts), np.vstack(faces))


def _frame(axis: Array, hint: Array | None = None) -> tuple[Array, Array, Array]:
    """Right-handed (u, v, a) with ``a`` the unit axis."""
    a = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if hint is None else np.asarray(hint, float)
    if abs(float(np.dot(ref, a))) > 0.9:
        ref = (
            np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        )
    u = ref - np.dot(ref, a) * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u), a


# A width axis closer than this to the segment axis is ill-conditioned: its
# projection onto the cross-section plane is dominated by the axis itself.
MIN_WIDTH_AXIS_ANGLE_DEG = 10.0
_MAX_AXIS_COS = math.cos(math.radians(MIN_WIDTH_AXIS_ANGLE_DEG))


def _unit(name: str, value: Array) -> Array:
    v = _check_finite(name, value, 3)
    norm = float(np.linalg.norm(v))
    require(norm > 1e-12, f"{name} must be nonzero", value)
    return v / norm


def section_width_axis(segment_axis: Array, depth_axis: Array) -> Array:
    """Width axis of a segment's cross-section: ``segment x depth``, unit.

    ``depth_axis`` is the body direction that gets the thin (``aspect``)
    semi-axis: anterior for the trunk, up for the foot. The width axis then
    carries the full radius and is perpendicular to both, so a vertical spine
    segment gets the mediolateral width and a segment that itself runs
    mediolaterally (pelvis to hip) gets the vertical one; depth stays depth.
    Precondition: the two axes are at least ``MIN_WIDTH_AXIS_ANGLE_DEG`` apart.
    Postcondition: unit, perpendicular to both inputs.
    """
    a = _unit("segment_axis", segment_axis)
    d = _unit("depth_axis", depth_axis)
    require(
        abs(float(np.dot(a, d))) <= _MAX_AXIS_COS,
        f"depth axis must be >= {MIN_WIDTH_AXIS_ANGLE_DEG} deg from the segment",
        (a, d),
    )
    width = np.asarray(np.cross(a, d), dtype=np.float64)
    width /= np.linalg.norm(width)
    ensure(
        abs(float(np.dot(width, a))) < 1e-9 and abs(float(np.dot(width, d))) < 1e-9,
        "width axis must be perpendicular to the segment and the depth axis",
        width,
    )
    return width


def _width_frame(axis: Array, width: Array) -> tuple[Array, Array, Array]:
    """Right-handed (u, v, a): ``u`` is ``width`` made perpendicular to ``axis``.

    Unlike ``_frame`` there is no silent fallback: a width axis within
    ``MIN_WIDTH_AXIS_ANGLE_DEG`` of the segment is a precondition violation.
    """
    a = axis / np.linalg.norm(axis)
    w = _unit("width_hint", width)
    require(
        abs(float(np.dot(w, a))) <= _MAX_AXIS_COS,
        f"width_hint must be >= {MIN_WIDTH_AXIS_ANGLE_DEG} deg from the segment axis",
        w,
    )
    u = w - np.dot(w, a) * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u), a


def _check_finite(name: str, value: Array, size: int) -> Array:
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return v


def _orient(vertices: Array, faces: NDArray[np.int64]) -> Mesh:
    mesh = Mesh(vertices, faces)
    if mesh.volume() < 0:
        mesh = Mesh(vertices, faces[:, ::-1].copy())
    return mesh


def _ring_faces(n_rings: int, sides: int) -> list[tuple[int, int, int]]:
    faces = []
    for k in range(n_rings - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a, b = k * sides + j, k * sides + j2
            c, d = (k + 1) * sides + j2, (k + 1) * sides + j
            faces += [(a, b, c), (a, c, d)]
    return faces


@dataclass(frozen=True)
class LoftOptions:
    """Shape options for :func:`lofted_segment`."""

    aspect: float = 1.0
    coverage: tuple[float, float] = (0.0, 1.0)
    thickness: float = 1.0
    rings: int = 14
    sides: int = 20
    # Direction of the full-radius (width) semi-axis; required when aspect != 1.
    width_hint: Array | None = None
    # Radius at the ``end`` over the radius at ``start`` (linear taper, 1 = none).
    end_scale: float = 1.0
    # This segment is the fraction ``(from, to)`` of a longer limb drawn as one
    # shape (#15262). An end inside the limb is cut flat on a round section;
    # taper, belly and end rounding are those of the whole limb.
    window: tuple[float, float] = (0.0, 1.0)
    # Muscle belly ``(gain, position, width)``: the radius grows by ``gain``
    # at ``position`` along the whole limb and falls to nothing (raised
    # cosine) ``width`` away on either side (fractions of its length). A gain
    # of 0 is no belly.
    belly: tuple[float, float, float] = (0.0, 0.5, 1.0)
    # Ends of the whole limb (#15262). False: a spindle, already thinner at
    # the joint. True: the radius holds to each joint and the skin beyond is a
    # ball about the joint, so a neighbour turning there opens no gap.
    capsule_ends: bool = False
    # Fraction of the whole limb by which the taper has reached ``end_scale``.
    taper_end: float = 1.0
    # Depth/width of the section at the far end of the whole limb (a flat
    # wrist, #15262). The section is ``aspect`` up to ``flatten_from`` of the
    # limb and eases to ``end_aspect`` by ``taper_end``. ``None`` keeps
    # ``aspect`` throughout.
    end_aspect: float | None = None
    flatten_from: float = 0.0
    # Length of the far end's rounding over its radius (capsule ends): 1 is a
    # ball about the joint, less is a shallower cap that stops sooner (#15264).
    end_cap: float = 1.0
    # Sideways displacement (m) of the centreline at the far end of the whole
    # limb, eased in between ``flatten_from`` and ``taper_end``; its part
    # along the limb is ignored (#15264).
    end_shift: Array | None = None
    # Sideways bow ``(vector_m, position, width)`` of the centreline (#15299):
    # the sections move by ``vector_m`` at ``position`` along the whole limb,
    # falling to nothing ``width`` away like ``belly`` (a calf behind the
    # shin). The vector's part along the limb is ignored.
    offset: tuple[Array, float, float] | None = None
    # Radius at the near joint over the limb's radius, easing to the full
    # radius by ``start_until`` of the whole limb (an elbow no wider than the
    # upper arm it meets, #15299). 1 is no narrowing.
    start_scale: float = 1.0
    start_until: float = 1.0
    # Cloth ease ``(near, far, from)`` (#15299): the radius is times ``near``
    # up to ``from`` of the whole limb and grows evenly to ``far`` at its far
    # end (a trouser leg that hangs loose below a fitted thigh).
    drape: tuple[float, float, float] = (1.0, 1.0, 0.0)


def _checked_shape(opts: LoftOptions) -> None:
    """Preconditions of the whole-limb shape options (``ValueError``)."""
    if not 0.0 < opts.end_scale <= 1.0:  # NaN fails too
        raise ValueError("taper end_scale must be in (0, 1]")
    w0, w1 = opts.window
    if not 0.0 <= w0 < w1 <= 1.0:
        raise ValueError("window must satisfy 0 <= from < to <= 1")
    gain, position, width = opts.belly
    if not (gain >= 0.0 and math.isfinite(gain) and 0.0 <= position <= 1.0):
        raise ValueError("belly needs a finite gain >= 0 and a position in [0, 1]")
    if not (width > 0.0 and math.isfinite(width)):
        raise ValueError("belly width must be positive")
    if not 0.0 < opts.taper_end <= 1.0:
        raise ValueError("taper_end must be in (0, 1]")
    if not (0.0 < opts.start_scale <= 1.0 and 0.0 < opts.start_until <= 1.0):
        raise ValueError("start_scale and start_until must be in (0, 1]")
    if not 0.0 < opts.end_cap <= 1.0:
        raise ValueError("end_cap must be in (0, 1]")
    near, far, begin = opts.drape
    if not (near > 0.0 and far > 0.0 and math.isfinite(near + far)):
        raise ValueError("drape needs finite factors > 0")
    if not 0.0 <= begin < 1.0:
        raise ValueError("drape must start in [0, 1) of the limb")
    if opts.offset is not None:
        vector, position, width = opts.offset
        _check_finite("offset vector", vector, 3)
        if not (0.0 <= position <= 1.0 and width > 0.0 and math.isfinite(width)):
            raise ValueError("offset needs a position in [0, 1] and a width > 0")
    if opts.end_shift is not None and not opts.flatten_from < opts.taper_end:
        raise ValueError("end_shift needs flatten_from < taper_end")
    if opts.end_aspect is not None and not (
        opts.end_aspect > 0.0 and 0.0 <= opts.flatten_from < opts.taper_end
    ):
        raise ValueError("end_aspect needs a value > 0 and flatten_from < taper_end")


def _aspect_at(opts: LoftOptions, frac: float) -> float:
    """Depth/width of the section at ``frac`` of the whole limb."""
    if opts.end_aspect is None:
        return opts.aspect
    return opts.aspect + (opts.end_aspect - opts.aspect) * _ease(opts, frac)


def _ease(opts: LoftOptions, frac: float) -> float:
    """0 up to ``flatten_from`` of the whole limb, 1 from ``taper_end`` (smooth)."""
    span = opts.taper_end - opts.flatten_from
    t = min(1.0, max(0.0, (frac - opts.flatten_from) / span))
    return t * t * (3.0 - 2.0 * t)


def _limb_scale(opts: LoftOptions, frac: float) -> float:
    """Taper times belly times drape at ``frac`` of the whole limb (clamped)."""
    frac = min(1.0, max(0.0, frac))
    gain, position, width = opts.belly
    reach = min(1.0, abs(frac - position) / width)
    swell = 1.0 + gain * math.cos(0.5 * math.pi * reach) ** 2
    lo, far, begin = opts.drape
    swell *= lo + (far - lo) * max(0.0, (frac - begin) / (1.0 - begin))
    t = min(1.0, frac / opts.start_until)
    near = opts.start_scale + (1.0 - opts.start_scale) * t * t * (3.0 - 2.0 * t)
    taper = 1.0 + (opts.end_scale - 1.0) * min(1.0, frac / opts.taper_end)
    return swell * near * taper


def _spindle_stations(
    total: float,
    over: tuple[float, float],
    window: tuple[float, float],
    rings: int,
    joint_rings: bool,
) -> tuple[Array, float, float]:
    """Stations along the whole limb inside ``window``, and its two poles.

    An end of the window that is an end of the limb keeps its rounded
    overshoot (pole ``over`` past it); an end inside the limb is a flat cut
    (ring and pole at the cut). ``joint_rings`` (capsule ends) adds a ring
    exactly at each joint, where the end starts to round.
    """
    first, last = -over[0], total + over[1]
    theta = np.linspace(0.0, math.pi, rings + 2)[1:-1]
    stations = 0.5 * (first + last) - 0.5 * (last - first) * np.cos(theta)
    if joint_rings:
        clear = (np.abs(stations) > 1e-9) & (np.abs(stations - total) > 1e-9)
        stations = np.sort(np.append(stations[clear], [0.0, total]))
    low, high = window[0] * total, window[1] * total
    if window[0] > 0.0:
        stations, first = np.append(low, stations[stations > low + 1e-9]), low
    if window[1] < 1.0:
        stations, last = np.append(stations[stations < high - 1e-9], high), high
    return stations, first, last


def _checked_loft(opts: LoftOptions, length: float, radius: float) -> float:
    """Preconditions of :func:`lofted_segment` (``ValueError``); gives ``length``."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius, thickness and aspect must be positive")
    if opts.thickness <= 0 or opts.aspect <= 0:
        raise ValueError("radius, thickness and aspect must be positive")
    t0, t1 = opts.coverage
    if not 0.0 <= t0 < t1 <= 1.0:
        raise ValueError("coverage must satisfy 0 <= start < end <= 1")
    if length <= 1e-9 or opts.rings < 4 or opts.sides < 6:
        raise ValueError("Segment needs positive length, rings >= 4, sides >= 6")
    _checked_shape(opts)
    if opts.aspect != 1.0 or opts.end_aspect not in (None, opts.aspect):
        require(
            opts.width_hint is not None,
            "a non-round cross-section (aspect != 1) needs an explicit width_hint",
            opts.aspect,
        )
    return length


def lofted_segment(
    start: Array,
    end: Array,
    radius: float,
    options: LoftOptions | None = None,
) -> Mesh:
    """Smooth lofted body between ``start`` and ``end``.

    Full coverage gives a closed spindle that overshoots both ends by half a
    radius so neighbouring segments overlap at joints. A partial ``coverage``
    (fractions of the length) gives a flat-capped band, used for sleeves,
    shorts and shoes. ``aspect`` is depth/width of the cross-section: the
    extent along ``width_hint`` (made perpendicular to the segment) is
    ``2 * radius * thickness`` and along ``segment x width`` it is ``aspect``
    times that. ``window`` and ``belly`` draw the segment as part of a
    longer limb (see :class:`LoftOptions`).
    Preconditions: finite points, positive length and radius; when
    ``aspect != 1`` an explicit ``width_hint`` at least
    ``MIN_WIDTH_AXIS_ANGLE_DEG`` from the segment axis (no world default).
    """
    opts = options or LoftOptions()
    aspect, thickness = opts.aspect, opts.thickness
    rings, sides, width_hint = opts.rings, opts.sides, opts.width_hint
    p0, p1 = _check_finite("start", start, 3), _check_finite("end", end, 3)
    length = _checked_loft(opts, float(np.linalg.norm(p1 - p0)), radius)
    t0, t1 = opts.coverage
    u, v, a = (
        _frame(p1 - p0)
        if width_hint is None
        else _width_frame(p1 - p0, np.asarray(width_hint, float))
    )
    # Whole-limb coordinates: this segment starts ``offset`` along the limb.
    total = length / (opts.window[1] - opts.window[0])
    offset = opts.window[0] * total
    full = radius * thickness
    ends = (full * _limb_scale(opts, 0.0), full * _limb_scale(opts, 1.0))
    over = (
        (ends[0], ends[1] * opts.end_cap)
        if opts.capsule_ends
        else (0.5 * radius, 0.5 * radius)
    )
    half = total / 2.0 + over[0]

    def profile(whole: float) -> float:
        """Skin radius (m) at ``whole`` metres along the whole limb."""
        if opts.capsule_ends:
            past = max(-whole, whole - total, 0.0)
            near = whole < 0.0
            end, depth = (ends[0], over[0]) if near else (ends[1], over[1])
            if past:
                return end * math.sqrt(max(0.0, 1.0 - (past / depth) ** 2))
            return full * _limb_scale(opts, whole / total)
        x = min(1.0, abs(whole - total / 2.0) / half)
        return (
            full
            * math.sqrt(max(0.0, 1.0 - x * x)) ** 0.55
            * _limb_scale(opts, whole / total)
        )

    if t0 == 0.0 and t1 == 1.0:
        whole, first, last = _spindle_stations(
            total, over, opts.window, rings, opts.capsule_ends
        )
        stations, poles = whole - offset, (first - offset, last - offset)
    else:
        stations = np.asarray(
            np.linspace(t0 * length, t1 * length, rings), dtype=np.float64
        )
        poles = (float(stations[0]), float(stations[-1]))
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    across, deep = np.outer(np.cos(phi), u), np.outer(np.sin(phi), v)
    shift = np.zeros(3) if opts.end_shift is None else np.asarray(opts.end_shift)
    shift = shift - (shift @ a) * a

    bow, bow_at, bow_width = opts.offset or (np.zeros(3), 0.5, 1.0)
    bow = np.asarray(bow, dtype=float) - (np.asarray(bow, dtype=float) @ a) * a

    def centre(s: float) -> Array:
        """Centreline point ``s`` metres along this segment."""
        frac = (s + offset) / total
        lean = 0.0 if opts.end_shift is None else _ease(opts, frac)
        reach = min(1.0, abs(frac - bow_at) / bow_width)
        return p0 + a * s + lean * shift + math.cos(0.5 * math.pi * reach) ** 2 * bow

    verts: list[Array] = []
    for s in stations:
        at = float(s) + offset
        ring = across + _aspect_at(opts, at / total) * deep
        verts.extend(centre(float(s)) + profile(at) * ring)
    face_list = _ring_faces(len(stations), sides)
    n = len(verts)
    verts += [centre(poles[0]), centre(poles[1])]
    first_ring = (len(stations) - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        face_list.append((n, j2, j))
        face_list.append((n + 1, first_ring + j, first_ring + j2))
    return _orient(
        np.asarray(verts, dtype=float), np.asarray(face_list, dtype=np.int64)
    )


def ellipsoid_mesh(
    center: Array,
    half_sizes: Array,
    axis: Array,
    *,
    rings: int = 12,
    sides: int = 20,
    width_hint: Array | None = None,
) -> Mesh:
    """Ellipsoid whose first half size lies along ``axis`` (then u, v)."""
    c = _check_finite("center", center, 3)
    h = _check_finite("half_sizes", half_sizes, 3)
    if (h <= 0).any():
        raise ValueError("Ellipsoid half sizes must be positive")
    if np.linalg.norm(np.asarray(axis, float)) <= 1e-12:
        raise ValueError("Ellipsoid axis must be nonzero")
    u, v, a = _frame(np.asarray(axis, float), width_hint)
    theta = np.linspace(0.0, math.pi, rings + 2)[1:-1]
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    verts = []
    for t in theta:
        ring = np.outer(math.sin(t) * np.cos(phi), h[1] * u) + np.outer(
            math.sin(t) * np.sin(phi), h[2] * v
        )
        verts.extend(c + a * (h[0] * -math.cos(t)) + ring)
    faces = _ring_faces(rings, sides)
    n = len(verts)
    verts += [c - a * h[0], c + a * h[0]]
    last = (rings - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((n, j2, j))
        faces.append((n + 1, last + j, last + j2))
    return _orient(np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64))


def closed_grid_mesh(grid: Array, top: Array, bottom: Array) -> Mesh:
    """Closed mesh from ``(rings, sides, 3)`` vertex rings between two poles.

    ``top`` is fanned to the first ring and ``bottom`` to the last, so a
    deformed sphere (skull) or a dome fanned to an interior point (hair, cap)
    is built the same way. Preconditions: finite grid with at least two rings
    and six sides. Postcondition: positive enclosed volume (winding repaired).
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 3 or g.shape[2] != 3 or g.shape[0] < 2 or g.shape[1] < 6:
        raise ValueError("grid must be (rings >= 2, sides >= 6, 3)")
    if not np.isfinite(g).all():
        raise ValueError("grid must be finite")
    t = _check_finite("top", top, 3)
    b = _check_finite("bottom", bottom, 3)
    rings, sides = g.shape[:2]
    faces = _ring_faces(rings, sides)
    n = rings * sides
    verts = np.vstack([g.reshape(-1, 3), t, b])
    last = (rings - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((n, j2, j))
        faces.append((n + 1, last + j, last + j2))
    return _orient(verts, np.asarray(faces, dtype=np.int64))


@dataclass(frozen=True)
class SplitEllipsoid:
    """Ellipsoid of two half heights about a widest level, cut flat on top.

    ``centre`` is the widest point; ``up`` the vertical and ``width`` the
    full-width axis (made perpendicular to ``up``). The lower half is a plain
    half-ellipsoid of height ``half_down``; the upper half has height
    ``half_up`` but is cut flat ``top`` above the centre (``top <= half_up``),
    so the cap sits exactly at that plane.
    """

    centre: Array
    up: Array
    width: Array
    half_width: float
    half_depth: float
    half_up: float
    half_down: float
    top: float
    # The buttock (#15299): behind the widest plane (``back`` is the unit
    # direction) the depth is times ``back_scale`` and the outline is a
    # superellipse of exponent ``back_square`` (2 = ellipse). Both are full
    # from the lower pole to the widest level and ease out to nothing at the
    # flat cap, so the belt line keeps its section.
    back: Array | None = None
    back_scale: float = 1.0
    back_square: float = 2.0


def split_ellipsoid_mesh(
    shape: SplitEllipsoid, *, thickness: float = 1.0, rings: int = 14, sides: int = 24
) -> Mesh:
    """Closed mesh of ``shape``; ``thickness`` scales width and depth only.

    Preconditions: finite inputs, positive sizes, ``0 < top <= half_up``,
    even ``sides`` (left/right mirror symmetry), ``rings >= 4``. Postcondition:
    positive volume, symmetric about the plane through ``centre`` normal to
    ``width``; the highest point is the flat cap at ``centre + top * up``.
    """
    c = _check_finite("centre", shape.centre, 3)
    up = _check_finite("up", shape.up, 3)
    wid = _check_finite("width", shape.width, 3)
    sizes = (shape.half_width, shape.half_depth, shape.half_up, shape.half_down)
    if not all(math.isfinite(x) and x > 0 for x in (*sizes, thickness)):
        raise ValueError("ellipsoid sizes and thickness must be positive and finite")
    if not 0 < shape.top <= shape.half_up:
        raise ValueError("top must satisfy 0 < top <= half_up")
    if sides < 6 or sides % 2 or rings < 4:
        raise ValueError("need even sides >= 6 and rings >= 4")
    bulged = (shape.back_scale, shape.back_square) != (1.0, 2.0)
    if not (shape.back_scale > 0.0 and 2.0 <= shape.back_square <= 8.0):
        raise ValueError("back_scale must be > 0 and back_square in [2, 8]")
    if bulged and shape.back is None:
        raise ValueError("a back bulge needs the back direction")
    a = up / np.linalg.norm(up)
    u = wid - np.dot(wid, a) * a
    if np.linalg.norm(u) < 1e-9:
        raise ValueError("width axis must not be parallel to up")
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    lower = -np.linspace(math.pi / 2, 0.0, rings, endpoint=False)[1:]  # pole -> 0
    top_angle = math.asin(shape.top / shape.half_up)
    angles = np.concatenate([lower, np.linspace(0.0, top_angle, rings)])[::-1]
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos, sin = np.cos(phi), np.sin(phi)
    behind = np.zeros(sides, dtype=bool)
    if bulged and shape.back is not None:
        behind = sin * float(v @ _check_finite("back", shape.back, 3)) > 1e-12
    grid = np.empty((len(angles), sides, 3))
    for k, ang in enumerate(angles):
        height = math.sin(ang) * (shape.half_up if ang > 0 else shape.half_down)
        # 1 at the hips, 0 at the cap, and nearly gone by the belt line
        full = (1.0 - (max(height, 0.0) / shape.top) ** 2) ** 2
        power = 2.0 / (2.0 + (shape.back_square - 2.0) * full)
        deep = np.where(behind, 1.0 + (shape.back_scale - 1.0) * full, 1.0)
        across = np.where(behind, np.sign(cos) * np.abs(cos) ** power, cos)
        along = np.where(behind, np.sign(sin) * np.abs(sin) ** power, sin) * deep
        plane = np.outer(across, u * shape.half_width * thickness) + np.outer(
            along, v * shape.half_depth * thickness
        )
        grid[k] = c + a * height + math.cos(ang) * plane
    return closed_grid_mesh(grid, c + a * shape.top, c - a * shape.half_down)


_MAX_RASTER_CANDIDATES = 1_600_000


def rasterize_face_zbuffer(
    sx: Array,
    sy: Array,
    sz: Array,
    faces: NDArray[np.int64],
    width: int,
    height: int,
    face_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.uint64]:
    """Orthographic Z-buffer of ``faces`` into a ``width`` x ``height`` pixel grid.

    Returns one packed ``uint64`` per pixel, ``depth_quantised << 32 | face + 1``,
    so 0 means "no hit" and the low word decodes back into an index into
    ``faces`` (not into ``face_mask``'s selected subset).

    Precondition: ``sx``/``sy``/``sz`` are one value per vertex, ``faces`` holds
    triangle vertex indices, ``face_mask`` (when given) has one entry per face.
    Postcondition: the nearest triangle wins each pixel it covers.

    Candidate pixels are expanded in bounding-box-sized batches so a dense mesh
    cannot allocate an unbounded temporary; ``face_mask`` exists because the
    shaded hand preview culls to the triangles in front of the camera while the
    evidence renderer draws all of them -- one rasteriser serves both.
    """
    sel = (
        np.arange(len(faces), dtype=np.int64)
        if face_mask is None
        else np.flatnonzero(face_mask)
    )
    best = np.zeros(width * height, dtype=np.uint64)
    if len(sel) == 0:
        return best
    f_sel = faces[sel]
    p0x, p0y = sx[f_sel[:, 0]], sy[f_sel[:, 0]]
    p1x, p1y = sx[f_sel[:, 1]], sy[f_sel[:, 1]]
    p2x, p2y = sx[f_sel[:, 2]], sy[f_sel[:, 2]]
    tz0, tz1, tz2 = sz[f_sel[:, 0]], sz[f_sel[:, 1]], sz[f_sel[:, 2]]
    x0i = np.maximum(
        0, np.floor(np.minimum(np.minimum(p0x, p1x), p2x)).astype(np.int64)
    )
    x1i = np.minimum(
        width - 1, np.ceil(np.maximum(np.maximum(p0x, p1x), p2x)).astype(np.int64)
    )
    y0i = np.maximum(
        0, np.floor(np.minimum(np.minimum(p0y, p1y), p2y)).astype(np.int64)
    )
    y1i = np.minimum(
        height - 1, np.ceil(np.maximum(np.maximum(p0y, p1y), p2y)).astype(np.int64)
    )
    bw = np.maximum(0, x1i - x0i + 1)
    bh = np.maximum(0, y1i - y0i + 1)
    area_bb = bw * bh
    area2 = (p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x)
    v_idx = np.flatnonzero((area_bb > 0) & (np.abs(area2) > 1e-6))
    if len(v_idx) == 0:
        return best
    z_min = float(sz.min()) - 1e-4
    z_max = float(sz.max()) + 1e-4
    z_scale = 4294967290.0 / max(z_max - z_min, 1e-6)
    cum = np.cumsum(area_bb[v_idx])
    batch_start = 0
    while batch_start < len(v_idx):
        base_c = 0 if batch_start == 0 else int(cum[batch_start - 1])
        end = max(
            int(np.searchsorted(cum, base_c + _MAX_RASTER_CANDIDATES, side="right")),
            batch_start + 1,
        )
        sl = v_idx[batch_start:end]
        counts = area_bb[sl]
        bws = bw[sl]
        local_idx = np.repeat(sl, counts)
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        rel = np.arange(int(counts.sum()), dtype=np.int64) - starts
        bws_rep = np.repeat(bws, counts)
        gx = np.repeat(x0i[sl], counts) + (rel % bws_rep)
        gy = np.repeat(y0i[sl], counts) + (rel // bws_rep)
        cx, cy = gx.astype(np.float64) + 0.5, gy.astype(np.float64) + 0.5
        t0x, t0y = p0x[local_idx], p0y[local_idx]
        t1x, t1y = p1x[local_idx], p1y[local_idx]
        t2x, t2y = p2x[local_idx], p2y[local_idx]
        inv_a = 1.0 / area2[local_idx]
        w0 = ((t1x - cx) * (t2y - cy) - (t1y - cy) * (t2x - cx)) * inv_a
        w1 = ((t2x - cx) * (t0y - cy) - (t2y - cy) * (t0x - cx)) * inv_a
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if np.any(inside):
            hi = local_idx[inside]
            hz = w0[inside] * tz0[hi] + w1[inside] * tz1[hi] + w2[inside] * tz2[hi]
            hpx = gy[inside] * width + gx[inside]
            zq = np.clip((hz - z_min) * z_scale, 1.0, 4294967294.0).astype(np.uint64)
            packed = (zq << np.uint64(32)) | (sel[hi].astype(np.uint64) + np.uint64(1))
            np.maximum.at(best, hpx, packed)
        batch_start = end
    return best
