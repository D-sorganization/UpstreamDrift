"""Cross-sections of the MakeHuman skin along a bone axis (build-time only).

The rest pose of the base mesh is an A-pose, so a body segment is not sliced
along a world axis: each segment has its own axis between two skeleton joints,
and the planes are square to that axis (#15299). Which skin belongs to a
segment is decided by MakeHuman's own skin weights, summed over the bones the
settings list for it, so a plane through the upper arm does not pick up the
chest beside it. The result does not change when the whole rig is turned.

A section is measured the way the procedural lofts are written
(``model_appearance.torso_profile``): half width, depth in front of and
behind the axis, the girth a tape would read (convex hull) and the
superellipse exponent that fits its outline best. Units are the rig's.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from mh_rig_io import Array, MakeHumanRig, dense_weights
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

JOINT_ENDS = ("head", "tail")


@dataclass(frozen=True)
class SliceFrame:
    """Axis of a segment and the two directions its sections are measured in."""

    origin: Array
    axis: Array  # unit, start joint to end joint
    length: float
    width_axis: Array  # unit, square to the axis and to the front
    depth_axis: Array  # unit, the front made square to the axis


@dataclass(frozen=True)
class Section:
    """One cross-section: ``along`` is a fraction of the segment length."""

    along: float
    half_width: float
    width_centre: float  # middle of the section, off the axis along the width
    depth_front: float  # skin in front of the axis
    depth_back: float  # skin behind the axis
    girth: float  # convex hull perimeter
    squareness: float  # superellipse exponent (2 = ellipse)


@dataclass(frozen=True)
class SegmentTable:
    """Sections of one segment, start joint first."""

    length: float
    sections: tuple[Section, ...]

    @property
    def belly_along(self) -> float:
        """Fraction of the length at which the girth is largest."""
        return max(self.sections, key=lambda s: s.girth).along


def joint_point(rig: MakeHumanRig, reference: str) -> Array:
    """Position of ``"<bone>/head"`` or ``"<bone>/tail"``."""
    bone, _, end = reference.rpartition("/")
    if end not in JOINT_ENDS or not bone:
        raise ValueError(f"joint reference must be '<bone>/head|tail': {reference!r}")
    return rig.bone_head(bone) if end == "head" else rig.bone_tail(bone)


def expand_bones(rig: MakeHumanRig, patterns: Sequence[str]) -> list[str]:
    """Weighted bones matching the shell ``patterns``; each must match one."""
    bones: list[str] = []
    for pattern in patterns:
        found = sorted(fnmatch.filter(rig.weights, pattern))
        if not found:
            raise ValueError(f"no weighted MakeHuman bone matches {pattern!r}")
        bones += [bone for bone in found if bone not in bones]
    return bones


def skin_volume(rig: MakeHumanRig) -> float:
    """Enclosed volume of the body skin, rig units cubed."""
    q = rig.body_quads
    tri = rig.vertices[np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])]
    return abs(
        float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6
    )


def slice_frame(start: Array, end: Array, front: Array) -> SliceFrame:
    """Frame of the segment from ``start`` to ``end`` whose front is ``front``.

    Raises ``ValueError`` for a segment of no length or a front along it.
    """
    axis = np.asarray(end, float) - np.asarray(start, float)
    length = float(np.linalg.norm(axis))
    if not (math.isfinite(length) and length > 1e-9):
        raise ValueError("segment has no length")
    axis = axis / length
    depth = np.asarray(front, float) - (np.asarray(front, float) @ axis) * axis
    if not np.linalg.norm(depth) > 1e-9:
        raise ValueError("segment front must not lie along its axis")
    depth = depth / np.linalg.norm(depth)
    return SliceFrame(
        np.asarray(start, float), axis, length, np.cross(depth, axis), depth
    )


def weighted_vertices(
    rig: MakeHumanRig, bones: Sequence[str], min_weight: float
) -> NDArray[np.bool_]:
    """Vertices whose skin weights on ``bones`` sum to ``min_weight`` or more."""
    if not 0.0 < min_weight <= 1.0:
        raise ValueError("min_weight must be in (0, 1]")
    missing = [bone for bone in bones if bone not in rig.weights]
    if missing or not bones:
        raise ValueError(f"MakeHuman rig has no bone weights for {missing}")
    keep: NDArray[np.bool_] = dense_weights(rig, tuple(bones)).sum(axis=1) >= min_weight
    return keep


def skin_edges(quads: NDArray[np.int64], keep: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Unique edges of the quads whose corners are all kept."""
    kept = quads[keep[quads].all(axis=1)]
    edges = np.concatenate([kept[:, [k, (k + 1) % 4]] for k in range(4)])
    return np.unique(np.sort(edges, axis=1), axis=0)


def section_points(
    vertices: Array, edges: NDArray[np.int64], frame: SliceFrame, along: float
) -> Array:
    """``(K, 2)`` width and depth of where ``edges`` cross the plane at ``along``."""
    local = np.asarray(vertices, float) - frame.origin
    height = local @ frame.axis - along * frame.length
    a, b = height[edges[:, 0]], height[edges[:, 1]]
    crossing = (a * b <= 0.0) & (a != b)
    first, second = edges[crossing, 0], edges[crossing, 1]
    share = (a[crossing] / (a[crossing] - b[crossing]))[:, None]
    points = local[first] + share * (local[second] - local[first])
    return np.column_stack([points @ frame.width_axis, points @ frame.depth_axis])


def _outline(points: Array, samples: int) -> tuple[Array, float]:
    """Convex outline resampled evenly along its length, and that length."""
    ring = points[ConvexHull(points).vertices]
    closed = np.vstack([ring, ring[:1]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    at = np.linspace(0.0, travelled[-1], samples, endpoint=False)
    even = np.column_stack([np.interp(at, travelled, closed[:, k]) for k in (0, 1)])
    return even, float(travelled[-1])


def measure_section(points: Array, along: float, fit: Mapping[str, Any]) -> Section:
    """Sizes of one section from its ``(K, 2)`` width and depth points.

    ``fit``: ``hull_samples`` (points on the outline) and ``squareness_bounds``
    (range searched for the superellipse exponent). The exponent is the one
    whose superellipse, on the section's own half width and half depth about
    its middle, lies closest to the convex outline.
    """
    outline, girth = _outline(points, int(fit["hull_samples"]))
    low, high = outline.min(axis=0), outline.max(axis=0)
    centre, half = 0.5 * (low + high), 0.5 * (high - low)
    unit = np.abs(outline - centre) / half

    def miss(exponent: float) -> float:
        radius = (unit[:, 0] ** exponent + unit[:, 1] ** exponent) ** (1.0 / exponent)
        return float(np.mean((radius - 1.0) ** 2))

    lowest, highest = (float(x) for x in fit["squareness_bounds"])
    best = minimize_scalar(miss, bounds=(lowest, highest), method="bounded")
    return Section(
        along=float(along),
        half_width=float(half[0]),
        width_centre=float(centre[0]),
        depth_front=float(high[1]),
        depth_back=float(-low[1]),
        girth=girth,
        squareness=float(best.x),
    )


def segment_table(
    rig: MakeHumanRig, segment: Mapping[str, Any], front: Array, fit: Mapping[str, Any]
) -> SegmentTable:
    """Sections of one segment of the settings' bone-to-segment table.

    ``segment``: ``start`` and ``end`` (joint references), ``slice_bones`` and
    ``min_weight`` (which skin is cut), ``along`` (first and last station, as
    fractions of the length), ``stations`` (how many) and ``reach`` (skin
    farther from the axis than this many segment lengths is ignored). Raises
    ``ValueError`` if a plane does not cut the segment's skin.
    """
    stations = int(segment["stations"])
    if stations < 2:
        raise ValueError("a segment needs at least two stations")
    start, end = (joint_point(rig, segment[key]) for key in ("start", "end"))
    frame = slice_frame(start, end, front)
    keep = weighted_vertices(rig, segment["slice_bones"], float(segment["min_weight"]))
    edges = skin_edges(rig.body_quads, keep)
    reach = float(segment["reach"]) * frame.length
    sections = []
    first, last = (float(x) for x in segment["along"])
    for along in np.linspace(first, last, stations):
        points = section_points(rig.vertices, edges, frame, float(along))
        points = points[np.linalg.norm(points, axis=1) <= reach]
        if len(points) < 3:
            raise ValueError(f"the plane at {along:.3f} does not cut the segment")
        sections.append(measure_section(points, float(along), fit))
    return SegmentTable(frame.length, tuple(sections))
