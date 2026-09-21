"""Model-free posture metrics from markers, and the same metrics on a model.

Golf set-up posture is described by the tilt of the pelvis and of the trunk
from vertical, and by the angle between them (the spine bend), split into a
sagittal (forward) and a lateral (side) component. Computing the same numbers
straight from the capture markers and from a model's joint centres shows
whether a bend seen in a rendering belongs to the golfer or to the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class SegmentTilt:
    """Tilt of a segment axis from the up direction, and its forward/side parts."""

    total_deg: float
    forward_deg: float  # positive when the top leans toward ``forward``
    lateral_deg: float  # positive when the top leans toward ``right``


@dataclass(frozen=True)
class SpineBend:
    """Angle between the pelvis axis and the trunk axis, split by plane."""

    total_deg: float
    forward_deg: float  # trunk more forward than pelvis: positive (flexion)
    lateral_deg: float  # trunk more to the right than pelvis: positive


def _unit(v: Array, name: str) -> Array:
    u = np.asarray(v, dtype=float)
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) < 1e-12:
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return u / np.linalg.norm(u)


def segment_tilt(axis: Array, up: Array, forward: Array, right: Array) -> SegmentTilt:
    """Tilt of ``axis`` (bottom to top) from ``up`` and its components.

    Precondition: ``up``, ``forward``, ``right`` are finite nonzero and
    ``forward`` and ``right`` are orthogonal to ``up`` after normalisation
    (checked to 1e-6). Postcondition: total is in [0, 180] degrees.
    """
    a, u = _unit(axis, "axis"), _unit(up, "up")
    f, r = _unit(forward, "forward"), _unit(right, "right")
    if abs(f @ u) > 1e-6 or abs(r @ u) > 1e-6:
        raise ValueError("forward and right must be orthogonal to up")
    total = math.degrees(math.acos(float(np.clip(a @ u, -1.0, 1.0))))
    return SegmentTilt(
        total_deg=total,
        forward_deg=math.degrees(math.atan2(float(a @ f), float(a @ u))),
        lateral_deg=math.degrees(math.atan2(float(a @ r), float(a @ u))),
    )


def spine_bend(
    pelvis_axis: Array, trunk_axis: Array, up: Array, forward: Array, right: Array
) -> SpineBend:
    """Bend between the pelvis and trunk axes (both bottom to top)."""
    p = segment_tilt(pelvis_axis, up, forward, right)
    t = segment_tilt(trunk_axis, up, forward, right)
    a, b = _unit(pelvis_axis, "pelvis_axis"), _unit(trunk_axis, "trunk_axis")
    return SpineBend(
        total_deg=math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0)))),
        forward_deg=t.forward_deg - p.forward_deg,
        lateral_deg=t.lateral_deg - p.lateral_deg,
    )


def plane_normal(points: Array, toward: Array) -> Array:
    """Unit normal of the least-squares plane through >= 3 points, oriented toward ``toward``."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 3 or not np.isfinite(x).all():
        raise ValueError("Plane fit needs at least three finite 3-D points")
    centred = x - x.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][2]
    if normal @ _unit(toward, "toward") < 0:
        normal = -normal
    return normal


def elbow_pit_direction(
    shoulder: Array, elbow: Array, wrist: Array, min_flexion_deg: float = 5.0
) -> Array | None:
    """Unit direction the elbow pit faces, from three arm markers.

    The forearm folds toward the pit, so the pit direction is the component
    of the forearm direction perpendicular to the upper-arm direction. With
    the elbow straighter than ``min_flexion_deg`` the fold is not observable
    and ``None`` is returned. Precondition: finite 3-vectors, upper arm and
    forearm of nonzero length.
    """
    upper = _unit(np.asarray(elbow, float) - np.asarray(shoulder, float), "upper arm")
    fore = _unit(np.asarray(wrist, float) - np.asarray(elbow, float), "forearm")
    flexion = np.degrees(np.arccos(np.clip(upper @ fore, -1.0, 1.0)))
    if flexion < min_flexion_deg:
        return None
    pit = fore - (fore @ upper) * upper
    return pit / np.linalg.norm(pit)
