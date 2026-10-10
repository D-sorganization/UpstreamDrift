"""Anatomical constraints on calibrated leg-marker placements (OSV-6, #11737).

Each leg segment carries one lateral marker (``*KneeOut`` on the femur,
``*AnkleOut`` on the tibia) and the foot two forefoot markers on ``calcn``.
When the alternating calibration places those markers freely, a segment's
twist about its long axis (body ``y``) trades exactly against the marker's
azimuth around that axis, and the forefoot pair's stagger trades against foot
yaw. Hip rotation and foot yaw then become gauge modes held only by the
offset prior; on the capture-A driver the calibrated left forefoot line ends
28 degrees off square and the lead foot 15-32 degrees too far out.

The constraint keeps each lateral marker on its seed's azimuth (radius and
axial position stay free) and the forefoot line perpendicular to the foot
long axis (OSV-4 measured the capture forefoot line square within a few
degrees). The placement step is a weighted mean, so the constrained
least-squares placement is the orthogonal projection of the free one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "LATERAL_MARKERS",
    "anatomical_leg_constraint",
    "preserve_azimuth",
    "square_forefoot",
]

Offset = tuple[float, float, float]
Offsets = dict[str, tuple[str, Offset]]

#: Single lateral markers whose azimuth about the segment axis is pinned.
LATERAL_MARKERS: tuple[str, ...] = ("RKneeOut", "LKneeOut", "RAnkleOut", "LAnkleOut")
_MIN_SEED_RADIUS_M = 1.0e-3


def preserve_azimuth(offset: Sequence[float], seed: Sequence[float]) -> Offset:
    """Project ``offset`` onto the half-plane through the segment axis and ``seed``.

    Body frames follow OpenSim (``x`` forward, ``y`` along the segment, ``z``
    right). The axial ``y`` is kept; the ``(x, z)`` part is projected onto the
    seed's unit direction and clamped at the axis (a marker cannot cross to
    the other side). Raises ``ValueError`` if the seed lies on the axis.
    Postcondition: the result has the seed's azimuth or zero radius.
    """
    p = np.asarray(offset, dtype=float)
    s = np.asarray(seed, dtype=float)
    if p.shape != (3,) or s.shape != (3,):
        raise ValueError("offset and seed must be 3-vectors")
    if not (np.isfinite(p).all() and np.isfinite(s).all()):
        raise ValueError("offset and seed must be finite")
    radial = np.array([s[0], s[2]])
    norm = float(np.linalg.norm(radial))
    if norm < _MIN_SEED_RADIUS_M:
        raise ValueError("seed has no azimuth: it lies on the segment axis")
    unit = radial / norm
    length = max(0.0, float(unit @ np.array([p[0], p[2]])))
    return (float(length * unit[0]), float(p[1]), float(length * unit[1]))


def square_forefoot(
    offsets: Mapping[str, tuple[str, Offset]],
    side: str,
    mid_z: float | None = None,
) -> Offsets:
    """Copy of ``offsets`` whose ``<side>ToeIn``/``<side>ToeOut`` share one ``x``.

    The shared ``x`` is their mean, which is the equal-weight least-squares
    placement under the constraint. With ``mid_z`` the pair is also shifted
    sideways (spacing kept) so its midpoint sits at ``z = mid_z``: a sideways
    shift of the pair is foot yaw about the ankle in disguise. Raises
    ``ValueError`` if either marker is missing or ``mid_z`` is not finite.
    """
    inner, outer = f"{side}ToeIn", f"{side}ToeOut"
    for label in (inner, outer):
        if label not in offsets:
            raise ValueError(f"offsets must contain {label}")
    if mid_z is not None and not np.isfinite(mid_z):
        raise ValueError("mid_z must be finite")
    mean_x = 0.5 * (offsets[inner][1][0] + offsets[outer][1][0])
    shift_z = (
        0.0
        if mid_z is None
        else mid_z - 0.5 * (offsets[inner][1][2] + offsets[outer][1][2])
    )
    out: Offsets = dict(offsets)
    for label in (inner, outer):
        body, (_, y, z) = offsets[label]
        out[label] = (body, (float(mean_x), float(y), float(z + shift_z)))
    return out


def _seed_mid_z(seeds: Mapping[str, tuple[str, Sequence[float]]], side: str) -> float:
    return 0.5 * (
        float(seeds[f"{side}ToeIn"][1][2]) + float(seeds[f"{side}ToeOut"][1][2])
    )


def anatomical_leg_constraint(
    seeds: Mapping[str, tuple[str, Sequence[float]]],
) -> Callable[[Offsets], Offsets]:
    """Constraint for ``calibrate_marker_offsets(constrain=...)``.

    Lateral markers keep their seed azimuth; each forefoot pair present is
    squared; every other label passes through unchanged. Raises
    ``ValueError`` when a constrained marker sits on a different body than its
    seed.
    """
    for label in LATERAL_MARKERS:
        if label in seeds:
            preserve_azimuth((0.0, 0.0, 0.0), seeds[label][1])  # validates the seed

    def constrain(offsets: Offsets) -> Offsets:
        out: Offsets = dict(offsets)
        for label in LATERAL_MARKERS:
            if label not in out or label not in seeds:
                continue
            body, offset = out[label]
            if body != seeds[label][0]:
                raise ValueError(
                    f"{label} is on body {body}, its seed on {seeds[label][0]}"
                )
            out[label] = (body, preserve_azimuth(offset, seeds[label][1]))
        for side in ("R", "L"):
            pair = (f"{side}ToeIn", f"{side}ToeOut")
            if all(label in out for label in pair):
                pinned = all(label in seeds for label in pair)
                mid_z = _seed_mid_z(seeds, side) if pinned else None
                out = square_forefoot(out, side, mid_z=mid_z)
        return out

    return constrain
