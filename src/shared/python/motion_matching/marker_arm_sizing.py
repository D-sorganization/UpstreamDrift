"""Arm scale from a capture's shoulder, elbow and wrist marker distances (#15258).

The three markers sit on the skin outside the joints they name, so the
marker-to-marker distance is longer than the joint-to-joint length a model
needs. The geometry assumed here, every offset a setting:

* the shoulder marker sits ``shoulder_marker_above_joint_m`` beyond the
  shoulder joint centre along the upper arm's line;
* the elbow marker sits ``elbow_marker_from_joint_m`` to the side of the elbow
  joint centre, at right angles to both arm segments;
* the wrist marker sits ``wrist_marker_from_joint_m`` off the wrist joint
  centre, at right angles to the forearm and to the elbow marker's offset.

With one wrist marker on each forearm the capture cannot fix these offsets
itself, so the result carries the range the scale takes when each offset is
wrong by ``offset_doubt_m``. This is an estimate with stated assumptions, not
a measurement of the joint centres.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.anthropometry import segment_parameters

Array = NDArray[np.float64]
SIDES = ("L", "R")


@dataclass(frozen=True)
class ArmMarkerSettings:
    """Marker names (without the ``L``/``R`` prefix) and skin-to-joint offsets."""

    shoulder_marker: str = "ShoulderTop"
    elbow_marker: str = "ElbowOut"
    wrist_marker: str = "WristTop"
    shoulder_marker_above_joint_m: float = 0.04
    elbow_marker_from_joint_m: float = 0.035
    wrist_marker_from_joint_m: float = 0.025
    #: How far each offset may be wrong; sets ``ArmSizing.arm_scale_range``.
    offset_doubt_m: float = 0.015

    def __post_init__(self) -> None:
        offsets = (
            self.shoulder_marker_above_joint_m,
            self.elbow_marker_from_joint_m,
            self.wrist_marker_from_joint_m,
            self.offset_doubt_m,
        )
        if any(not np.isfinite(v) or v < 0.0 for v in offsets):
            raise ValueError("every marker offset must be finite and non-negative")


@dataclass(frozen=True)
class ArmSizing:
    """Joint-to-joint arm lengths per side and the scale against de Leva.

    ``marker_spread_m`` is the standard deviation over frames of each marker
    distance (upper arm, forearm): a large value means the markers do not
    ride one rigid segment pair.
    """

    upper_arm_m: dict[str, float]
    forearm_m: dict[str, float]
    arm_scale: float
    arm_scale_range: tuple[float, float]
    marker_spread_m: dict[str, tuple[float, float]]
    frames_used: int


def _lengths(
    upper_d: float, fore_d: float, s: ArmMarkerSettings
) -> tuple[float, float]:
    elbow, wrist = s.elbow_marker_from_joint_m, s.wrist_marker_from_joint_m
    upper_sq = upper_d**2 - elbow**2
    fore_sq = fore_d**2 - elbow**2 - wrist**2
    if upper_sq <= 0.0 or fore_sq <= 0.0:
        raise ValueError("a marker offset is longer than the marker distance")
    return float(np.sqrt(upper_sq)) - s.shoulder_marker_above_joint_m, float(
        np.sqrt(fore_sq)
    )


def _scale_range(
    distances: dict[str, tuple[float, float]], s: ArmMarkerSettings, reference: float
) -> tuple[float, float]:
    names = (
        "shoulder_marker_above_joint_m",
        "elbow_marker_from_joint_m",
        "wrist_marker_from_joint_m",
    )
    scales = []
    for signs in product((-1.0, 1.0), repeat=len(names)):
        moved = replace(
            s,
            **{
                name: max(0.0, getattr(s, name) + sign * s.offset_doubt_m)
                for name, sign in zip(names, signs, strict=True)
            },
        )
        total = sum(sum(_lengths(*distances[side], moved)) for side in SIDES)
        scales.append(total / (len(SIDES) * reference))
    return min(scales), max(scales)


def arm_scale_from_markers(
    points_m: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    stature_m: float,
    mass_kg: float,
    settings: ArmMarkerSettings | None = None,
) -> ArmSizing:
    """Arm scale (upper arm plus forearm against de Leva) from marker distances.

    Uses the median distance over the frames where all six arm markers are
    valid. Preconditions: ``points_m`` is ``(frames, markers, 3)`` and
    ``valid`` ``(frames, markers)`` in ``labels`` order; the six arm markers
    are present; at least one frame has them all (``ValueError`` otherwise).
    Postcondition: ``arm_scale`` lies inside ``arm_scale_range``.
    """
    s = settings or ArmMarkerSettings()
    points, mask = np.asarray(points_m, dtype=float), np.asarray(valid, dtype=bool)
    if points.shape != (*mask.shape, 3) or mask.shape[1:] != (len(labels),):
        raise ValueError("points, valid and labels disagree in shape")
    needed = [
        side + name
        for side in SIDES
        for name in (s.shoulder_marker, s.elbow_marker, s.wrist_marker)
    ]
    missing = [name for name in needed if name not in labels]
    if missing:
        raise ValueError(f"capture lacks arm markers: {missing}")
    index = [list(labels).index(name) for name in needed]
    frames = mask[:, index].all(axis=1)
    if not frames.any():
        raise ValueError("no frame has every arm marker valid")
    arm = points[frames][:, index].reshape(-1, len(SIDES), 3, 3)
    upper_d = np.linalg.norm(arm[:, :, 0] - arm[:, :, 1], axis=-1)
    fore_d = np.linalg.norm(arm[:, :, 1] - arm[:, :, 2], axis=-1)
    distances = {
        side: (float(np.median(upper_d[:, k])), float(np.median(fore_d[:, k])))
        for k, side in enumerate(SIDES)
    }
    reference = sum(
        segment_parameters(stature_m, mass_kg, name).length_m
        for name in ("upper_arm", "forearm")
    )
    lengths = {side: _lengths(*distances[side], s) for side in SIDES}
    return ArmSizing(
        upper_arm_m={side: lengths[side][0] for side in SIDES},
        forearm_m={side: lengths[side][1] for side in SIDES},
        arm_scale=sum(sum(pair) for pair in lengths.values())
        / (len(SIDES) * reference),
        arm_scale_range=_scale_range(distances, s, reference),
        marker_spread_m={
            side: (float(upper_d[:, k].std()), float(fore_d[:, k].std()))
            for k, side in enumerate(SIDES)
        },
        frames_used=int(frames.sum()),
    )
