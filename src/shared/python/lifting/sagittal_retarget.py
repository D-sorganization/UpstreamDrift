"""Retarget a sagittal 3-link lift solution onto full-body landmarks (LIFT-5).

The Movement Optimizer (Tools ``movement_optimizer``) solves a lift as a
planar shank/thigh/trunk chain: each segment's absolute angle from vertical,
positive when the distal end moves forward.  A full-body model has its own
segment lengths and left/right geometry, so its targets are not the
optimizer's joint positions.  Instead each segment of the model's measured
standing pose is rotated rigidly about the lateral axis until its sagittal
angle equals the optimizer's (#11745):

* the ankles stay where they stand (feet planted);
* the knees ride the shank rotation about the ankle midpoint;
* the hips ride the thigh rotation about the moving knee midpoint;
* trunk landmarks (shoulders, bar) ride the trunk rotation about the moving
  hip midpoint.

Lateral offsets of each landmark from its segment midpoint are preserved, so
at the standing angles the targets reproduce the standing landmarks exactly
and segment lengths never change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

SEGMENTS: tuple[str, str, str] = ("shank", "thigh", "trunk")
LANDMARK_LABELS: tuple[str, ...] = (
    "ankle_l",
    "ankle_r",
    "knee_l",
    "knee_r",
    "hip_l",
    "hip_r",
    "shoulder_l",
    "shoulder_r",
    "bar",
)
_TRUNK_LABELS: tuple[str, ...] = ("shoulder_l", "shoulder_r", "bar")


def _finite(values: Any, name: str) -> Array:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class SagittalFrame:
    """World directions of the sagittal plane: ``forward`` and ``up`` (unit).

    Postcondition: ``forward`` and ``up`` are orthonormal; ``left`` completes a
    right-handed frame.
    """

    forward: tuple[float, float, float]
    up: tuple[float, float, float]

    def __post_init__(self) -> None:
        axes = []
        for name in ("forward", "up"):
            vector = _finite(getattr(self, name), name)
            norm = float(np.linalg.norm(vector))
            if vector.shape != (3,) or norm < 1e-12:
                raise ValueError(f"{name} must be a finite nonzero 3-vector")
            axes.append(vector / norm)
        if abs(float(axes[0] @ axes[1])) > 1e-9:
            raise ValueError("forward and up must be orthogonal")
        object.__setattr__(self, "forward", tuple(map(float, axes[0])))
        object.__setattr__(self, "up", tuple(map(float, axes[1])))

    @property
    def left(self) -> Array:
        return np.cross(np.asarray(self.up), np.asarray(self.forward))


@dataclass(frozen=True)
class SagittalSolution:
    """A planar lift solution sampled on a time grid.

    Attributes:
        time_s: Strictly increasing sample times (N,).
        segment_angles_rad: Shank, thigh, trunk angles from vertical (N, 3).
        bar_m: Bar (forward, up) relative to the ankle, metres (N, 2).
        source: Provenance of the solution (optimizer, version, parameters).
    """

    time_s: Array
    segment_angles_rad: Array
    bar_m: Array
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, Mapping):
            raise TypeError("source must be a mapping of provenance fields")
        time = _finite(self.time_s, "time_s")
        angles = _finite(self.segment_angles_rad, "segment_angles_rad")
        bar = _finite(self.bar_m, "bar_m")
        if time.ndim != 1 or time.size < 2:
            raise ValueError("time_s must hold at least two samples")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if angles.shape != (time.size, len(SEGMENTS)):
            raise ValueError("segment_angles_rad must be (N, 3) matching time_s")
        if bar.shape != (time.size, 2):
            raise ValueError("bar_m must be (N, 2) matching time_s")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "segment_angles_rad", angles)
        object.__setattr__(self, "bar_m", bar)
        object.__setattr__(self, "source", dict(self.source))

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable form (exact float round trip through ``repr``)."""
        return {
            "time_s": self.time_s.tolist(),
            "segment_angles_rad": self.segment_angles_rad.tolist(),
            "bar_m": self.bar_m.tolist(),
            "source": dict(self.source),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SagittalSolution:
        """Inverse of :meth:`to_record`."""
        return cls(
            time_s=np.asarray(record["time_s"], dtype=float),
            segment_angles_rad=np.asarray(record["segment_angles_rad"], dtype=float),
            bar_m=np.asarray(record["bar_m"], dtype=float),
            source=dict(record.get("source", {})),
        )


def rotate_sagittal(vector: Array, angle_rad: float, frame: SagittalFrame) -> Array:
    """Rotate ``vector`` about the lateral axis; positive tips ``up`` forward."""
    v = np.asarray(vector, dtype=float)
    forward, up = np.asarray(frame.forward), np.asarray(frame.up)
    f, u = float(v @ forward), float(v @ up)
    lateral = v - f * forward - u * up
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return lateral + (f * c + u * s) * forward + (u * c - f * s) * up


def _require(landmarks: Mapping[str, Any]) -> dict[str, Array]:
    missing = [label for label in LANDMARK_LABELS if label not in landmarks]
    if missing:
        raise ValueError(f"landmarks missing {missing}")
    points = {label: _finite(landmarks[label], label) for label in LANDMARK_LABELS}
    if any(point.shape != (3,) for point in points.values()):
        raise ValueError("every landmark must be a 3-vector")
    return points


def _midpoint(points: Mapping[str, Array], stem: str) -> Array:
    return 0.5 * (points[f"{stem}_l"] + points[f"{stem}_r"])


def segment_angles(landmarks: Mapping[str, Any], frame: SagittalFrame) -> Array:
    """Shank, thigh and trunk sagittal angles of a landmark set (radians)."""
    points = _require(landmarks)
    forward, up = np.asarray(frame.forward), np.asarray(frame.up)
    chain = [
        _midpoint(points, "ankle"),
        _midpoint(points, "knee"),
        _midpoint(points, "hip"),
        _midpoint(points, "shoulder"),
    ]
    return np.array(
        [
            np.arctan2(float((b - a) @ forward), float((b - a) @ up))
            for a, b in zip(chain[:-1], chain[1:], strict=True)
        ]
    )


def retarget_landmarks(
    solution: SagittalSolution,
    standing: Mapping[str, Any],
    frame: SagittalFrame,
) -> Array:
    """Full-body landmark targets (N, len(LANDMARK_LABELS), 3) for ``solution``.

    Args:
        solution: Planar lift solution.
        standing: Model landmarks at its standing pose (world, metres).
        frame: Sagittal plane of the standing pose.

    Postconditions: the ankles equal their standing positions at every frame;
    every frame's :func:`segment_angles` equal the solution's angles.
    """
    points = _require(standing)
    ankle, knee = _midpoint(points, "ankle"), _midpoint(points, "knee")
    hip = _midpoint(points, "hip")
    deltas = solution.segment_angles_rad - segment_angles(points, frame)
    frames = []
    for d_shank, d_thigh, d_trunk in deltas:
        knee_k = ankle + rotate_sagittal(knee - ankle, d_shank, frame)
        hip_k = knee_k + rotate_sagittal(hip - knee, d_thigh, frame)
        target = {f"ankle_{s}": points[f"ankle_{s}"] for s in "lr"} | {
            f"knee_{s}": knee_k + points[f"knee_{s}"] - knee for s in "lr"
        }
        target |= {f"hip_{s}": hip_k + points[f"hip_{s}"] - hip for s in "lr"}
        for label in _TRUNK_LABELS:
            target[label] = hip_k + rotate_sagittal(points[label] - hip, d_trunk, frame)
        frames.append([target[label] for label in LANDMARK_LABELS])
    return np.asarray(frames, dtype=float)
