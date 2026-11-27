"""Markers re-placed on their segments from a whole solved swing (#15267).

The matching pipeline places a marker on its segment from one posture. Any
error in that posture is then carried by the marker for the whole swing.
:func:`place_markers_over_swing` takes the segment poses of a solved swing and
moves each marker to its mean position in its segment's frame over every frame
it was seen in: the placement that fits those poses best, in the least-squares
sense.

Kinematics only. The poses come from a fit that used the old placements, so a
caller solves the swing again and may repeat. Nothing here says a placement is
anatomical; the distance each marker moved is reported and can be limited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Offset = tuple[float, float, float]


@dataclass(frozen=True)
class PlacementSettings:
    """How markers are re-placed.

    ``min_frames``: a marker seen in fewer frames keeps its placement.
    ``max_move_m``: the furthest a marker may move from its placement; a
    longer move is cut to this length along its own direction. ``None`` sets
    no limit.
    """

    min_frames: int = 10
    max_move_m: float | None = 0.05

    def __post_init__(self) -> None:
        """Reject a frame count below 1 or a negative move limit."""
        if self.min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        if self.max_move_m is not None and not self.max_move_m >= 0.0:
            raise ValueError("max_move_m must be non-negative or None")


@dataclass(frozen=True)
class MarkerPlacement:
    """One marker's placement on its segment and what the move did.

    ``offset_m`` is in the segment's frame. The two RMS values are the
    marker's distance from its measured positions, in the segment's frame,
    with the old and the new placement at the same segment poses.
    ``replaced`` is false for a marker seen in too few frames; ``limited`` is
    true when the move was cut to ``max_move_m``.
    """

    body: str
    offset_m: Offset
    moved_m: float
    rms_before_m: float
    rms_after_m: float
    frames_used: int
    replaced: bool
    limited: bool


def _rms(distances: Array) -> float:
    """Root-mean-square of the distances, or NaN when there are none."""
    return float(np.sqrt(np.mean(distances**2))) if len(distances) else float("nan")


def _checked(
    points_m: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    placements: Mapping[str, tuple[str, Sequence[float]]],
    rotations: Mapping[str, Array],
    origins_m: Mapping[str, Array],
) -> int:
    """The number of frames, after the preconditions of the placement."""
    frames = len(points_m)
    if points_m.shape != (frames, len(labels), 3) or valid.shape != points_m.shape[:2]:
        raise ValueError(
            "points_m must have shape (frames, markers, 3) and valid "
            "(frames, markers), with one marker per label"
        )
    for label in labels:
        if label not in placements:
            raise ValueError(f"marker {label!r} has no placement")
        body = placements[label][0]
        if body not in rotations or body not in origins_m:
            raise ValueError(f"segment {body!r} of marker {label!r} has no poses")
        if np.shape(rotations[body]) != (frames, 3, 3) or np.shape(origins_m[body]) != (
            frames,
            3,
        ):
            raise ValueError(f"segment {body!r} must have one pose per {frames} frames")
    return frames


def _place(
    seen_m: Array, body: str, start: Array, settings: PlacementSettings
) -> MarkerPlacement:
    """Placement of one marker from its positions in the segment's frame."""
    before = _rms(np.linalg.norm(seen_m - start, axis=1))
    if len(seen_m) < settings.min_frames:
        offset = (float(start[0]), float(start[1]), float(start[2]))
        return MarkerPlacement(
            body, offset, 0.0, before, before, len(seen_m), False, False
        )
    move = seen_m.mean(axis=0) - start
    length = float(np.linalg.norm(move))
    limit = settings.max_move_m
    limited = limit is not None and length > limit
    if limited and limit is not None:
        move, length = move * (limit / length), limit
    placed = start + move
    return MarkerPlacement(
        body=body,
        offset_m=(float(placed[0]), float(placed[1]), float(placed[2])),
        moved_m=length,
        rms_before_m=before,
        rms_after_m=_rms(np.linalg.norm(seen_m - placed, axis=1)),
        frames_used=len(seen_m),
        replaced=True,
        limited=limited,
    )


def place_markers_over_swing(
    points_m: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    placements: Mapping[str, tuple[str, Sequence[float]]],
    rotations: Mapping[str, Array],
    origins_m: Mapping[str, Array],
    settings: PlacementSettings | None = None,
) -> dict[str, MarkerPlacement]:
    """Each marker's best placement on its segment over a solved swing.

    ``points_m`` are the measured marker positions, world frame, one row per
    frame and one column per label; ``valid`` marks the frames each marker was
    seen in. ``placements`` gives each marker's segment and its current offset
    in that segment's frame. ``rotations`` and ``origins_m`` give each
    segment's world rotation matrix and origin in every frame of the solved
    swing.

    Preconditions: ``points_m`` is (frames, markers, 3) and ``valid`` (frames,
    markers), one marker per label; every label has a placement; every
    segment named has one rotation and one origin per frame (``ValueError``
    otherwise).
    Postconditions: one entry per label; no marker moves further than
    ``settings.max_move_m``; a marker's RMS never rises unless its move was
    limited; a marker seen in fewer than ``settings.min_frames`` frames keeps
    its offset.
    """
    chosen = settings or PlacementSettings()
    points = np.asarray(points_m, dtype=float)
    seen = np.asarray(valid, dtype=bool)
    _checked(points, seen, labels, placements, rotations, origins_m)
    placed: dict[str, MarkerPlacement] = {}
    for column, label in enumerate(labels):
        body, offset = placements[label]
        rows = seen[:, column]
        turn = np.asarray(rotations[body], dtype=float)[rows]
        origin = np.asarray(origins_m[body], dtype=float)[rows]
        # World point into the segment's frame: R^T (p - x), frame by frame.
        local = np.einsum("fji,fj->fi", turn, points[rows, column] - origin)
        placed[label] = _place(local, body, np.asarray(offset, dtype=float), chosen)
    return placed
