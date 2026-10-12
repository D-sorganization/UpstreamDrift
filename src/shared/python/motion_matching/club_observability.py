"""Club face-angle observability audit (#10382, MS-108).

Face angle depends on the club's axial rotation (roll about the shaft), which
only a rigid, non-collinear head-marker triad can fix. This module classifies
a capture's club marker configuration as observable or not, and reports where
the full triad is missing (gaps), so the output policy can keep face angle
unavailable instead of inventing it.

The thresholds are those of :mod:`club_face_target`
(``MIN_TRIAD_SPREAD_M`` for collinearity and ``MAX_TRIAD_SHAPE_RESIDUAL_M``
for the rigid-shape check); this module adds no tolerance of its own.

Unobservable reasons (``FaceObservability.reason``):

* ``REASON_TOO_FEW_MARKERS``: fewer than three head markers are attached.
* ``REASON_DEGENERATE_TRIAD``: the attached offsets are collinear or coincident.
* ``REASON_MARKERS_NOT_CAPTURED``: a head marker has no capture column.
* ``REASON_NO_OBSERVED_FRAME``: no frame has the full triad valid and finite.
* ``REASON_SHAPE_MISMATCH``: the offsets do not describe the captured triad.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching import club_face_target as cft

__all__ = [
    "REASON_DEGENERATE_TRIAD",
    "REASON_MARKERS_NOT_CAPTURED",
    "REASON_NO_OBSERVED_FRAME",
    "REASON_OBSERVABLE",
    "REASON_SHAPE_MISMATCH",
    "REASON_TOO_FEW_MARKERS",
    "FaceObservability",
    "audit_face_observability",
]

Array = NDArray[np.float64]

REASON_OBSERVABLE = "observable"
REASON_TOO_FEW_MARKERS = "fewer than three club head markers"
REASON_DEGENERATE_TRIAD = "club head markers are collinear or coincident"
REASON_MARKERS_NOT_CAPTURED = "club head markers missing from the capture"
REASON_NO_OBSERVED_FRAME = "no frame observes the full club head triad"
REASON_SHAPE_MISMATCH = "club head offsets do not match the captured triad"


@dataclass(frozen=True)
class FaceObservability:
    """Face-angle observability of one capture's club marker configuration.

    ``gaps`` are half-open ``(start, stop)`` frame runs without the full
    triad. Invariants: ``0 <= observed_frames <= total_frames`` and an
    observable result has at least one observed frame.
    """

    observable: bool
    reason: str
    observed_frames: int
    total_frames: int
    gaps: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not 0 <= self.observed_frames <= self.total_frames:
            raise ValueError("observed_frames must lie in [0, total_frames]")
        if self.observable and self.observed_frames == 0:
            raise ValueError("an observable result needs an observed frame")

    @property
    def coverage(self) -> float:
        """Fraction of frames observing the full triad (0 for no frames)."""
        if self.total_frames == 0:
            return 0.0
        return self.observed_frames / self.total_frames


def _unobserved_runs(seen: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    """Half-open runs of ``False`` in ``seen``."""
    padded = np.concatenate(([True], seen, [True])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return tuple((int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True))


def _unobservable(reason: str, total: int) -> FaceObservability:
    return FaceObservability(False, reason, 0, total, ((0, total),))


def _configuration_reason(
    labels: Sequence[str], offsets: Mapping[str, Sequence[float] | Array]
) -> str | None:
    """Reason the marker configuration alone rules out face angle, else None."""
    if len(offsets) < 3:
        return REASON_TOO_FEW_MARKERS
    if cft.triad_is_degenerate(offsets):
        return REASON_DEGENERATE_TRIAD
    if any(label not in labels for label in offsets):
        return REASON_MARKERS_NOT_CAPTURED
    return None


def audit_face_observability(
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    offsets: Mapping[str, Sequence[float] | Array],
) -> FaceObservability:
    """Classify whether the capture can observe the club face angle.

    Args:
        points: ``(frames, markers, 3)`` marker positions in metres.
        valid: ``(frames, markers)`` validity mask.
        labels: Marker name of every column of ``points``.
        offsets: Head-triad offsets in the club marker frame, for example
            :func:`club_face_target.triad_offsets` of the model attachments.

    Returns:
        The observability verdict, observed-frame count and gap runs.

    Raises:
        ValueError: For an empty capture, mismatched shapes, a label list that
            does not name every column, or a non-finite offset. Unobservable
            configurations are a result, not an error.
    """
    names = list(labels)
    total = int(cft.validate_marker_capture(points, valid, names)[0].shape[0])
    if total < 1:
        raise ValueError("capture must contain at least one frame")
    reason = _configuration_reason(names, offsets)
    if reason is not None:
        return _unobservable(reason, total)
    seen = cft.triad_observed_mask(points, valid, names, offsets)
    gaps = _unobserved_runs(seen)
    observed = int(seen.sum())
    if observed == 0:
        return FaceObservability(False, REASON_NO_OBSERVED_FRAME, 0, total, gaps)
    residual = cft.triad_shape_residual_m(points, valid, names, offsets)
    if residual > cft.MAX_TRIAD_SHAPE_RESIDUAL_M:
        return FaceObservability(False, REASON_SHAPE_MISMATCH, 0, total, gaps)
    return FaceObservability(True, REASON_OBSERVABLE, observed, total, gaps)
