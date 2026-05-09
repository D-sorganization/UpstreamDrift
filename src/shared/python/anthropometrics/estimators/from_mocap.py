"""Segment-length estimator from mocap marker trajectories.

Wave 2 of the anthropometrics EPIC (#4797). Closes #4815.

This module is a *thin wrapper* around the per-frame distance helper
in :mod:`motion_pipeline.scaling.anthropometric`. The wrap exists
because the existing module's public entry point
(:func:`scale_skeleton`) returns a :class:`SkeletonRig`, whereas the
anthropometrics pipeline needs a plain ``{segment_name: length_m}``
mapping with NaN-tolerant inputs that the rest of the pipeline can
feed into :class:`anthropometrics.SegmentProperties`. Wrapping (rather
than re-implementing the Euclidean-distance arithmetic) keeps the
distance algorithm in a single place — see DRY assertion in the
matching test module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

# DRY: reuse the per-frame distance computation from the motion pipeline.
# Importing the contracts here lets us construct the per-frame MarkerFrame
# the helper expects without duplicating the Euclidean-distance kernel.
from motion_pipeline.contracts import Marker, MarkerFrame
from motion_pipeline.scaling.anthropometric import _compute_segment_length

__all__ = [
    "SegmentDef",
    "estimate_segment_lengths_from_markers",
]


_Method = Literal["mean_distance", "median_distance", "min_distance"]
_VALID_METHODS: tuple[_Method, ...] = (
    "mean_distance",
    "median_distance",
    "min_distance",
)


@dataclass(frozen=True)
class SegmentDef:
    """Definition of a body segment in terms of its bounding markers.

    Attributes:
        name: Segment name (used as the key in the returned mapping).
        proximal_marker: Name of the marker at the proximal end.
        distal_marker: Name of the marker at the distal end.
    """

    name: str
    proximal_marker: str
    distal_marker: str


def _reduce(values: list[float], method: _Method) -> float:
    """Reduce a list of per-frame distances to a single segment length."""
    arr = np.asarray(values, dtype=float)
    if method == "mean_distance":
        return float(np.mean(arr))
    if method == "median_distance":
        return float(np.median(arr))
    if method == "min_distance":
        return float(np.min(arr))
    # Defensive — Literal-typed but validated explicitly for runtime callers.
    raise ValueError(f"Unknown method {method!r}. Expected one of {_VALID_METHODS}.")


def estimate_segment_lengths_from_markers(
    markers: dict[str, np.ndarray],
    segment_definitions: Sequence[SegmentDef],
    *,
    method: _Method = "median_distance",
) -> dict[str, float]:
    """Per-segment length in metres from marker trajectories.

    NaN-tolerant: frames where either the proximal or distal marker is
    non-finite (NaN/Inf in any coordinate) are excluded from the
    reduction. The remaining finite frames are reduced via *method*.

    Args:
        markers: Mapping of marker name to ``(T, 3)`` ``np.ndarray`` in
            metres. Each array gives the trajectory of one marker over
            ``T`` frames; non-finite rows mark missing/occluded samples.
        segment_definitions: Iterable of :class:`SegmentDef` describing
            which marker pairs delimit each segment. Order is preserved
            in the result.
        method: Reduction strategy across the finite frames.
            ``"mean_distance"`` — arithmetic mean (sensitive to
            outliers).
            ``"median_distance"`` — robust median (default).
            ``"min_distance"`` — conservative minimum.

    Returns:
        Dict mapping segment name to its estimated length in metres.

    Raises:
        ValueError: If *segment_definitions* is empty, if a required
            marker is missing from *markers*, if marker arrays are not
            ``(T, 3)`` with consistent ``T``, if *method* is unknown,
            or if no finite frames remain for any required segment
            (the message lists the offending segment(s) and missing
            markers).
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Expected one of {_VALID_METHODS}."
        )

    seg_list = list(segment_definitions)
    if not seg_list:
        raise ValueError("segment_definitions must be non-empty.")

    # Identify required markers and report all missing ones at once
    # for a more useful error message.
    required: set[str] = set()
    for seg in seg_list:
        required.add(seg.proximal_marker)
        required.add(seg.distal_marker)

    missing = sorted(name for name in required if name not in markers)
    if missing:
        raise ValueError(
            "Missing required marker(s) in `markers` mapping: "
            f"{missing}. Required markers: {sorted(required)}."
        )

    # Validate shapes and infer the common frame count T.
    frame_counts: set[int] = set()
    for name in required:
        arr = np.asarray(markers[name])
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Marker {name!r} must be a (T, 3) array; got shape {arr.shape}."
            )
        frame_counts.add(arr.shape[0])
    if len(frame_counts) != 1:
        raise ValueError(
            "All marker arrays must share the same number of frames T; "
            f"got {sorted(frame_counts)}."
        )
    (n_frames,) = frame_counts
    if n_frames == 0:
        raise ValueError("Marker arrays have zero frames (T == 0).")

    # Per-frame: build a MarkerFrame containing only the finite markers,
    # then delegate to the motion-pipeline distance helper. This routes
    # through the canonical Euclidean-distance kernel (DRY) instead of
    # recomputing it here.
    per_segment_values: dict[str, list[float]] = {seg.name: [] for seg in seg_list}
    finite_mask: dict[str, np.ndarray] = {
        name: np.all(np.isfinite(np.asarray(markers[name])), axis=1)
        for name in required
    }

    for t in range(n_frames):
        frame_markers: dict[str, Marker] = {}
        for name in required:
            if finite_mask[name][t]:
                xyz = np.asarray(markers[name])[t]
                frame_markers[name] = Marker(
                    name=name,
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                )
        # Timestamp is not used by the distance helper; supply a
        # monotonically-increasing placeholder to satisfy contracts.
        marker_frame = MarkerFrame(timestamp=float(t), markers=frame_markers)
        for seg in seg_list:
            length = _compute_segment_length(
                marker_frame, seg.proximal_marker, seg.distal_marker
            )
            if length is not None:
                per_segment_values[seg.name].append(length)

    empty = [name for name, vals in per_segment_values.items() if not vals]
    if empty:
        raise ValueError(
            "No finite frames available for segment(s): "
            f"{sorted(empty)}. Check marker NaN coverage and definitions."
        )

    return {seg.name: _reduce(per_segment_values[seg.name], method) for seg in seg_list}
