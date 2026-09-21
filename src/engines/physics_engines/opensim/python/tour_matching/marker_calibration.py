"""Alternating marker-placement calibration for a body-fixed MarkerSet (OS-3).

Re-exports from src.shared.python.motion_matching.marker_calibration for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
import math

from src.shared.python.contracts import require
from src.shared.python.motion_matching.marker_calibration import (
    Array,
    CalibrationResult,
    IkFn,
    Offsets,
    Pose,
    PoseFn,
    _marker_rms,
    _marker_rms_and_per_marker,
    _placements,
    calibrate_marker_offsets,
    express_in_body,
    rigid_pose_from_markers,
    static_marker_offsets,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture


def bound_marker_offsets(
    offsets: Offsets,
    max_offset_radius_m: float = 0.50,
    prior_offsets: Mapping[str, Sequence[float]] | None = None,
    max_deviation_from_prior_m: float = 0.08,
) -> Offsets:
    """Clamp body-fixed marker offsets within anatomical and prior bounds (OG-05).

    Preconditions:
    - max_offset_radius_m and max_deviation_from_prior_m must be strictly positive.

    Postconditions:
    - Returns bounded offsets where every marker is within allowable radii.
    """
    require(max_offset_radius_m > 0, "max_offset_radius_m must be positive")
    require(
        max_deviation_from_prior_m > 0, "max_deviation_from_prior_m must be positive"
    )

    bounded: Offsets = {}
    for label, (body, offset) in offsets.items():
        v = np.asarray(offset, dtype=float)
        # Check against anatomical prior
        if prior_offsets is not None and label in prior_offsets:
            prior = np.asarray(prior_offsets[label], dtype=float)
            diff = v - prior
            dist = float(math.sqrt(np.vdot(diff, diff)))  # Bolt optimization
            if dist > max_deviation_from_prior_m:
                v = prior + diff * (max_deviation_from_prior_m / dist)

        # Check total radius from body origin
        rad = float(math.sqrt(np.vdot(v, v)))  # Bolt optimization
        if rad > max_offset_radius_m:
            v = v * (max_offset_radius_m / rad)

        bounded[label] = (body, (float(v[0]), float(v[1]), float(v[2])))

    return bounded


def calibrate_marker_offsets_with_holdout(
    capture: TourCapture,
    bodies: Mapping[str, str],
    pose_fn: PoseFn,
    ik_fn: IkFn,
    *,
    initial_q: Array,
    iterations: int,
    holdout_labels: Sequence[str],
) -> tuple[CalibrationResult, float]:
    """Perform alternating marker calibration with held-out validation markers (OG-05).

    Preconditions:
    - holdout_labels must be a strict non-empty subset of capture.labels.

    Postconditions:
    - Returns CalibrationResult on the training set and holdout RMS in metres.
    """
    holdout_set = set(holdout_labels)
    require(len(holdout_set) > 0, "holdout_labels must be non-empty")
    require(
        holdout_set.issubset(set(capture.labels)),
        "holdout_labels must be a subset of capture.labels",
    )
    require(
        len(holdout_set) < len(capture.labels),
        "At least one training label must remain",
    )

    train_labels = [lb for lb in capture.labels if lb not in holdout_set]
    train_capture = capture.subset(train_labels)

    result = calibrate_marker_offsets(
        train_capture,
        bodies,
        pose_fn,
        ik_fn,
        initial_q=initial_q,
        iterations=iterations,
    )

    # Evaluate holdout markers across the fitted trajectory
    holdout_capture = capture.subset(list(holdout_labels))
    poses = [pose_fn(result.q[f]) for f in range(holdout_capture.frames)]
    holdout_offsets = static_marker_offsets(holdout_capture, bodies, poses)
    holdout_rms = _marker_rms(holdout_capture, holdout_offsets, poses)

    return result, holdout_rms


__all__ = [
    "Array",
    "CalibrationResult",
    "IkFn",
    "Offsets",
    "Pose",
    "PoseFn",
    "_marker_rms",
    "_marker_rms_and_per_marker",
    "_placements",
    "bound_marker_offsets",
    "calibrate_marker_offsets",
    "calibrate_marker_offsets_with_holdout",
    "express_in_body",
    "rigid_pose_from_markers",
    "static_marker_offsets",
]
