"""Alternating marker-placement calibration for a body-fixed MarkerSet (OS-3).

Re-exports from src.shared.python.motion_matching.marker_calibration for backwards compatibility.
"""

from __future__ import annotations

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
    "calibrate_marker_offsets",
    "express_in_body",
    "rigid_pose_from_markers",
    "static_marker_offsets",
]
