"""Motion capture pipeline package.

Public re-exports of the Canonical Intermediate Representation (CIR) types
and the source adapter framework. New code should import from this package
rather than from individual submodules where possible.
"""

from __future__ import annotations

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    CameraExtrinsics,
    CameraIntrinsics,
    JointDef,
    JointLimit,
    JointStateFrame,
    JointTrajectory,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    MotionMatchingRequest,
    MotionMatchingResult,
    MotionTrajectory,
    SkeletonRig,
)

__all__ = [
    "Calibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "JointDef",
    "JointLimit",
    "JointStateFrame",
    "JointTrajectory",
    "Keypoint",
    "KeypointFrame",
    "KeypointSequence",
    "Marker",
    "MarkerFrame",
    "MarkerTrajectory",
    "MotionMatchingRequest",
    "MotionMatchingResult",
    "MotionTrajectory",
    "SkeletonRig",
]
