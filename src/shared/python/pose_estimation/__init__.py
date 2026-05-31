"""Pose Estimation Package for Golf Modeling Suite.

This package provides interfaces and implementations for estimating
human pose / joint angles from video or mocap data.
"""

from .interface import PoseEstimationResult, PoseEstimator
from .observations import (
    CANONICAL_OBSERVATIONS_SCHEMA_VERSION,
    TRACE_META_OBSERVATIONS_JSON,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    CanonicalObservations,
    DetectorLayout,
    KeypointObservation,
)
from .openpose_estimator import OpenPoseEstimator

__all__ = [
    "CANONICAL_OBSERVATIONS_SCHEMA_VERSION",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "CanonicalObservations",
    "DetectorLayout",
    "KeypointObservation",
    "OpenPoseEstimator",
    "PoseEstimationResult",
    "PoseEstimator",
    "TRACE_META_OBSERVATIONS_JSON",
]
