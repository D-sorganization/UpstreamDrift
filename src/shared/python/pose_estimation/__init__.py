"""Pose Estimation Package for Golf Modeling Suite.

This package provides interfaces and implementations for estimating
human pose / joint angles from video or mocap data.
"""

from .interface import PoseEstimationResult, PoseEstimator
from .observations import (
    CanonicalObservations,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    observations_from_dict,
    observations_to_dict,
)
from .openpose_estimator import OpenPoseEstimator

__all__ = [
    "CanonicalObservations",
    "CameraCalibration",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "OpenPoseEstimator",
    "PoseEstimationResult",
    "PoseEstimator",
    "observations_from_dict",
    "observations_to_dict",
]
