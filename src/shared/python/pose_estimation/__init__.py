"""Pose Estimation Package for Golf Modeling Suite.

This package provides interfaces and implementations for estimating
human pose / joint angles from video or mocap data.
"""

from .interface import PoseEstimationResult, PoseEstimator
from .keypoint_offsets import (
    KeypointOffsetEstimate,
    KeypointOffsetModel,
    KeypointOffsetSite,
    estimate_keypoint_offset,
    estimate_keypoint_offset_model,
)
from .openpose_estimator import OpenPoseEstimator

__all__ = [
    "KeypointOffsetEstimate",
    "KeypointOffsetModel",
    "KeypointOffsetSite",
    "OpenPoseEstimator",
    "PoseEstimator",
    "PoseEstimationResult",
    "estimate_keypoint_offset",
    "estimate_keypoint_offset_model",
]
