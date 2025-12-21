"""Pose Estimation Package for Golf Modeling Suite.

This package provides interfaces and implementations for estimating
human pose / joint angles from video or mocap data.
"""

from .interface import PoseEstimationResult, PoseEstimator

__all__ = ["PoseEstimator", "PoseEstimationResult"]
