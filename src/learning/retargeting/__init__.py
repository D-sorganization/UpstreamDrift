"""Motion Retargeting for transferring motion between embodiments.

This module provides tools for:
- Mapping motion between different skeleton structures
- Motion capture to robot retargeting
- Skeleton configuration and joint mapping
"""

from __future__ import annotations

from src.learning.retargeting.retargeter import MotionRetargeter, SkeletonConfig

__all__ = [
    "MotionRetargeter",
    "SkeletonConfig",
]
