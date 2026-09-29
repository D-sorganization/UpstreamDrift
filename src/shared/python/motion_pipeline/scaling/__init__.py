"""
Anthropometric scaling for motion capture pipeline.

Part of issue #4565. Scales a generic SkeletonRig to match subject-specific
marker data using segment length estimation.
"""

from .anthropometric import MarkerMap, scale_skeleton
from .marker_maps import (
    IOR,
    PLUG_IN_GAIT,
    THEIA,
    VICON_FULL_BODY,
    MarkerSet,
    get_marker_set,
)

__all__ = [
    "scale_skeleton",
    "MarkerMap",
    "get_marker_set",
    "MarkerSet",
    "PLUG_IN_GAIT",
    "IOR",
    "THEIA",
    "VICON_FULL_BODY",
]
