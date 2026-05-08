"""
Anthropometric scaling for motion capture pipeline.

Part of issue #4565. Scales a generic SkeletonRig to match subject-specific
marker data using segment length estimation.
"""

from .anthropometric import scale_skeleton, MarkerMap, get_marker_map
from .marker_maps import MarkerSet, PLUG_IN_GAIT, IOR, THEIA, VICON_FULL_BODY

__all__ = [
    "scale_skeleton",
    "MarkerMap",
    "get_marker_map",
    "MarkerSet",
    "PLUG_IN_GAIT",
    "IOR",
    "THEIA",
    "VICON_FULL_BODY",
]