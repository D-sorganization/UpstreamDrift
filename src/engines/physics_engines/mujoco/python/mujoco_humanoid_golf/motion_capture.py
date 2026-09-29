# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.  # noqa: E501
# It requires domain-aware structural extraction to isolate its internal classes appropriately.  # noqa: E501

"""Motion capture integration and retargeting for golf swing analysis.

This module provides comprehensive motion capture data handling, including:
- Loading mocap data from multiple formats (BVH, C3D, CSV, JSON)
- Motion retargeting to MuJoCo models using IK
- Kinematic trajectory extraction and processing
- Marker-based and markerless mocap support
- Temporal alignment and filtering
"""

from ._mocap_data import MarkerSet, MotionCaptureFrame, MotionCaptureSequence
from ._mocap_loader import MotionCaptureLoader
from ._mocap_processor import MotionCaptureProcessor
from ._mocap_retargeting import MotionRetargeting
from ._mocap_validator import MotionCaptureValidator

__all__ = [
    "MarkerSet",
    "MotionCaptureFrame",
    "MotionCaptureLoader",
    "MotionCaptureProcessor",
    "MotionCaptureSequence",
    "MotionCaptureValidator",
    "MotionRetargeting",
]
