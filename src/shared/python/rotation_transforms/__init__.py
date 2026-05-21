"""Shared rotation and reference-frame primitives.

This package is intentionally under ``src/shared/python`` so shared services
such as ``calc_backend`` can use rotation math without depending on the
leaf ``rotation_converter`` tool package.
"""

from rotation_transforms.reference_frame_operations import (
    OperationName,
    ReferenceFrameResult,
    compute_homogeneous_transform,
    compute_reference_frame_operation,
    compute_so3_so3_maps,
    compute_twist_frame_conversion,
)
from rotation_transforms.rotation import Rotation

__all__ = [
    "OperationName",
    "ReferenceFrameResult",
    "Rotation",
    "compute_homogeneous_transform",
    "compute_reference_frame_operation",
    "compute_so3_so3_maps",
    "compute_twist_frame_conversion",
]
