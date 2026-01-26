"""
Shared spatial algebra utilities.

This package provides a canonical implementation of Featherstone-style
spatial algebra that can be reused across physics engines.
"""

from .inertia import mcI, mci, transform_spatial_inertia
from .joints import (
    JOINT_AXIS_INDICES,
    S_PX,
    S_PY,
    S_PZ,
    S_RX,
    S_RY,
    S_RZ,
    jcalc,
)
from .spatial_vectors import (
    crf,
    crm,
    cross_force,
    cross_force_fast,
    cross_motion,
    cross_motion_axis,
    cross_motion_fast,
    skew,
    spatial_cross,
)
from .transforms import inv_xtrans, xlt, xrot, xtrans

__all__ = [
    "JOINT_AXIS_INDICES",
    "S_PX",
    "S_PY",
    "S_PZ",
    "S_RX",
    "S_RY",
    "S_RZ",
    "crf",
    "crm",
    "cross_force",
    "cross_force_fast",
    "cross_motion",
    "cross_motion_axis",
    "cross_motion_fast",
    "inv_xtrans",
    "jcalc",
    "mcI",
    "mci",
    "skew",
    "spatial_cross",
    "transform_spatial_inertia",
    "xlt",
    "xrot",
    "xtrans",
]
