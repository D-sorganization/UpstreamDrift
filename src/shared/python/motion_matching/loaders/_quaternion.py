"""Compatibility exports for canonical quaternion helpers."""

from __future__ import annotations

from src.shared.python.math_utils.quaternion import (
    _canonicalize_sign,
    quat_inverse_distance,
    rotmat_to_quat,
    slerp,
)

__all__ = ["_canonicalize_sign", "quat_inverse_distance", "rotmat_to_quat", "slerp"]
