"""Rigid, ground-aligned placement of the plate stance frame in the world (OF-01).

A vertical plate measures load along the ground normal, so the stance frame may
only turn about the vertical axis and translate: any tilt would make the
measured load non-vertical, and any reflection would silently swap left and
right.  :class:`PlatePlacement` refuses both.  World frame: Z-up, metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.force_plates.contracts import Array

__all__ = ["ROTATION_ATOL", "PlatePlacement"]

#: Tolerance on orthonormality and on keeping the vertical axis vertical.
ROTATION_ATOL = 1e-9
_ZHAT = np.array([0.0, 0.0, 1.0])


def _frozen(value: ArrayLike) -> Array:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, eq=False)
class PlatePlacement:
    """``world = origin_m + rotation @ (x, y, 0)`` for a stance point ``(x, y)``.

    ``origin_m`` is the world position of the stance origin on the plate top
    surface; its ``z`` is the surface height.  ``rotation`` must be a proper
    rotation that keeps world ``+z`` fixed (a yaw).

    Raises ``ValueError`` for a non-finite origin or a rotation that is not
    orthonormal, is a reflection, or tilts the vertical axis.
    """

    rotation: Array
    origin_m: Array

    def __post_init__(self) -> None:
        rot = _frozen(self.rotation)
        origin = _frozen(self.origin_m)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("origin_m must be a finite 3-vector")
        if rot.shape != (3, 3) or not np.isfinite(rot).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rot.T @ rot, np.eye(3), rtol=0.0, atol=ROTATION_ATOL):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rot) < 0.0:
            raise ValueError(
                "rotation must be proper (a reflection would swap the feet)"
            )
        if not np.allclose(rot @ _ZHAT, _ZHAT, rtol=0.0, atol=ROTATION_ATOL):
            raise ValueError("rotation must keep the vertical axis vertical (yaw only)")
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "origin_m", origin)

    @classmethod
    def identity(cls) -> PlatePlacement:
        """Stance frame equal to the world frame, surface at ``z = 0``."""
        return cls(rotation=np.eye(3), origin_m=np.zeros(3))

    @classmethod
    def from_yaw(cls, yaw_rad: float, *, origin_m: ArrayLike) -> PlatePlacement:
        """Placement turned by ``yaw_rad`` about world ``+z`` (counter-clockwise)."""
        if not math.isfinite(yaw_rad):
            raise ValueError(f"yaw_rad must be finite, got {yaw_rad!r}")
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return cls(rotation=rot, origin_m=np.asarray(origin_m, dtype=np.float64))

    @classmethod
    def facing(cls, facing_xy: ArrayLike, *, origin_m: ArrayLike) -> PlatePlacement:
        """Placement that turns stance ``+y`` (where the golfer faces) onto ``facing_xy``.

        ``facing_xy`` is a non-zero horizontal world direction.  Under ADR-0041
        (golfer faces world ``-X``) pass ``(-1, 0)``: stance ``+x`` then lies
        along world ``+Y``, so a right-handed golfer's target is ``-Y``.  The
        placement never mirrors, so a left-handed golfer standing on the same
        plates keeps its own target direction; mirroring a swing is a separate,
        explicit model decision.
        """
        xy = np.asarray(facing_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.any(xy):
            raise ValueError(f"facing_xy must be a finite non-zero 2-vector, got {xy}")
        yaw = math.atan2(float(xy[1]), float(xy[0])) - math.pi / 2
        return cls.from_yaw(yaw, origin_m=origin_m)

    @property
    def surface_height_m(self) -> float:
        """World height of the plate top surface [m]."""
        return float(self.origin_m[2])

    def to_world(self, stance_xy_m: ArrayLike) -> Array:
        """World points ``(T, 3)`` for stance points ``(T, 2)``; NaN rows stay NaN."""
        xy = np.asarray(stance_xy_m, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"stance points must have shape (T, 2), got {xy.shape}")
        stance = np.column_stack([xy, np.zeros(xy.shape[0])])
        return stance @ self.rotation.T + self.origin_m
