"""Validated public linearizations of declared native grip and ground geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_law import GroundPlane

Array: TypeAlias = NDArray[np.float64]


def _finite_scalar(value: float, name: str, *, positive: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar")
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def _identities(value: tuple[str, ...], name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a tuple of identities")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} identities must be nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} identities must be unique")


@dataclass(frozen=True)
class ConstraintOptions:
    """Dimensionless residual scaling for declared geometry, not measured contact.

    Each row multiplies its native residual by sqrt(weight) / scale. Grip
    position and ground use metres; principal world-axis grip rotation uses
    radians. Pinned spheres retain signed depth even above the ground plane.
    """

    ground: GroundPlane
    position_weight: float
    rotation_weight: float
    ground_weight: float
    position_scale_m: float
    rotation_scale_rad: float
    ground_scale_m: float
    pinned_spheres: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ground, GroundPlane):
            raise ValueError("ground must be a validated GroundPlane")
        for name in ("position_weight", "rotation_weight", "ground_weight"):
            _finite_scalar(getattr(self, name), name, positive=False)
        for name in ("position_scale_m", "rotation_scale_rad", "ground_scale_m"):
            _finite_scalar(getattr(self, name), name, positive=True)
        _identities(self.pinned_spheres, "pinned contact spheres", allow_empty=True)


@dataclass(frozen=True)
class ConstraintLinearization:
    """Immutable finite dimensionless rows and their native pose derivatives.

    Columns follow coordinate_order. Rows follow row_labels, including zero
    weighted and inactive ground rows. The rotation logarithm is discontinuous
    at pi; unpinned ground depth has a derivative kink at zero.
    """

    residual: Array
    jacobian: Array
    coordinate_order: tuple[str, ...]
    row_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        _identities(self.coordinate_order, "coordinate_order", allow_empty=False)
        _identities(self.row_labels, "row_labels", allow_empty=False)
        residual = np.array(self.residual, dtype=float, copy=True)
        jacobian = np.array(self.jacobian, dtype=float, copy=True)
        if residual.shape != (len(self.row_labels),):
            raise ValueError("Constraint residual shape must match row labels")
        if jacobian.shape != (len(self.row_labels), len(self.coordinate_order)):
            raise ValueError(
                "Constraint Jacobian shape must match rows and coordinates"
            )
        if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
            raise ValueError("Constraint residual and Jacobian must be finite")
        residual.setflags(write=False)
        jacobian.setflags(write=False)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "jacobian", jacobian)
