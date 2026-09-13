"""Named, fixed-frame SE(3) transport reusing shared Pluecker transforms."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.spatial_algebra.transforms import xtrans
from .se3 import compose_se3, inverse_se3, is_valid_se3

Array = NDArray[np.float64]


def _rigid(value: ArrayLike) -> Array:
    matrix = np.array(value, dtype=float, copy=True)
    if not np.isfinite(matrix).all() or not is_valid_se3(matrix):
        raise ValueError("Expected a finite rigid SE(3) transform")
    return matrix


def _six(value: ArrayLike) -> Array:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite spatial vector of length six")
    return vector


@dataclass(frozen=True)
class FixedFrameTransport:
    """T_target_source maps source coordinates into target coordinates.

    Spatial twists are [angular, linear], wrenches [moment, force]. Linear
    twist components are about the frame origin, not arbitrary point speeds.
    Wrench transport includes origin translation. These are fixed-frame maps;
    accelerating/moving reference frames require additional transport terms.
    """

    source: str
    target: str
    target_from_source: Array

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("Explicit nonempty frame names are required")
        matrix = _rigid(self.target_from_source)
        matrix.flags.writeable = False
        object.__setattr__(self, "target_from_source", matrix)

    def adjoint(self) -> Array:
        """Spatial motion map; uses the existing Featherstone transform."""
        matrix = self.target_from_source
        rotation, offset = matrix[:3, :3], matrix[:3, 3]
        return xtrans(rotation, -rotation.T @ offset)

    def pose(self, source_from_body: ArrayLike) -> Array:
        """Transform a body pose from source coordinates to target coordinates."""
        return compose_se3(self.target_from_source, _rigid(source_from_body))

    def twist(self, source_twist: ArrayLike) -> Array:
        """Transport a spatial twist to the named target frame and origin."""
        return self.adjoint() @ _six(source_twist)

    def wrench(self, source_wrench: ArrayLike) -> Array:
        """Dual map preserving wrench dot twist (instantaneous power)."""
        return np.asarray(
            np.linalg.solve(self.adjoint().T, _six(source_wrench)), dtype=np.float64
        )

    def inverse(self) -> "FixedFrameTransport":
        """Reverse direction, preserving explicit frame names."""
        return FixedFrameTransport(
            self.target, self.source, inverse_se3(self.target_from_source)
        )
