"""Coulomb friction cones and physical force constraints (#12552)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.motion_matching.force_nullspace._common import Array, _array


@dataclass(frozen=True)
class FrictionCone:
    """Exact Coulomb cone for three force entries in the declared world frame.

    frame columns are two tangents and the outward normal, expressed in the
    same frame as these force entries. Inactive contacts need zero force bounds.
    """

    indices: tuple[int, int, int]
    frame: Array
    mu: float

    def __post_init__(self) -> None:
        frame = _array(self.frame, 2, "frame")
        if frame.shape != (3, 3) or not np.allclose(
            frame.T @ frame, np.eye(3), atol=1e-12, rtol=0
        ):
            raise ValueError("surface frame must be orthonormal")
        if (
            len(self.indices) != 3
            or len(set(self.indices)) != 3
            or any(not isinstance(i, int) or i < 0 for i in self.indices)
        ):
            raise ValueError("cone indices must be three distinct nonnegative integers")
        if not np.isfinite(self.mu) or self.mu < 0:
            raise ValueError("friction coefficient must be finite and nonnegative")
        object.__setattr__(self, "frame", frame)

    def margins(self, forces: Array) -> Array:
        """Return unilateral and exact circular-cone margins in force units."""
        local = self.frame.T @ forces[list(self.indices)]
        return np.array([local[2], self.mu * local[2] - np.linalg.norm(local[:2])])


@dataclass(frozen=True)
class ForceConstraints:
    """Finite physical bounds, optional G x <= h, and exact friction cones.

    Bounds can encode zero trail effort or inactive contacts. G/h can encode
    grip limits or conservative COP inequalities. Geometry and constitutive
    compatibility remain caller obligations; they cannot be inferred from A.
    """

    lower: Array
    upper_bound: Array
    matrix: Array | None = None
    upper: Array | None = None
    cones: tuple[FrictionCone, ...] = ()

    def __post_init__(self) -> None:
        lo = _array(self.lower, 1, "lower")
        hi = _array(self.upper_bound, 1, "upper_bound")
        if not lo.size or lo.shape != hi.shape or np.any(lo > hi):
            raise ValueError("force bounds must have matching shape and lower <= upper")
        if (self.matrix is None) != (self.upper is None):
            raise ValueError("inequality matrix and upper must be supplied together")
        g = _array(
            np.empty((0, lo.size)) if self.matrix is None else self.matrix, 2, "matrix"
        )
        h = _array(np.empty(0) if self.upper is None else self.upper, 1, "upper")
        if g.shape != (h.size, lo.size):
            raise ValueError("inequality dimensions mismatch")
        if any(max(cone.indices) >= lo.size for cone in self.cones):
            raise ValueError("cone index exceeds force dimension")
        for name, value in (
            ("lower", lo),
            ("upper_bound", hi),
            ("matrix", g),
            ("upper", h),
        ):
            object.__setattr__(self, name, value)

    def margins(self, forces: Array) -> Array:
        """Nonnegative entries mean all supplied physical constraints hold."""
        linear = (
            []
            if self.matrix is None or self.upper is None
            else [self.upper - self.matrix @ forces]
        )
        return np.concatenate(
            [
                forces - self.lower,
                self.upper_bound - forces,
                *linear,
                *(cone.margins(forces) for cone in self.cones),
            ]
        )


__all__ = ["ForceConstraints", "FrictionCone"]
