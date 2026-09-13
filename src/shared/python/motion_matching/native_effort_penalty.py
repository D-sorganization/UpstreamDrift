"""Exact mean-square native sextic effort penalty with analytic control rows."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .native_effort_profile import NativeEffortProfile

Array = NDArray[np.float64]


@dataclass(frozen=True)
class NativeEffortPenalty:
    """Affine residual in physical Bernstein control increments, not total controls."""

    offset: Array
    matrix: Array

    def __post_init__(self) -> None:
        offset, matrix = (
            np.array(self.offset, copy=True),
            np.array(self.matrix, copy=True),
        )
        if (
            offset.ndim != 1
            or not offset.size
            or matrix.ndim != 2
            or matrix.shape[0] != offset.size
            or not matrix.shape[1]
            or not np.isfinite(offset).all()
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Invalid finite native effort penalty dimensions")
        offset.setflags(write=False)
        matrix.setflags(write=False)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "matrix", matrix)

    def _parameters(self, values: Array) -> Array:
        result = np.asarray(values, dtype=float)
        if result.shape != (self.matrix.shape[1],) or not np.isfinite(result).all():
            raise ValueError("Native effort penalty requires finite control increments")
        return result

    def residual(self, increments: Array) -> Array:
        """Return quadrature-scaled total primitive efforts, including the baseline."""
        result = self.offset + self.matrix @ self._parameters(increments)
        if not np.isfinite(result).all():
            raise ValueError("Native effort penalty overflowed")
        return result

    def jacobian(self, increments: Array) -> Array:
        """Return the owned constant residual-by-physical-control derivative."""
        self._parameters(increments)
        return self.matrix


def native_effort_penalty(
    profile: NativeEffortProfile,
    *,
    duration_s: float,
    effort_scales: Array,
    first_control: int = 6,
    weight: float = 1.0,
) -> NativeEffortPenalty:
    """Build weight/T * integral(sum((primitive_effort/scales)**2),0,T).

    Seven-point Gauss-Legendre quadrature is exact for squared degree-six
    polynomials up to roundoff. The existing profile supplies native force-frame
    mapping and Bernstein derivatives. Columns follow its coordinate-major,
    ascending-control order. Scales are positive N/Nm numerical normalizers,
    not physical limits. Weight sets objective balance, not acceptance tolerance.
    """
    scales = np.array(effort_scales, dtype=float, copy=True)
    count = len(profile.evaluate(0.0))
    if (
        not np.isfinite(duration_s)
        or duration_s <= 0
        or not np.isfinite(weight)
        or weight < 0
        or scales.shape != (count,)
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("Require positive duration/scales and nonnegative weight")
    nodes, weights = np.polynomial.legendre.leggauss(7)
    offsets, matrices = [], []
    for node, quadrature_weight in zip(nodes, weights, strict=True):
        time = float(duration_s * (node + 1) / 2)
        scale = np.sqrt(weight * quadrature_weight / 2) / scales
        offsets.append(scale * np.array(list(profile.evaluate(time).values())))
        matrices.append(
            scale[:, None]
            * profile.bernstein_control_jacobian(
                time, basis_duration_s=duration_s, first_control=first_control
            )
        )
    return NativeEffortPenalty(np.concatenate(offsets), np.concatenate(matrices))
