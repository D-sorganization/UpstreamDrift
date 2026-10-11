"""Clamped uniform B-spline basis with analytic derivative matrices.

The kinematic trajectory of every trial is represented as ``q(t) = P(t) C``
with ``C`` an ``(n_coefficients, n_dof)`` coefficient matrix.  Position,
velocity and acceleration at the sample times are then *linear* in ``C``
(``P``, ``V``, ``A`` matrices), which keeps the inverse-dynamics collocation
residual affine in the inner variables and makes Jacobians a matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BSplineBasis:
    """Evaluated basis matrices of a clamped B-spline on fixed sample times."""

    times: FloatArray
    knots: FloatArray
    degree: int
    position: FloatArray
    velocity: FloatArray
    acceleration: FloatArray

    @property
    def n_coefficients(self) -> int:
        return int(self.position.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.times.size)

    @classmethod
    def uniform(
        cls, times: FloatArray, n_coefficients: int, degree: int = 5
    ) -> BSplineBasis:
        """Build a clamped uniform-knot basis spanning ``[times[0], times[-1]]``.

        Preconditions: strictly increasing times, ``degree >= 2`` (the
        acceleration matrix needs a second derivative) and
        ``n_coefficients > degree`` (otherwise the spline cannot interpolate).
        """
        samples = np.asarray(times, dtype=np.float64)
        require(samples.ndim == 1 and samples.size >= 2, "need >= 2 sample times")
        require(
            bool(np.all(np.diff(samples) > 0.0)), "times must be strictly increasing"
        )
        require(degree >= 2, "degree must be >= 2 (acceleration needs it)", degree)
        require(
            n_coefficients > degree,
            "n_coefficients must exceed the degree",
            n_coefficients,
        )
        interior = np.linspace(samples[0], samples[-1], n_coefficients - degree + 1)
        knots = np.concatenate(
            [np.full(degree, samples[0]), interior, np.full(degree, samples[-1])]
        )
        identity = np.eye(n_coefficients)
        spline = BSpline(knots, identity, degree, extrapolate=False)
        matrices = [
            spline.derivative(order)(samples) if order else spline(samples)
            for order in range(3)
        ]
        position, velocity, acceleration = (
            np.nan_to_num(matrix) for matrix in matrices
        )
        ensure(bool(np.allclose(position.sum(axis=1), 1.0)), "basis partitions unity")
        return cls(samples, knots, degree, position, velocity, acceleration)

    def fit_least_squares(self, samples: FloatArray, ridge: float = 0.0) -> FloatArray:
        """Return coefficients ``C`` minimising ``|P C - samples|^2 + ridge |C|^2``."""
        require(
            samples.shape[0] == self.n_samples, "one sample row per time", samples.shape
        )
        require(ridge >= 0.0, "ridge must be non-negative", ridge)
        normal = self.position.T @ self.position + ridge * np.eye(self.n_coefficients)
        solution = np.linalg.solve(normal, self.position.T @ samples)
        return np.asarray(solution, dtype=np.float64)

    def evaluate(
        self, coefficients: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(q, v, a)`` at the sample times for coefficient matrix ``C``."""
        require(
            coefficients.shape[0] == self.n_coefficients,
            "coefficient rows",
            coefficients.shape,
        )
        return (
            self.position @ coefficients,
            self.velocity @ coefficients,
            self.acceleration @ coefficients,
        )
