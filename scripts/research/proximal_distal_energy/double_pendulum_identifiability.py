"""Exact coefficient and physical-parameter identifiability boundaries.

The planar double pendulum's inverse dynamics are linear in seven base
coefficients even though the reduced physical description below has eleven
entries.  This factorization proves a physical-parameter non-uniqueness that
cannot be repaired by more samples of the same ideal input/state experiment.
Finite-record regressor rank is reported separately as an excitation property.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.double_pendulum_physical_parameters import (
    BASE_COEFFICIENT_NAMES,
    BASE_COEFFICIENT_UNITS,
    PHYSICAL_PARAMETER_NAMES,
    DoublePendulumPhysicalParameters,
    StructuralRankWitness,
    exact_invariance_counterexamples,
    parameter_map_jacobian,
    physical_parameter_rank_witness,
)


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CoefficientUncertaintyLowerBound:
    """Best-case coefficient uncertainty under oracle kinematics and iid torque noise."""

    full_rank: bool
    matrix_shape: tuple[int, int]
    rank: int
    torque_noise_sd_nm: float
    dimensionless_singular_values: tuple[float, ...]
    dimensionless_retained_condition_number: float | None
    standard_errors: tuple[float, ...] | None
    ci95_relative_half_widths: tuple[float, ...] | None
    max_abs_parameter_correlation: float | None


@dataclass(frozen=True, slots=True)
class CoefficientScaleContract:
    """Positive coordinates used before numerical rank or Fisher analysis."""

    coefficient_scales: tuple[float, ...]
    torque_scale_nm: float

    def __post_init__(self) -> None:
        values = np.asarray(self.coefficient_scales, dtype=float)
        if (
            values.shape != (len(BASE_COEFFICIENT_NAMES),)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError("coefficient_scales must contain seven positive values")
        if not math.isfinite(self.torque_scale_nm) or self.torque_scale_nm <= 0.0:
            raise ValueError("torque_scale_nm must be positive and finite")

    def coefficient_array(self) -> FloatArray:
        """Return the declared coefficient scales in registered order."""
        return np.asarray(self.coefficient_scales, dtype=float)


def nondimensional_regressor(
    regressor: npt.ArrayLike, scales: CoefficientScaleContract
) -> FloatArray:
    """Map a dimensional torque regressor into dimensionless coordinates."""
    matrix = np.asarray(regressor, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(BASE_COEFFICIENT_NAMES)
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("regressor must be a finite nonempty (n, 7) matrix")
    return matrix * scales.coefficient_array()[None, :] / scales.torque_scale_nm


def coefficient_uncertainty_lower_bound(
    regressor: npt.ArrayLike,
    base_coefficients: npt.ArrayLike,
    *,
    scales: CoefficientScaleContract,
    torque_noise_sd_nm: float,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
) -> CoefficientUncertaintyLowerBound:
    """Return an oracle-kinematics Gaussian Fisher-information lower bound.

    The result assumes exact positions, velocities, accelerations, model form,
    event alignment, and independent homoscedastic generalized-torque noise.
    It is therefore a best-case lower bound, not a practical-identifiability
    result. Rank-deficient records fail closed without pseudo-precision.
    """
    matrix = np.asarray(regressor, dtype=float)
    coefficients = np.asarray(base_coefficients, dtype=float).reshape(-1)
    noise = float(torque_noise_sd_nm)
    if not math.isfinite(noise) or noise <= 0.0:
        raise ValueError("torque_noise_sd_nm must be positive and finite")
    dimensionless = nondimensional_regressor(matrix, scales)
    if (
        coefficients.shape != (len(BASE_COEFFICIENT_NAMES),)
        or not np.all(np.isfinite(coefficients))
        or np.any(coefficients == 0.0)
    ):
        raise ValueError("base_coefficients must contain seven finite nonzero values")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("rank tolerances must be nonnegative")

    _, singular_values, right_vectors = np.linalg.svd(
        dimensionless, full_matrices=False
    )
    leading = float(singular_values[0]) if singular_values.size else 0.0
    threshold = max(absolute_tolerance, relative_tolerance * leading)
    retained = singular_values > threshold
    rank = int(np.count_nonzero(retained))
    shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == matrix.shape[1]
        else None
    )
    if rank != matrix.shape[1]:
        return CoefficientUncertaintyLowerBound(
            full_rank=False,
            matrix_shape=shape,
            rank=rank,
            torque_noise_sd_nm=noise,
            dimensionless_singular_values=tuple(
                float(value) for value in singular_values
            ),
            dimensionless_retained_condition_number=condition,
            standard_errors=None,
            ci95_relative_half_widths=None,
            max_abs_parameter_correlation=None,
        )

    inverse_squared = 1.0 / np.square(singular_values)
    dimensionless_noise = noise / scales.torque_scale_nm
    covariance_dimensionless = dimensionless_noise**2 * (
        (right_vectors.T * inverse_squared) @ right_vectors
    )
    scale_diagonal = np.diag(scales.coefficient_array())
    covariance = scale_diagonal @ covariance_dimensionless @ scale_diagonal
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    relative_half_widths = 1.959963984540054 * standard_errors / np.abs(coefficients)
    correlation = covariance / np.outer(standard_errors, standard_errors)
    off_diagonal = correlation - np.diag(np.diag(correlation))
    return CoefficientUncertaintyLowerBound(
        full_rank=True,
        matrix_shape=shape,
        rank=rank,
        torque_noise_sd_nm=noise,
        dimensionless_singular_values=tuple(float(value) for value in singular_values),
        dimensionless_retained_condition_number=condition,
        standard_errors=tuple(float(value) for value in standard_errors),
        ci95_relative_half_widths=tuple(float(value) for value in relative_half_widths),
        max_abs_parameter_correlation=float(np.max(np.abs(off_diagonal))),
    )


def inverse_dynamics_regressor(
    q: npt.ArrayLike, velocity: npt.ArrayLike, acceleration: npt.ArrayLike
) -> FloatArray:
    """Return the exact ``2 x 7`` base-coefficient inverse-dynamics regressor."""
    position = _two_vector("q", q)
    rate = _two_vector("velocity", velocity)
    accel = _two_vector("acceleration", acceleration)
    q1, q2 = position
    v1, v2 = rate
    a1, a2 = accel
    cos_q2 = math.cos(q2)
    sin_q2 = math.sin(q2)
    distal_gravity_shape = math.sin(q1 + q2)
    return np.array(
        [
            [
                a1,
                2.0 * cos_q2 * a1 + cos_q2 * a2 - sin_q2 * (2.0 * v1 * v2 + v2**2),
                a2,
                math.sin(q1),
                distal_gravity_shape,
                v1,
                0.0,
            ],
            [
                0.0,
                cos_q2 * a1 + sin_q2 * v1**2,
                a1 + a2,
                0.0,
                distal_gravity_shape,
                0.0,
                v2,
            ],
        ],
        dtype=float,
    )


def stacked_inverse_dynamics_regressor(
    q: npt.ArrayLike, velocity: npt.ArrayLike, acceleration: npt.ArrayLike
) -> FloatArray:
    """Stack the two-row regressor over matching ``(n, 2)`` samples."""
    arrays = [np.asarray(value, dtype=float) for value in (q, velocity, acceleration)]
    if (
        any(array.ndim != 2 or array.shape[1:] != (2,) for array in arrays)
        or len({array.shape for array in arrays}) != 1
        or not all(np.all(np.isfinite(array)) for array in arrays)
    ):
        raise ValueError(
            "q, velocity, and acceleration must have matching (n, 2) shapes"
        )
    if arrays[0].shape[0] == 0:
        raise ValueError("at least one sample is required")
    return np.vstack(
        [
            inverse_dynamics_regressor(position, rate, accel)
            for position, rate, accel in zip(*arrays, strict=True)
        ]
    )


def _two_vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    return array
