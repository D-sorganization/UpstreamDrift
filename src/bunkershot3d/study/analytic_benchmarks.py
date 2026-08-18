"""Analytic test functions with closed-form Sobol' indices.

These are the oracles for the sensitivity estimators. Every value returned by
:func:`ishigami_indices` and :func:`sobol_g_indices` is *derived*, not
tabulated, so the reference cannot drift away from its own definition — the
published decimal values are only used in the tests as an independent
cross-check.

References:
    Ishigami, T. and Homma, T. (1990). An importance quantification technique
    in uncertainty analysis for computer models. *ISUMA '90*, 398-403.

    Sobol', I. M. (1993). Sensitivity estimates for nonlinear mathematical
    models. *Mathematical Modelling and Computational Experiments*, 1, 407-414.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design_space import DesignSpace

__all__ = [
    "AnalyticIndices",
    "ISHIGAMI_A",
    "ISHIGAMI_B",
    "ishigami",
    "ishigami_indices",
    "ishigami_space",
    "sobol_g",
    "sobol_g_indices",
    "sobol_g_space",
]

#: Standard Ishigami coefficients used by the published index values.
ISHIGAMI_A = 7.0
ISHIGAMI_B = 0.1


@dataclass(frozen=True, eq=False)
class AnalyticIndices:
    """Closed-form variance decomposition of a benchmark function.

    Attributes:
        first_order: ``(d,)`` analytic first-order indices ``S1``.
        total_order: ``(d,)`` analytic total-order indices ``ST``.
        variance: The total output variance.
    """

    first_order: np.ndarray
    total_order: np.ndarray
    variance: float


def ishigami_space() -> DesignSpace:
    """Build the Ishigami design space, ``U(-pi, pi)`` in three dimensions.

    Returns:
        A three-parameter design space named ``x1``, ``x2``, ``x3``.
    """
    return DesignSpace.from_bounds(
        {
            "x1": (-np.pi, np.pi),
            "x2": (-np.pi, np.pi),
            "x3": (-np.pi, np.pi),
        }
    )


def ishigami(
    points: np.ndarray,
    a: float = ISHIGAMI_A,
    b: float = ISHIGAMI_B,
) -> np.ndarray:
    """Evaluate the Ishigami function.

    ``f(x) = sin(x1) + a sin^2(x2) + b x3^4 sin(x1)``

    Args:
        points: ``(n, 3)`` array of inputs in ``[-pi, pi]^3``.
        a: Coefficient of the ``sin^2(x2)`` term.
        b: Coefficient of the ``x3^4 sin(x1)`` term.

    Returns:
        A ``(n,)`` array of outputs.

    Raises:
        ValueError: If ``points`` does not have three columns.
    """
    matrix = np.atleast_2d(np.asarray(points, dtype=float))
    if matrix.shape[1] != 3:
        raise ValueError(f"ishigami needs 3 columns, got {matrix.shape[1]}")
    x1, x2, x3 = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * (x3**4) * np.sin(x1)


def ishigami_indices(
    a: float = ISHIGAMI_A,
    b: float = ISHIGAMI_B,
) -> AnalyticIndices:
    """Compute the exact Sobol' indices of the Ishigami function.

    The partial variances are

    ``V1 = b pi^4 / 5 + b^2 pi^8 / 50 + 1/2``,
    ``V2 = a^2 / 8``, ``V3 = 0``,
    ``V13 = 8 b^2 pi^8 / 225``,

    with all remaining interaction terms zero, so
    ``ST = [V1 + V13, V2, V13] / V``.

    Args:
        a: Coefficient of the ``sin^2(x2)`` term.
        b: Coefficient of the ``x3^4 sin(x1)`` term.

    Returns:
        The analytic first-order and total-order indices.
    """
    pi4 = np.pi**4
    pi8 = np.pi**8
    v1 = b * pi4 / 5.0 + b**2 * pi8 / 50.0 + 0.5
    v2 = a**2 / 8.0
    v3 = 0.0
    v13 = 8.0 * b**2 * pi8 / 225.0
    total_variance = v1 + v2 + v3 + v13
    first = np.array([v1, v2, v3]) / total_variance
    total = np.array([v1 + v13, v2, v3 + v13]) / total_variance
    return AnalyticIndices(
        first_order=first,
        total_order=total,
        variance=float(total_variance),
    )


def sobol_g_space(n_factors: int) -> DesignSpace:
    """Build the g-function design space, ``U(0, 1)^n``.

    Args:
        n_factors: Number of factors ``d``.

    Returns:
        A design space with parameters ``x1 .. xd`` on ``[0, 1]``.

    Raises:
        ValueError: If ``n_factors`` is not positive.
    """
    if n_factors <= 0:
        raise ValueError(f"n_factors must be positive, got {n_factors}")
    return DesignSpace.from_bounds({f"x{i + 1}": (0.0, 1.0) for i in range(n_factors)})


def sobol_g(points: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Evaluate Sobol's g-function.

    ``g(x) = prod_i (|4 x_i - 2| + a_i) / (1 + a_i)``

    A large ``a_i`` makes factor ``i`` unimportant; ``a_i = 0`` makes it
    dominant.

    Args:
        points: ``(n, d)`` array of inputs in ``[0, 1]^d``.
        a: ``(d,)`` array of non-negative shape coefficients.

    Returns:
        A ``(n,)`` array of outputs.

    Raises:
        ValueError: If the shapes disagree or any ``a_i`` is negative.
    """
    matrix = np.atleast_2d(np.asarray(points, dtype=float))
    coeffs = np.asarray(a, dtype=float)
    if coeffs.ndim != 1:
        raise ValueError(f"a must be 1-dimensional, got {coeffs.ndim}D")
    if matrix.shape[1] != coeffs.size:
        raise ValueError(
            f"points has {matrix.shape[1]} columns but a has {coeffs.size} entries"
        )
    if np.any(coeffs < 0.0):
        raise ValueError("a must be non-negative")
    factors = (np.abs(4.0 * matrix - 2.0) + coeffs) / (1.0 + coeffs)
    return np.prod(factors, axis=1)


def sobol_g_indices(a: np.ndarray) -> AnalyticIndices:
    """Compute the exact Sobol' indices of the g-function.

    With ``V_i = (1/3) / (1 + a_i)^2`` the total variance is
    ``V = prod_i (1 + V_i) - 1`` and

    ``S_i = V_i / V``,
    ``ST_i = V_i prod_{j != i} (1 + V_j) / V``.

    Args:
        a: ``(d,)`` array of non-negative shape coefficients.

    Returns:
        The analytic first-order and total-order indices.

    Raises:
        ValueError: If any ``a_i`` is negative.
    """
    coeffs = np.asarray(a, dtype=float)
    if np.any(coeffs < 0.0):
        raise ValueError("a must be non-negative")
    partial = (1.0 / 3.0) / (1.0 + coeffs) ** 2
    total_variance = float(np.prod(1.0 + partial) - 1.0)
    product_all = np.prod(1.0 + partial)
    total = partial * (product_all / (1.0 + partial)) / total_variance
    return AnalyticIndices(
        first_order=partial / total_variance,
        total_order=total,
        variance=total_variance,
    )
