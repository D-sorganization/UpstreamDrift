"""Array and numeric validation helpers for contract checks.

Provides reusable predicates for common numeric constraints:
``check_finite``, ``check_shape``, ``check_positive``,
``check_non_negative``, ``check_symmetric``, ``check_positive_definite``.
"""

from __future__ import annotations

import numpy as np


def check_finite(arr: np.ndarray | None) -> bool:
    """Check if array contains only finite values (no NaN or Inf).

    Args:
        arr: NumPy array to check.

    Returns:
        True if array is not None and all values are finite.
    """
    if arr is None:
        return False
    return bool(np.all(np.isfinite(arr)))


def check_shape(arr: np.ndarray | None, expected_shape: tuple[int, ...]) -> bool:
    """Check if array has expected shape.

    Args:
        arr: NumPy array to check.
        expected_shape: Expected shape tuple.

    Returns:
        True if array has the expected shape.
    """
    if arr is None:
        return False
    return arr.shape == expected_shape


def check_positive(value: float | int | np.ndarray) -> bool:
    """Check if value(s) are positive.

    Args:
        value: Scalar or array to check.

    Returns:
        True if all values are > 0.
    """
    if isinstance(value, np.ndarray):
        return bool(np.all(value > 0))
    return value > 0


def check_non_negative(value: float | int | np.ndarray) -> bool:
    """Check if value(s) are non-negative.

    Args:
        value: Scalar or array to check.

    Returns:
        True if all values are >= 0.
    """
    if isinstance(value, np.ndarray):
        return bool(np.all(value >= 0))
    return value >= 0


def check_symmetric(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is symmetric within tolerance.

    Args:
        matrix: 2D NumPy array to check.
        tol: Tolerance for symmetry check.

    Returns:
        True if matrix is symmetric.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return bool(np.allclose(matrix, matrix.T, atol=tol))


def check_positive_definite(matrix: np.ndarray) -> bool:
    """Check if matrix is positive definite.

    Args:
        matrix: 2D symmetric NumPy array to check.

    Returns:
        True if all eigenvalues are positive.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > 0))
    except np.linalg.LinAlgError:
        return False
