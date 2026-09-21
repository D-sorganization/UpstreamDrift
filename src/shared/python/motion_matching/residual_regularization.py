"""Shared finite residual and derivative contracts for control penalties."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Penalty = Callable[[Array], Array]


def regularization_residual(parameters: Array, callback: Penalty) -> Array:
    """Flatten a finite residual, preserving the prefix fitter's existing layout."""
    result = np.asarray(callback(parameters), dtype=float).ravel()
    if not np.isfinite(result).all():
        raise ValueError("regularization residuals must be finite")
    return result


def regularization_derivative(
    parameters: Array, residual: Penalty, derivative: Penalty
) -> Array:
    """Validate rows against the actual residual and columns against controls."""
    rows = regularization_residual(parameters, residual).size
    result = np.asarray(derivative(parameters), dtype=float)
    if result.shape != (rows, parameters.size) or not np.isfinite(result).all():
        raise ValueError(
            "regularization_jacobian must match finite residual/control dimensions"
        )
    return result
