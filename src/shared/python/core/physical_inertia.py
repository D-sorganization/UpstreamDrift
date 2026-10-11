"""Lightweight physical inertia tensor validation shared by native adapters."""

from __future__ import annotations

import numpy as np

from src.shared.python.core.contracts import PreconditionError


def validate_physical_inertia(tensor: np.ndarray) -> np.ndarray:
    """Verify finite symmetry, positive definiteness and principal-moment bounds."""
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise PreconditionError("Inertia tensor must be finite 3x3 matrix")

    if not np.allclose(arr, arr.T, atol=1e-7, rtol=0):
        raise PreconditionError("Inertia tensor must be symmetric")

    eigvals = np.linalg.eigvalsh(arr)
    if np.any(eigvals <= 1e-9):
        raise PreconditionError("Inertia tensor is not positive definite")

    first, second, third = (float(value) for value in eigvals)
    tolerance = 1e-9
    if first + second < third - tolerance:
        raise PreconditionError(
            "Inertia tensor violates triangle inequality on principal moments: "
            f"I1={first}, I2={second}, I3={third}"
        )
    return arr
