"""Finite real-array coercion shared by optimization audits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def finite_real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``value`` as a finite float64 array.

    Preconditions: ``name`` labels ``value`` in any ``ValueError``.
    Postcondition: a float64 array, or ``ValueError`` when ``value`` is
    non-numeric or non-finite.
    """
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array
