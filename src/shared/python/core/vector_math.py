"""Small vectorized math helpers shared by physics and validation code."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def row_euclidean_norm(values: object) -> NDArray[np.float64]:
    """Return the Euclidean norm for each row of a 2-D real numeric array.

    The result has shape ``(values.shape[0],)`` and matches
    ``np.linalg.norm(values, axis=1)`` while avoiding NumPy's heavier norm
    dispatch for small fixed-width row vectors.
    """
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"values must be a 2-D array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise TypeError("values must be a real numeric array")
    arr64 = arr.astype(np.float64, copy=False)
    return np.sqrt(np.einsum("ij,ij->i", arr64, arr64))
