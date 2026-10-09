"""JSON-safe plot traces shared by the force plot models (GCV-5, #11711).

A sample that cannot be computed is NaN in the arrays and ``None`` in the
trace, never zero, so the PyQt plots and the web charts draw a gap.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["none_if_nan", "vector_trace"]


def none_if_nan(values: np.ndarray) -> list[float | None]:
    """Return ``values`` as floats with every non-finite entry as ``None``."""
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def vector_trace(arr: np.ndarray) -> dict[str, list[float | None]]:
    """Split a ``(T, 3)`` array into ``x``, ``y``, ``z`` and ``magnitude`` lists.

    Postcondition: each list has ``T`` entries; a row with any NaN component
    has a ``None`` magnitude.

    Raises:
        ValueError: if ``arr`` is not shaped ``(T, 3)``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"trace array must be shaped (T, 3), got {arr.shape}")
    mag = np.linalg.norm(arr, axis=1)  # NaN rows stay NaN
    return {
        "x": none_if_nan(arr[:, 0]),
        "y": none_if_nan(arr[:, 1]),
        "z": none_if_nan(arr[:, 2]),
        "magnitude": none_if_nan(mag),
    }
