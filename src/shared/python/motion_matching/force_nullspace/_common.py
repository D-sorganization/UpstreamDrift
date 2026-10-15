"""Shared array validation helpers for force null-space modules (#12552)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]


def _array(value: Array, ndim: int, name: str) -> Array:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result
