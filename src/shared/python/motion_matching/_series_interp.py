"""Shared dense-series interpolation helpers for motion-matching loaders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interp_xyz_series(
    query_t: NDArray[np.floating] | np.ndarray,
    raw_t: NDArray[np.floating] | np.ndarray,
    raw_xyz: NDArray[np.floating] | np.ndarray,
) -> NDArray[np.float64]:
    """Linear interpolation of an ``(N, 3)`` series onto ``query_t`` (endpoint clamp)."""
    query = np.asarray(query_t, dtype=np.float64)
    source_t = np.asarray(raw_t, dtype=np.float64)
    source_xyz = np.asarray(raw_xyz, dtype=np.float64)
    out = np.empty((query.shape[0], 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(query, source_t, source_xyz[:, k])
    return out
