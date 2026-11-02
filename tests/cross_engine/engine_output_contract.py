"""Strict output checks shared by real-engine parity cases."""

from __future__ import annotations

import numpy as np


def require_engine_array(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Reject malformed real-engine outputs instead of skipping the parity case."""

    if not isinstance(value, np.ndarray):
        raise AssertionError(
            f"{name} must be a numpy array, got {type(value).__name__}"
        )
    if value.shape != expected_shape:
        raise AssertionError(
            f"{name} must have shape {expected_shape}, got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise AssertionError(f"{name} must contain real numeric values")
    if not np.isfinite(value).all():
        raise AssertionError(f"{name} must contain only finite values")
    return value
