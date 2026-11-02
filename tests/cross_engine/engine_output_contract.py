"""Strict output checks shared by real-engine parity cases."""

from __future__ import annotations

import numpy as np


def require_engine_array(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != expected_shape
        or not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value)
        or not np.isfinite(value).all()
    ):
        raise AssertionError(
            f"{name} must be a finite real numeric numpy array of shape "
            f"{expected_shape}; got {type(value).__name__} "
            f"{getattr(value, 'shape', None)}"
        )
    return value
