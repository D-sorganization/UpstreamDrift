"""Explicit validation helpers for pendulum simulator public contracts."""

from __future__ import annotations

from typing import Any

import numpy as np


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a public contract condition is false."""
    if not condition:
        raise ValueError(message)


def require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    """Validate an array-like object's exact shape."""
    actual = getattr(value, "shape", None)
    require(actual == expected, f"{name} shape must be {expected}, got {actual}")


def require_finite_array(name: str, value: np.ndarray) -> None:
    """Validate that an array has only finite values."""
    require(bool(np.all(np.isfinite(value))), f"{name} must be finite")


def require_positive(name: str, value: float) -> None:
    """Validate that a scalar is strictly positive."""
    require(value > 0, f"{name} must be positive, got {value}")


def require_non_negative(name: str, value: float) -> None:
    """Validate that a scalar is non-negative."""
    require(value >= 0, f"{name} must be non-negative, got {value}")


def require_dt(dt: float, t_end: float) -> None:
    """Validate a simulation output time step against the integration horizon."""
    require(0 < dt < t_end, f"dt must be in (0, t_end), got {dt}")
