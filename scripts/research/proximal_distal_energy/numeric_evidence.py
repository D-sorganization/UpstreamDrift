"""Portable numeric serialization for registered research evidence."""

from __future__ import annotations

from typing import Any

import numpy as np


CANONICAL_SIGNIFICANT_DIGITS = 6


def canonicalize_published_numbers(
    value: Any,
    *,
    significant_digits: int = CANONICAL_SIGNIFICANT_DIGITS,
    context: str = "published research evidence",
) -> Any:
    """Return nested data with finite floats rounded for portable identity.

    Numerical decisions must be made before this function is called.  The
    normalization applies only to publication serialization; it must never be
    used to select rank, thresholds, events, or scientific classifications.
    """
    if significant_digits < 1:
        raise ValueError("significant_digits must be positive")
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{context} must contain only finite floats")
        if numeric == 0.0:
            return 0.0
        return float(f"{numeric:.{significant_digits}g}")
    if isinstance(value, dict):
        return {
            key: canonicalize_published_numbers(
                item,
                significant_digits=significant_digits,
                context=context,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            canonicalize_published_numbers(
                item,
                significant_digits=significant_digits,
                context=context,
            )
            for item in value
        ]
    return value
