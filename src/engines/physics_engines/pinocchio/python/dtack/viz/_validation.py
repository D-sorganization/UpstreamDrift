"""Shared input validation for optional Pinocchio viewer adapters."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np


def validated_configuration(
    model: Any, q: Any, neutral: Callable[[Any], Any]
) -> np.ndarray:
    """Return a finite, one-dimensional configuration for *model*.

    ``q=None`` is resolved by the caller's Pinocchio ``neutral`` function.
    Keeping this check in one place prevents the optional adapters from
    drifting apart while still allowing each backend to own its lifecycle.
    """
    try:
        raw_nq = model.nq
    except AttributeError as exc:
        raise ValueError("model.nq must be a non-negative integer") from exc
    if isinstance(raw_nq, bool) or not isinstance(raw_nq, Integral):
        raise ValueError("model.nq must be a non-negative integer")
    nq = int(raw_nq)
    if nq < 0:
        raise ValueError("model.nq must be a non-negative integer")

    values = neutral(model) if q is None else q
    try:
        configuration = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("q must contain numeric values") from exc
    if configuration.shape != (nq,):
        raise ValueError(f"q shape {configuration.shape} does not match ({nq},)")
    if not bool(np.all(np.isfinite(configuration))):
        raise ValueError("q must contain only finite values")
    return configuration
