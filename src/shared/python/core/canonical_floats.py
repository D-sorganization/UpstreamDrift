"""Canonical float precision for hashed or exactly compared payloads (#14397).

Owner decision (option A, HANDOFF #49): round to a canonical precision before
hashing, so a different BLAS kernel or system libm cannot move a hash.

:func:`canonical_floats` rounds every float in a JSON-like tree to
``CANONICAL_SIGNIFICANT_DIGITS`` significant digits and to a multiple of
``CANONICAL_ABSOLUTE_RESOLUTION``, whichever is coarser, and folds signed zero
to ``0.0``.

Why this precision: the measured cross-kernel drift (AVX-512 vs generic
OpenBLAS; ubuntu-24.04 vs 26.04 libm) is at most 2.9e-14 absolute on order-one
values, about 1e-16 relative.  Cancellation residuals (rotation entries near
1e-8 or 1e-16) drift by about 4e-18 absolute but up to 1e-11 relative, which a
relative rounding alone cannot fold, hence the absolute resolution.  Twelve
digits and a 1e-12 resolution sit at least four orders of magnitude above that
drift and far below any modelled quantity's precision (metres, kilograms,
kg m^2, radians).  A value lying exactly on a rounding boundary can still flip;
that is not observed in the hashed payloads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = [
    "CANONICAL_ABSOLUTE_RESOLUTION",
    "CANONICAL_SIGNIFICANT_DIGITS",
    "canonical_floats",
]

CANONICAL_SIGNIFICANT_DIGITS = 12
CANONICAL_ABSOLUTE_RESOLUTION = 1e-12
_ABSOLUTE_DECIMALS = 12


def _canonical_float(value: float) -> float:
    if not math.isfinite(value):
        return value
    significant = float(format(value, f".{CANONICAL_SIGNIFICANT_DIGITS}g"))
    return round(significant, _ABSOLUTE_DECIMALS) + 0.0  # + 0.0 folds -0.0


def canonical_floats(obj: Any) -> Any:
    """``obj`` with every float rounded to the canonical precision.

    Mappings and lists/tuples are rebuilt (tuples become lists, as JSON would);
    bools, ints, strings and ``None`` pass through unchanged.  Non-finite floats
    are kept as they are.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return _canonical_float(obj)
    if isinstance(obj, Mapping):
        return {key: canonical_floats(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [canonical_floats(value) for value in obj]
    return obj
