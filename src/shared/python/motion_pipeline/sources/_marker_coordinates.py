"""Shared helpers for source-adapter marker coordinate rows."""

from __future__ import annotations

import math


def has_nan_coordinate(x: float, y: float, z: float) -> bool:
    """Return whether a marker coordinate triplet represents an occlusion.

    C3D and TRC adapters both encode occluded marker samples as NaN coordinate
    triplets. Keep the predicate centralized so Python and Rust-backed loaders
    apply the same marker omission rule.
    """
    return math.isnan(x) or math.isnan(y) or math.isnan(z)
