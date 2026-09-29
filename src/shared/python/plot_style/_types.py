"""Internal type aliases shared by plot_style modules.

These aliases are not part of the public API and may change without
notice. Public re-exports live in :mod:`src.shared.python.plot_style`.
"""

from __future__ import annotations

from typing import Final

# (r, g, b, a) tuple with each component in [0, 1].
RGBATuple = tuple[float, float, float, float]

# Sentinel returned for non-finite / out-of-range channel values.
NAN_RGBA_FALLBACK: Final[RGBATuple] = (0.5333, 0.5333, 0.5333, 1.0)

__all__ = ["NAN_RGBA_FALLBACK", "RGBATuple"]
