"""Compatibility facade for gap-filling helpers.

The maintained implementation lives in :mod:`gap_fill`; this module remains
importable for callers that historically selected the pure-Python fallback
directly.
"""

from __future__ import annotations

from .gap_fill import GapFillStrategy, gap_fill

__all__ = ["GapFillStrategy", "gap_fill"]
