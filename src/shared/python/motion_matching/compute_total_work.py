"""Public re-export of :func:`compute_total_work`.

Mirror of ``compute_total_work.m``. The implementation lives in
:mod:`motion_matching.cost` (where the rest of the regularizer dispatch
sits); this module hoists it to the top-level package surface so callers
that only need the regularizer don't have to import the full cost module.
"""

from __future__ import annotations

from .cost import compute_total_work

__all__ = ["compute_total_work"]
