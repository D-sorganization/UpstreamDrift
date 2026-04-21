"""Terrain physics helpers.

.. deprecated::
    Import directly from :mod:`src.shared.python.physics.terrain` instead.
    This module is kept for backwards compatibility and re-exports the
    canonical implementations from the ``terrain`` sub-package.
"""

from __future__ import annotations

from src.shared.python.physics.terrain import (
    compute_gravity_on_slope,
    compute_roll_direction,
    get_contact_normal,
)

__all__ = [
    "compute_gravity_on_slope",
    "compute_roll_direction",
    "get_contact_normal",
]
