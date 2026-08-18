"""SI unit strings for the sand-field quantities (issue #8710).

One module so the container and the standing can both name a unit
without either importing the other, and so a unit string exists in
exactly one place: a view that prints "kg/m^3" from its own literal is a
second source of truth waiting to drift.
"""

from __future__ import annotations

__all__ = [
    "DENSITY_UNIT",
    "SHEAR_RATE_UNIT",
    "TIME_UNIT",
    "VELOCITY_UNIT",
]

VELOCITY_UNIT = "m/s"
DENSITY_UNIT = "kg/m^3"
SHEAR_RATE_UNIT = "1/s"
TIME_UNIT = "s"
