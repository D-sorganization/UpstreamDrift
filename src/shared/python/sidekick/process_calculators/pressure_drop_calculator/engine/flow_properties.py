"""Compatibility facade for pressure-drop flow helpers.

The canonical implementation lives in ``._flow_calculations``.  Keep this
module as an import-stable facade so legacy callers do not fork the flow
calculation logic.
"""

from __future__ import annotations

from ._flow_calculations import (
    GRAVITY,
    PI,
    calculate_elevation_pressure_drop,
    calculate_erosional_velocity,
    calculate_flow_properties,
    calculate_frictional_pressure_drop,
    classify_flow_regime,
)

__all__ = [
    "GRAVITY",
    "PI",
    "calculate_elevation_pressure_drop",
    "calculate_erosional_velocity",
    "calculate_flow_properties",
    "calculate_frictional_pressure_drop",
    "classify_flow_regime",
]
