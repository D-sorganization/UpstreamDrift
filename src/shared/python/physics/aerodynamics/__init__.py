"""Aerodynamics package for golf ball flight simulation.

Public API is re-exported from sub-modules for backward compatibility.
Import any class directly from this package:

    from src.shared.python.physics.aerodynamics import AerodynamicsEngine

Design Principles:
- Reversible: All effects toggleable at runtime
- Reusable: Modular components that compose well
- DRY: Shared calculations in dedicated modules
- Orthogonal: Independent force models with no hidden coupling

References:
    - Bearman & Harvey (1976). Golf ball aerodynamics.
    - Smits & Ogg (2004). Golf ball aerodynamics. Physics Today.
    - Jorgensen (1999). The Physics of Golf. Springer.
"""

from __future__ import annotations

from ._config import (
    MIN_AIR_DENSITY_KG_M3,
    AerodynamicsConfig,
    RandomizationConfig,
    WindConfig,
)
from ._engine import AerodynamicsEngine
from ._environment import EnvironmentRandomizer, EnvironmentSnapshot
from ._models import DragModel, LiftModel, MagnusModel
from ._rust_facade import (
    AerodynamicsSpec,
    compute_acceleration,
    compute_total_force,
    is_rust_available,
)
from ._wind import TurbulenceModel, WindGust, WindModel

__all__ = [
    "AerodynamicsConfig",
    "AerodynamicsEngine",
    "AerodynamicsSpec",
    "DragModel",
    "EnvironmentRandomizer",
    "EnvironmentSnapshot",
    "LiftModel",
    "MIN_AIR_DENSITY_KG_M3",
    "MagnusModel",
    "RandomizationConfig",
    "TurbulenceModel",
    "WindConfig",
    "WindGust",
    "WindModel",
    "compute_acceleration",
    "compute_total_force",
    "is_rust_available",
]
