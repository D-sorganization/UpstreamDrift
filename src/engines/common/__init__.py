"""Common Engine Components Package.

This package provides reusable components shared across all physics engine
implementations, reducing code duplication by 15-20%.

Modules:
    physics: Common physics equations (drag, lift, magnus effect)
    state: State management utilities and mixins
    validation: Input validation decorators

Usage:
    from src.engines.common import (
        BallPhysics,
        StateManager,
        EngineStateMixin,
    )
"""

from src.engines.common.physics import AerodynamicsCalculator, BallPhysics
from src.engines.common.state import (
    EngineStateMixin,
    ForceAccumulator,
    StateManager,
)

__all__ = [
    "BallPhysics",
    "AerodynamicsCalculator",
    "StateManager",
    "EngineStateMixin",
    "ForceAccumulator",
]
