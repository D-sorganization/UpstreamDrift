"""Putting Green Simulator.

A physics-based putting green simulation with realistic ball rolling dynamics,
configurable turf properties, slope handling, and club interaction.

This module follows the Pragmatic Programmer principles:
- DRY: Shared physics from common module
- Orthogonality: Components are independent and composable
- Design by Contract: Clear pre/post conditions
- Tracer Bullets: Minimal viable implementation first
"""

from src.engines.physics_engines.putting_green.python.turf_properties import (
    GrassType,
    TurfCondition,
    TurfProperties,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
    ContourPoint,
)
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
    RollMode,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    StrokeParameters,
    PutterType,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
    SimulationResult,
)

__all__ = [
    # Turf
    "GrassType",
    "TurfCondition",
    "TurfProperties",
    # Surface
    "GreenSurface",
    "SlopeRegion",
    "ContourPoint",
    # Ball Physics
    "BallRollPhysics",
    "BallState",
    "RollMode",
    # Putter
    "PutterStroke",
    "StrokeParameters",
    "PutterType",
    # Simulator
    "PuttingGreenSimulator",
    "SimulationConfig",
    "SimulationResult",
]
