"""Putting Green Simulator - Main Physics Engine.

This module implements the main PuttingGreenSimulator class that provides
a complete putting simulation conforming to the PhysicsEngine protocol.

Features:
    - Real-time ball rolling simulation
    - Configurable turf and surface properties
    - Support for topographical data loading
    - Putter stroke simulation
    - Trajectory recording and replay
    - Wind effects (optional)
    - Practice mode with feedback

Design by Contract:
    - Follows PhysicsEngine protocol
    - Thread-safe state management
    - Deterministic simulation (same inputs = same outputs)
"""

from src.engines.physics_engines.putting_green.python._sim_config import (
    SimulationConfig,
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python._sim_core import (
    PuttingGreenSimulator,
)

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "PuttingGreenSimulator",
]
