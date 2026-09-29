# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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

from src.engines.physics_engines.putting_green.python._checkpoint import (
    PUTTING_CHECKPOINT_SCHEMA_VERSION,
    CheckpointProvenance,
    read_checkpoint_provenance,
)
from src.engines.physics_engines.putting_green.python._sim_config import (
    SimulationConfig,
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python._sim_core import (
    PuttingGreenSimulator,
)
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    KNOWN_ROLL_MODELS,
    ROLL_MODEL_FIELD,
    UD_LEGACY_ROLL_MODEL,
    USGA_STIMP_ROLL_MODEL,
    RollModelProvenanceError,
    require_roll_model,
)

__all__ = [
    "KNOWN_ROLL_MODELS",
    "PUTTING_CHECKPOINT_SCHEMA_VERSION",
    "ROLL_MODEL_FIELD",
    "UD_LEGACY_ROLL_MODEL",
    "USGA_STIMP_ROLL_MODEL",
    "CheckpointProvenance",
    "RollModelProvenanceError",
    "SimulationConfig",
    "SimulationResult",
    "PuttingGreenSimulator",
    "read_checkpoint_provenance",
    "require_roll_model",
]
