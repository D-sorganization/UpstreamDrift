"""Same-input cross-engine dynamics parity (epic #11605).

The same spec, initial state and torque sequence must produce the same motion
in every engine.  This package holds the engine-agnostic pieces: a vector view
of each full-body adapter and the closure projection shared by all engines.
"""

from src.shared.python.motion_matching.same_input.closure import (
    ClosureProjection,
    project_to_closure,
)
from src.shared.python.motion_matching.same_input.plant import (
    PARITY_ENGINES,
    VectorPlant,
)

__all__ = [
    "PARITY_ENGINES",
    "ClosureProjection",
    "VectorPlant",
    "project_to_closure",
]
