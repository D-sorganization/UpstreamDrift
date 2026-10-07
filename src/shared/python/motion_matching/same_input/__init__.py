"""Same-input cross-engine dynamics parity (epic #11605).

The same spec, initial state and torque sequence must produce the same motion
in every engine.  This package holds the engine-agnostic pieces: a vector view
of each full-body adapter, the closure projection, the shared integrator, and
the input bundle that carries a motion between engines.
"""

from src.shared.python.motion_matching.same_input.bundle import (
    SCHEMA,
    InputBundle,
)
from src.shared.python.motion_matching.same_input.closure import (
    ClosureProjection,
    project_to_closure,
)
from src.shared.python.motion_matching.same_input.integrator import (
    ROOT_COORDINATES,
    Rollout,
    integrate,
    open_loop,
    zoh_rk4_step,
)
from src.shared.python.motion_matching.same_input.plant import (
    ALL_ENGINES,
    OPTIONAL_ENGINES,
    PARITY_ENGINES,
    VectorPlant,
)
from src.shared.python.motion_matching.same_input.reference import (
    closed_loop,
    generate_reference_bundle,
)
from src.shared.python.motion_matching.same_input.scoring import (
    ReplayScore,
    growth_rate,
    score_replay,
    segmented_replay,
)

__all__ = [
    "ALL_ENGINES",
    "OPTIONAL_ENGINES",
    "PARITY_ENGINES",
    "ROOT_COORDINATES",
    "SCHEMA",
    "ClosureProjection",
    "InputBundle",
    "ReplayScore",
    "Rollout",
    "VectorPlant",
    "closed_loop",
    "generate_reference_bundle",
    "growth_rate",
    "integrate",
    "open_loop",
    "project_to_closure",
    "score_replay",
    "segmented_replay",
    "zoh_rk4_step",
]
