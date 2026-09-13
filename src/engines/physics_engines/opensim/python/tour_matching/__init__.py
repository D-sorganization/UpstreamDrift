"""OpenSim tour-average marker matching (epic #10003, OS-1 onward).

Pure-Python contracts that run without the OpenSim bindings: capture label
to body mapping, TRC export/import and MarkerSet authoring. Runtime-bound
steps (IK, Moco tracking) live in drivers under
docs/development/opensim_tour_matching and skip without ``opensim``.
"""

from src.engines.physics_engines.opensim.python.tour_matching.marker_map import (
    GOLF_HUMANOID_MARKER_BODIES,
    body_for,
    labels_per_body,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (
    MarkerPlacement,
    attach_marker_set,
    locked_coordinates,
    unlock_coordinates,
    write_model,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import (
    read_trc,
    write_trc,
)

__all__ = [
    "GOLF_HUMANOID_MARKER_BODIES",
    "MarkerPlacement",
    "attach_marker_set",
    "body_for",
    "labels_per_body",
    "locked_coordinates",
    "read_trc",
    "unlock_coordinates",
    "write_model",
    "write_trc",
]
