"""OpenSim tour-average marker matching (epic #10003, OS-1 onward).

Pure-Python contracts that run without the OpenSim bindings: capture label
to body mapping, TRC export/import and MarkerSet authoring. Runtime-bound
steps (IK, Moco tracking) live in drivers under
docs/development/opensim_tour_matching and skip without ``opensim``.
"""

from src.engines.physics_engines.opensim.python.tour_matching.marker_calibration import (
    CalibrationResult,
    calibrate_marker_offsets,
)
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
from src.engines.physics_engines.opensim.python.tour_matching.metrics import (
    SharedMetrics,
    compute_shared_metrics,
)
from src.engines.physics_engines.opensim.python.tour_matching.scale import (
    DEFAULT_NOMINAL_LENGTHS_M,
    SegmentScaleResult,
    estimate_segment_scales,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import (
    read_trc,
    write_trc,
)

__all__ = [
    "DEFAULT_NOMINAL_LENGTHS_M",
    "CalibrationResult",
    "GOLF_HUMANOID_MARKER_BODIES",
    "MarkerPlacement",
    "SegmentScaleResult",
    "SharedMetrics",
    "attach_marker_set",
    "body_for",
    "calibrate_marker_offsets",
    "compute_shared_metrics",
    "estimate_segment_scales",
    "labels_per_body",
    "locked_coordinates",
    "read_trc",
    "unlock_coordinates",
    "write_model",
    "write_trc",
]
