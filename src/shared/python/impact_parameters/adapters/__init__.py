"""Per-engine ClubheadSeries adapters (GCV-16, #11722).

Engine modules import their physics package lazily inside each function so
this package is importable without any engine installed.
"""

from .club_face import NATIVE_CLUB_FACE, ClubFaceSpec, rigid_body_series
from .drake_adapter import clubhead_series_from_drake
from .impact_time import select_impact_index
from .mujoco_adapter import clubhead_series_from_mujoco, clubhead_series_from_myosuite
from .opensim_adapter import clubhead_series_from_opensim
from .pinocchio_adapter import clubhead_series_from_pinocchio
from .simscape_adapter import clubhead_series_from_simscape

__all__ = [
    "NATIVE_CLUB_FACE",
    "ClubFaceSpec",
    "clubhead_series_from_drake",
    "clubhead_series_from_mujoco",
    "clubhead_series_from_myosuite",
    "clubhead_series_from_opensim",
    "clubhead_series_from_pinocchio",
    "clubhead_series_from_simscape",
    "rigid_body_series",
    "select_impact_index",
]
