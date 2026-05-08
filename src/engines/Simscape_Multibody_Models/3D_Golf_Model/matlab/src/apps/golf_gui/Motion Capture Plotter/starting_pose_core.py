"""DEPRECATED — relocated to ``src.tools.starting_pose_matcher.core``.

This shim re-exports the new public API so any sibling code or test
fixture that still imports ``starting_pose_core`` from this directory
keeps working through one release cycle.  See issue #4376.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

warnings.warn(
    "starting_pose_core at this path is deprecated.  Import from "
    "``src.tools.starting_pose_matcher.core`` instead.  See issue #4376.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the new core's public surface unchanged.
# Add repo-root fallback for legacy usage where repo root is not on sys.path
# (e.g., `python -m starting_pose_matcher` from this directory)
# Path depth: Motion Capture Plotter -> golf_gui -> apps -> src -> matlab -> 3D_Golf_Model ->
#             Simscape_Multibody_Models -> engines -> src -> repo root = 9 levels
_repo_root = Path(__file__).resolve().parents[9]  # Motion Capture Plotter -> repo root
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_core = importlib.import_module("src.tools.starting_pose_matcher.core")

CM_TO_M = _core.CM_TO_M
DEFAULT_EVENT_PRESET = _core.DEFAULT_EVENT_PRESET
DEFAULT_PHASE = _core.DEFAULT_PHASE
EVENT_KEYS = _core.EVENT_KEYS
EVENT_LABEL_PRESETS = _core.EVENT_LABEL_PRESETS
FALLBACK_SEGMENTS = _core.FALLBACK_SEGMENTS
MocapEvents = _core.MocapEvents
PHASE_BOUNDS = _core.PHASE_BOUNDS
PHASE_KEYS = _core.PHASE_KEYS
PHASE_LEGACY_LABELS = _core.PHASE_LEGACY_LABELS
PoseSlot = _core.PoseSlot
RigidTransform = _core.RigidTransform
SESSION_SCHEMA_VERSION = _core.SESSION_SCHEMA_VERSION
Skeleton = _core.Skeleton
SkeletonTrajectory = _core.SkeletonTrajectory
fallback_skeleton = _core.fallback_skeleton
load_mocap_xlsx = _core.load_mocap_xlsx
load_simscape_trajectory_csv = _core.load_simscape_trajectory_csv
load_skeleton = _core.load_skeleton
phase_display_label = _core.phase_display_label
phase_key_from_label = _core.phase_key_from_label
read_event_header = _core.read_event_header
solve_shaft_rz_deg = _core.solve_shaft_rz_deg


# Legacy-compat dict aliases — older code occasionally referenced these
# constants directly.  Keep them around as derived dicts.
def _legacy_fallback_dict(pose_name: str) -> dict[str, list[float]]:
    skel = fallback_skeleton(pose_name)
    return {k: v.tolist() for k, v in skel.joints.items()}


FALLBACK_IMPACT = _legacy_fallback_dict("Impact")
FALLBACK_TOB = _legacy_fallback_dict("TopofBackswing")
