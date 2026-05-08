"""Starting-Pose Matcher — align a Simscape / MuJoCo / Drake / Pinocchio
golfer model's starting pose to a motion-capture target frame before any
optimisation runs.

Public entry points:

    python -m src.tools.starting_pose_matcher          # launches the GUI

Public modules:

    .core                math, dataclasses, xlsx + skeleton + trajectory loaders
    .gui                 PyQt6 QMainWindow
    .skeleton_provider   pluggable source of model skeleton joints
"""

from .core import (
    CM_TO_M,
    DEFAULT_EVENT_PRESET,
    DEFAULT_PHASE,
    EVENT_KEYS,
    EVENT_LABEL_PRESETS,
    PHASE_BOUNDS,
    PHASE_KEYS,
    SESSION_SCHEMA_VERSION,
    MocapEvents,
    PoseSlot,
    RigidTransform,
    Skeleton,
    SkeletonTrajectory,
    load_mocap_xlsx,
    load_simscape_trajectory_csv,
    load_skeleton,
    phase_display_label,
    phase_key_from_label,
    read_event_header,
    solve_shaft_rz_deg,
)

__all__ = [
    "CM_TO_M",
    "DEFAULT_EVENT_PRESET",
    "DEFAULT_PHASE",
    "EVENT_KEYS",
    "EVENT_LABEL_PRESETS",
    "MocapEvents",
    "PHASE_BOUNDS",
    "PHASE_KEYS",
    "PoseSlot",
    "RigidTransform",
    "SESSION_SCHEMA_VERSION",
    "Skeleton",
    "SkeletonTrajectory",
    "load_mocap_xlsx",
    "load_simscape_trajectory_csv",
    "load_skeleton",
    "phase_display_label",
    "phase_key_from_label",
    "read_event_header",
    "solve_shaft_rz_deg",
]
