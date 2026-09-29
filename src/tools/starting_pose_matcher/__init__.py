"""Starting-Pose Matcher — align a Simscape / MuJoCo / Drake / Pinocchio
golfer model's starting pose to a motion-capture target frame before any
optimisation runs.

Public entry points:

    python -m src.tools.starting_pose_matcher          # launches the GUI

Public modules:

    .core                math, dataclasses, xlsx + skeleton + trajectory loaders
    .gui                 PyQt6 standalone QMainWindow shell
    .gui_main_widget     embeddable :class:`MainWidget` used by the launcher
    .skeleton_provider   pluggable source of model skeleton joints

Importing this package registers the
:class:`_MotionMatchPreviewEmbedAdapter` with the embeddable-tool
registry so the launcher can host the tool as a tab or dock without
spawning a separate process. Registration is guarded so reimports
(test reloads) are a quiet no-op, and wrapped in
``contextlib.suppress(ImportError)`` so headless contexts where PyQt6
is unavailable still get a usable package. See Subtask 5 / #4998 of
EPIC #4993.
"""

from __future__ import annotations

import contextlib

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

with contextlib.suppress(ImportError):
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )

    from ._embed_adapter import _MotionMatchPreviewEmbedAdapter

    _ADAPTER = _MotionMatchPreviewEmbedAdapter()
    if get_embeddable_tool(_ADAPTER.tool_id) is None:
        register_embeddable_tool(_ADAPTER)

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
