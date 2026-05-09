"""Pose Studio — interactive cross-engine pose editor.

A single-purpose PyQt6 tool that lets the user hand-edit a
:class:`CanonicalPose` and see the result rendered through any of the
supported engines without restarting.

Public entry point::

    python -m src.tools.pose_studio

Public modules:

    .core            engine-agnostic pure-data primitives, no Qt
    .controllers     EngineController + HistoryController (Qt-free)
    .widgets         per-component PyQt6 widgets
    .gui             :class:`PoseStudioWindow` + ``main()``

Save/load is stubbed in v1; real save/load lives in Subtask 6 / #4900.
IK drag-handles are deferred to a follow-up issue.
"""

from __future__ import annotations

from src.tools.pose_studio.core import (
    JOINT_REGION_LAYOUT,
    SUPPORTED_ENGINES,
    EngineStatus,
    joint_region_partitions_reference_fields,
)

__all__ = [
    "JOINT_REGION_LAYOUT",
    "SUPPORTED_ENGINES",
    "EngineStatus",
    "joint_region_partitions_reference_fields",
]
