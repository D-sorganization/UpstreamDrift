"""Pure-data state machine for the Pose Studio tool.

This module is **engine-agnostic** and contains **no Qt imports**.  It
exposes the small pure-data primitives that the rest of the package
composes:

* :data:`SUPPORTED_ENGINES` — ordered list of engine names available in
  the engine picker (pulled from the registries shipped by Subtasks 2/3
  of EPIC #4895).
* :class:`EngineStatus` — value enum reported by the controllers and
  rendered as a colour pill in the UI.
* :data:`JOINT_REGION_LAYOUT` — the canonical joint-name → body-region
  grouping used by the joint accordion.  Lives here so the unit tests
  can assert that every canonical field belongs to exactly one region.

Everything in this file is deliberately import-cheap so that the unit
tests can exercise the controllers without spinning up Qt.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Final

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange.adapters import ADAPTER_REGISTRY
from src.shared.python.pose_interchange.services import (
    KINEMATICS_SERVICE_REGISTRY,
)


class EngineStatus(str, Enum):
    """Status pill values rendered by :class:`EnginePicker`.

    Members map 1-to-1 to the colours rendered by the status pill:

    * ``MOCK`` — yellow.  The engine wheel is not installed; we fell
      back to :class:`MockKinematicsService`.
    * ``LIVE`` — green.  Real engine wheel was loaded.
    * ``ERROR`` — red.  The factory raised when constructing the
      service or the adapter for the requested engine.
    """

    MOCK = "mock"
    LIVE = "live"
    ERROR = "error"


# Engine names exposed by the picker, in display order.  Anchored on
# the intersection of the two registries shipped by Subtasks 2 and 3 so
# we never offer an engine that lacks either an adapter or a service.
SUPPORTED_ENGINES: Final[tuple[str, ...]] = tuple(
    name
    for name in ("drake", "mujoco", "pinocchio", "opensim", "simscape")
    if name in ADAPTER_REGISTRY and name in KINEMATICS_SERVICE_REGISTRY
)


# Body-region grouping for the joint accordion.  The values are the
# canonical field names from :data:`REFERENCE_GOLFER_FIELDS`; together
# the regions partition the field tuple (verified by a unit test).
JOINT_REGION_LAYOUT: Final[Mapping[str, tuple[str, ...]]] = {
    "Pelvis": (
        "HipStartPositionX",
        "HipStartPositionY",
        "HipStartPositionZ",
    ),
    "Spine": (
        "SpineStartPositionX",
        "SpineStartPositionY",
        "TorsoStartPosition",
    ),
    "Shoulders": (
        "LScapStartPositionX",
        "LScapStartPositionY",
        "RScapStartPositionX",
        "RScapStartPositionY",
        "LSStartPositionX",
        "LSStartPositionY",
        "LSStartPositionZ",
        "RSStartPositionX",
        "RSStartPositionY",
        "RSStartPositionZ",
    ),
    "Elbows": (
        "LEStartPosition",
        "REStartPosition",
        "LFStartPosition",
        "RFStartPosition",
    ),
    "Wrists": (
        "LWStartPositionX",
        "LWStartPositionY",
        "RWStartPositionX",
        "RWStartPositionY",
    ),
}


def joint_region_partitions_reference_fields() -> bool:
    """Return ``True`` iff :data:`JOINT_REGION_LAYOUT` partitions the
    canonical field tuple :data:`REFERENCE_GOLFER_FIELDS`.

    Used by the unit tests; exposed as a function so the property is
    cheap to assert from the GUI layer too.
    """
    flat = tuple(name for region in JOINT_REGION_LAYOUT.values() for name in region)
    return set(flat) == set(REFERENCE_GOLFER_FIELDS) and len(flat) == len(
        REFERENCE_GOLFER_FIELDS
    )


__all__ = [
    "JOINT_REGION_LAYOUT",
    "SUPPORTED_ENGINES",
    "EngineStatus",
    "joint_region_partitions_reference_fields",
]
