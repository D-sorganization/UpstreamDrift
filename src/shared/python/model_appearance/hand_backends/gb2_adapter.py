"""Adapter for the unmerged GB2 curl builder (#15134).

GB2 is lead-finger only and is not on this branch. When it can be imported,
its angles are recorded beside the local solution and are not applied. The
local tangent pose is what the contact tests judge.
"""

from __future__ import annotations

import importlib
from typing import Any

from src.shared.python.grip_spec.types import GripSpec

GB2_ISSUE = 15134
GB2_BRANCH = "cursor/hrg-gb2-grip-builder-0932"
GB2_MODULE = "src.shared.python.grip_build"
_ABSENT = (
    "GB2 (#15134, branch cursor/hrg-gb2-grip-builder-0932) is not importable "
    "on this branch. The local tangent solution is the contact pose. Do not "
    "copy ClubSDF.signed_distance_capsule: it keeps the maximum sampled "
    "distance, not the minimum clearance. Do not place a phalanx along the "
    "shaft and hinge it about the shaft; that rotation does not wrap the grip."
)


def gb2_curl_status(spec: GripSpec | None = None) -> dict[str, Any]:
    """Availability of GB2 and, when present, its lead-finger angles.

    Preconditions: ``spec`` is a GripSpec or None. Postcondition:
    ``applied`` is false. A missing package returns ``available`` false and
    does not invent angles.
    """
    if spec is not None and not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be a GripSpec or None, got {type(spec).__name__}")
    try:
        module = importlib.import_module(GB2_MODULE)
    except (ImportError, ModuleNotFoundError):
        return _status(False, None, None, _ABSENT)
    builder = getattr(module, "build_lead_fingers", None)
    version = getattr(module, "BUILDER_VERSION", None)
    if builder is None or spec is None:
        return _status(
            True,
            version,
            None,
            "GB2 imported without a usable build_lead_fingers result.",
        )
    result = builder(spec)
    angles = getattr(result, "joint_angles_rad", None)
    return _status(
        True,
        version,
        angles,
        "Recorded for comparison. Local contact angles are not replaced.",
    )


def _status(
    available: bool,
    version: str | None,
    angles: object,
    note: str,
) -> dict[str, Any]:
    return {
        "available": available,
        "applied": False,
        "dependency": f"UpstreamDrift#{GB2_ISSUE}",
        "branch": GB2_BRANCH,
        "builder_version": version,
        "angles_rad": angles,
        "note": note,
    }


__all__ = [
    "GB2_BRANCH",
    "GB2_ISSUE",
    "GB2_MODULE",
    "gb2_curl_status",
]
