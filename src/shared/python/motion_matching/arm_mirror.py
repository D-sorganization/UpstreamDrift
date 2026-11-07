"""Right-arm joint-axis mirroring (G1, #14746).

The native Simscape export gives the left and right arm joints *identical*
axis vectors, so a positive right coordinate is the mirror image of a negative
left one for every in-plane axis. A sagittal reflection maps a rotation about
axis ``a`` to one about ``-M a`` (``M = diag(1, -1, 1)``): in-plane axes
(x, z) flip, the lateral axis (y) keeps its sign.

Conjugating a joint by a half turn ``F`` (right-multiplying both
``parent_to_base`` and ``child_to_follower``) leaves the zero pose and every
child frame unchanged but sends ``Rk(q)`` to ``Rk(+-q)``, so the physical motion
is identical when the stored coordinate is negated. ``F = Ry(pi)`` flips Rx and
Rz; ``F = Rx(pi)`` flips Ry and Rz.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.shared.python.contracts import require

RY_PI = np.diag([-1.0, 1.0, -1.0, 1.0])
RX_PI = np.diag([1.0, -1.0, -1.0, 1.0])

#: Joint-name keyword -> half turn that mirrors the right joint.
ARM_FLIP = {"Scapula": RY_PI, "Shoulder": RY_PI, "Forearm": RX_PI, "Wrist": RX_PI}

#: Right coordinates whose sign flips (stored ``q' = -q``); elbow Ry and wrist X
#: were already mirrored.
FLIPPED_COORDINATES = frozenset(
    {"RScapInputX", "RScapInputY", "RSInputX", "RSInputZ", "RFInput", "RWInputY"}
)


def _right_joint(document: dict[str, Any], key: str) -> dict[str, Any]:
    word = f"Right {key}"
    found = [j for j in document["joints"] if word in j["name"]]
    require(len(found) == 1, f"expected one '{word}' joint, found {len(found)}")
    return found[0]


def _left_joint(document: dict[str, Any], key: str) -> dict[str, Any]:
    return next(j for j in document["joints"] if f"Left {key}" in j["name"])


def arm_is_mirrored(document: dict[str, Any]) -> bool:
    """True when the right arm axes already differ from the left (mirrored)."""
    right = np.asarray(_right_joint(document, "Shoulder")["parent_to_base"])
    left = np.asarray(_left_joint(document, "Shoulder")["parent_to_base"])
    return not np.allclose(right[:3, :3], left[:3, :3])


#: The wrist half turn as a rotation block (``Rx(pi)``); an involution.
WRIST_FLIP = RX_PI[:3, :3]


def native_wrist_rotation(document: dict[str, Any], rotation: Any, side: str) -> Any:
    """A wrist ``child_to_follower`` rotation in the native (pre-mirror) frame.

    The right wrist carries ``N @ M @ Rx(pi)`` once the arm is mirrored; the grip
    contract (``N @ M(angles)``) is stated before that half turn, so the right
    rotation is un-flipped here. Left, and unmirrored documents, pass through.
    """
    rot = np.asarray(rotation, dtype=float)
    if side == "R" and arm_is_mirrored(document):
        return rot @ WRIST_FLIP
    return rot


def _flip_limits(document: dict[str, Any]) -> None:
    """Negate the flipped coordinates' ranges (swapping bounds) and seeds."""
    ranges = document.get("coordinate_ranges_deg", {})
    for name in FLIPPED_COORDINATES & ranges.keys():
        low, high = ranges[name]
        ranges[name] = [-high, low * -1.0]
    seeds = document.get("address_seed_deg", {})
    for name in FLIPPED_COORDINATES & seeds.keys():
        seeds[name] = -seeds[name]


def mirror_document_arm(document: dict[str, Any], *, flip_limits: bool = True) -> bool:
    """Mirror the right arm joint frames in place; idempotent.

    Returns ``False`` when already mirrored. Postcondition: the zero pose and
    every body frame are unchanged and each right coordinate in
    :data:`FLIPPED_COORDINATES` now carries the opposite sign. ``flip_limits``
    also negates those coordinates' ranges and seeds; pass ``False`` when the
    limits were already written in the mirrored convention (the builder tables).
    """
    if arm_is_mirrored(document):
        return False
    for key, flip in ARM_FLIP.items():
        joint = _right_joint(document, key)
        for field in ("parent_to_base", "child_to_follower"):
            joint[field] = (np.asarray(joint[field], dtype=float) @ flip).tolist()
    if flip_limits:
        _flip_limits(document)
    return True


def native_limits(
    table: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """A range table written in the mirrored convention, in the native one."""
    out = dict(table)
    for name in FLIPPED_COORDINATES & table.keys():
        low, high = table[name]
        out[name] = (-high, -low)
    return out
