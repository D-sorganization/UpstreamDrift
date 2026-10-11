"""Bushing-grip topology transform for the full-body spec (issue #11739, OSV-7).

Pure dictionary transform with no OpenSim dependency.  The default ``weld``
spec makes the club a child of the left forearm (2-DOF wrist) and welds the
right hand to it, with both hand masses carried as solids of the club body.
For the ``bushing`` grip model the club becomes a free body and each hand is a
separate body joined to the club by its own six-axis bushing at the shared
grip frames of :class:`~src.shared.python.grip_contact.GripInterface`:

* the two hand solids (``LHand``/``LHandStandoff`` and ``RHand``) move from the
  club to the left hand body and ``RHandStandoff`` respectively, with
  placements re-expressed so every solid keeps its world pose, mass and
  inertia; total model mass is unchanged;
* the left hand body sits at the wrist joint frame (the club's old joint
  frame), so the 44 original coordinates keep their meaning;
* six new passive coordinates ``ClubFree*`` carry the club's free pose.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.grip_contact import GripInterface

LEFT_HAND_BODY = "solid_reference:GolfSwing3D_Kinetic/Grip/LHandBody"
CLUB_FREE_COORDINATES = (
    "ClubFreeRX",
    "ClubFreeRY",
    "ClubFreeRZ",
    "ClubFreeTX",
    "ClubFreeTY",
    "ClubFreeTZ",
)
_PRIMITIVES = ("Rx", "Ry", "Rz", "Px", "Py", "Pz")
_LEFT_SOLIDS = (
    "GolfSwing3D_Kinetic/Grip/LHand",
    "GolfSwing3D_Kinetic/Grip/LHandStandoff",
)
_RIGHT_SOLIDS = ("GolfSwing3D_Kinetic/Grip/RHand",)


def _inv(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return out


def _body(spec: Mapping[str, Any], name: str) -> dict[str, Any]:
    for body in spec["bodies"]:
        if body["name"] == name:
            return body
    raise ValueError(f"body {name!r} not found in spec")


def _move_solids(
    club: dict[str, Any],
    target: dict[str, Any],
    names: tuple[str, ...],
    club_to_target: np.ndarray,
) -> None:
    """Move named solids from the club to ``target``, preserving world pose."""
    keep: list[dict[str, Any]] = []
    moved: list[dict[str, Any]] = []
    for solid in club["solids"]:
        (moved if solid["name"] in names else keep).append(solid)
    if {s["name"] for s in moved} != set(names):
        raise ValueError(f"club body lacks hand solids {names}")
    for solid in moved:
        solid["placement"] = (club_to_target @ np.asarray(solid["placement"])).tolist()
    club["solids"] = keep
    target.setdefault("solids", []).extend(moved)


def build_bushing_spec(
    spec: Mapping[str, Any], interface: GripInterface
) -> dict[str, Any]:
    """Return a deep-copied spec in the bushing topology plus ``grip_bushing``.

    ``grip_bushing[side]`` holds ``hand_body``, ``hand_frame`` (4x4 on the hand
    body), ``club_body`` and ``club_frame`` (4x4 on the club); the two frames
    coincide in the weld pose, so the bushing starts unloaded there.

    Raises:
        ValueError: if the spec lacks the closure, the club parent joint or the
            hand solids.
    """
    out = copy.deepcopy(dict(spec))
    closure = out.get("closure")
    if not closure:
        raise ValueError("spec has no closure")
    club_name, right_name = closure["body_b"], closure["body_a"]
    joints = [j for j in out["joints"] if j["child"] == club_name]
    if len(joints) != 1:
        raise ValueError("expected exactly one joint parenting the club")
    left_joint = joints[0]
    t_left = np.asarray(left_joint["child_to_follower"], dtype=float)
    pa = np.asarray(closure["placement_a"], dtype=float)
    pb = np.asarray(closure["placement_b"], dtype=float)
    right_in_club = pb @ _inv(pa)

    club = _body(out, club_name)
    right = _body(out, right_name)
    left = {"name": LEFT_HAND_BODY, "solids": []}
    _move_solids(club, left, _LEFT_SOLIDS, _inv(t_left))
    _move_solids(club, right, _RIGHT_SOLIDS, _inv(right_in_club))
    out["bodies"].append(left)

    left_joint["child"] = LEFT_HAND_BODY
    left_joint["child_to_follower"] = np.eye(4).tolist()
    ident = np.eye(4).tolist()
    out["joints"].append(
        {
            "name": "GolfSwing3D_Kinetic/Club/Free Joint",
            "type": "free",
            "parent": "world",
            "child": club_name,
            "parent_to_base": ident,
            "child_to_follower": ident,
            "primitives": [
                {"coordinate": c, "primitive": p}
                for c, p in zip(CLUB_FREE_COORDINATES, _PRIMITIVES, strict=True)
            ],
        }
    )
    out["coordinate_order"] = [*out["coordinate_order"], *CLUB_FREE_COORDINATES]
    out["passive_coordinates"] = list(CLUB_FREE_COORDINATES)

    g_left, g_right = interface.left.matrix(), interface.right.matrix()
    out["grip_bushing"] = {
        "L": {
            "hand_body": LEFT_HAND_BODY,
            "hand_frame": (_inv(t_left) @ g_left).tolist(),
            "club_body": club_name,
            "club_frame": g_left.tolist(),
        },
        "R": {
            "hand_body": right_name,
            "hand_frame": (_inv(right_in_club) @ g_right).tolist(),
            "club_body": club_name,
            "club_frame": g_right.tolist(),
        },
    }
    return out
