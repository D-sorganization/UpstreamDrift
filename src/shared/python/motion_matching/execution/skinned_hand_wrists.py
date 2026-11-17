"""Move a spec document's wrist joints to the skinned hands' wrists (#15253).

The capsule hands put each wrist joint a fixed anthropometric offset from the
grip axis (``grip_rotation._place_wrist_origins``). The skinned human pair of
:mod:`skinned_club_hands` is posed on the grip with its own wrist joints, a few
centimetres away, so a model forearm stopped short of the hand it should carry.
:func:`apply_skinned_hand_wrists` returns a revision whose wrist joints sit at
the skinned wrists. Only placements move: no mass or inertia changes, and the
point of the trail weld on the club stays where it was.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.model_appearance.skinned_club_hands import wrist_joints_m
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

Array = NDArray[np.float64]

CONFIGURATION = "skinned_hand_wrists"
_BACKEND = "skinned_human"
_CLUB_SUFFIX = "Clubface Vector"
_TRAIL_HAND_SUFFIX = "RHandStandoff"


def _wrist_joint(document: Mapping[str, Any], suffix: str) -> dict[str, Any]:
    for joint in document["joints"]:
        if joint["child"].endswith(suffix):
            return joint  # type: ignore[no-any-return]
    raise ValueError(f"document has no wrist joint whose child ends with {suffix}")


def _matrix(value: Any) -> Array:
    return np.asarray(value, dtype=float).copy()


def _trail_wrist_in_club(document: Mapping[str, Any]) -> Array:
    """4x4 pose of the trail wrist joint in the club frame, through the weld."""
    closure = document["closure"]
    follower = _matrix(_wrist_joint(document, _TRAIL_HAND_SUFFIX)["child_to_follower"])
    return (
        _matrix(closure["placement_b"])
        @ np.linalg.inv(_matrix(closure["placement_a"]))
        @ follower
    )


def wrist_joints_in_club_m(document: Mapping[str, Any]) -> dict[str, Array]:
    """Wrist joint centre of each hand (``lead``, ``trail``) in the club frame."""
    lead = _matrix(_wrist_joint(document, _CLUB_SUFFIX)["child_to_follower"])
    return {"lead": lead[:3, 3], "trail": _trail_wrist_in_club(document)[:3, 3]}


def _move(out: dict[str, Any], target: Mapping[str, Array]) -> None:
    lead_joint = _wrist_joint(out, _CLUB_SUFFIX)
    lead = _matrix(lead_joint["child_to_follower"])
    lead[:3, 3] = target["lead"]
    lead_joint["child_to_follower"] = lead.tolist()
    # The trail hand body is welded to the club: keep the weld point on the
    # club and the hand's turn, and move the weld point on the hand body so
    # the wrist joint lands on the target.
    wrist = _trail_wrist_in_club(out)
    wrist[:3, 3] = target["trail"]
    follower = _matrix(_wrist_joint(out, _TRAIL_HAND_SUFFIX)["child_to_follower"])
    closure = out["closure"]
    closure["placement_a"] = (
        follower @ np.linalg.inv(wrist) @ _matrix(closure["placement_b"])
    ).tolist()
    for frame in out["frames"]:
        if frame["name"] == "LW" and frame["body"].endswith(_CLUB_SUFFIX):
            placement = _matrix(frame["placement"])
            placement[:3, 3] = target["lead"]
            frame["placement"] = placement.tolist()


def apply_skinned_hand_wrists(document: Mapping[str, Any]) -> dict[str, Any]:
    """New spec revision with the wrist joints at the skinned hands' wrists.

    The lead wrist joint and the ``LW`` frame move on the club body; the trail
    wrist moves by shifting ``closure.placement_a`` on the trail hand body.
    ``subject.hand_visual_backend`` becomes ``skinned_human`` and
    ``spec_revision`` records the parent hashes.

    Preconditions: a right-handed document with a club (``ValueError``
    otherwise; the skinned pair is posed right-handed). Postconditions: the
    input is not mutated; bodies, joint rotations and ``placement_b`` are
    unchanged; applying it to its own result returns an equal document.
    """
    if subject_handedness(document) == "left":
        raise ValueError(
            "left-handed skinned hand wrists are not implemented: the skinned "
            "pair is posed for a right-handed golfer"
        )
    club = assembly_from_spec(document)
    if club is None:
        raise ValueError("document has no club to hold")
    target = wrist_joints_m(club)
    now = wrist_joints_in_club_m(document)
    out = copy.deepcopy(dict(document))
    placed = all(np.allclose(now[side], target[side], atol=1e-12) for side in target)
    if placed and out["subject"].get("hand_visual_backend") == _BACKEND:
        return out
    _move(out, target)
    out["subject"]["hand_visual_backend"] = _BACKEND
    version = int(document.get("spec_revision", {}).get("version", 0)) + 1
    out["spec_revision"] = {
        "version": version,
        "configuration": CONFIGURATION,
        "parent_spec_canonical_sha256": canonical_sha256(document),
        "parent_physics_spec_sha256": physics_spec_sha256(document),
    }
    out["provenance"] = (
        out.get("provenance", "")
        + f"; skinned hand wrist joints revision {version} (#15253)"
    )
    return out
