"""Put a spec document's wrist joints at the skinned hands' wrists (#15253, #15297).

The capsule hands put each wrist joint a fixed anthropometric offset from the
grip axis (``grip_rotation._place_wrist_origins``). The skinned human pair of
:mod:`skinned_club_hands` is posed on the grip with its own wrist joints, a few
centimetres away, so a model forearm stopped short of the hand it should carry.
:func:`apply_skinned_hand_wrists` returns a revision whose wrist joints sit at
the skinned wrists and are turned with them: at the wrist angles the skinned
hands are posed with (:func:`skinned_wrist_coordinates_rad`) each model forearm
has the radial, distal and palmar axes of its hand's forearm. Moved alone, the
joints kept the capsule grip's hand turn and a forearm met its hand 44 to 130
degrees off. Only placements change: no mass or inertia does, and the point of
the trail weld on the club stays where it was.

Wrist convention (``coordinate_sign_map``, rows ``*WInputX``): the first wrist
primitive is deviation, radial positive; the second is flexion/extension, its
sign found here from the hand's side, as it differs between spec families.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.model_appearance.skinned_club_hands import (
    forearm_axes,
    wrist_joints_m,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

Array = NDArray[np.float64]

CONFIGURATION = "skinned_hand_wrists"
_BACKEND = "skinned_human"
_CLUB_SUFFIX = "Clubface Vector"
_TRAIL_HAND_SUFFIX = "RHandStandoff"
#: Hand body of each skinned hand, and whether its triad is right-handed (+1).
_HANDS = {"lead": (_CLUB_SUFFIX, -1.0), "trail": (_TRAIL_HAND_SUFFIX, 1.0)}
_AXES = {"Rx": (1.0, 0.0, 0.0), "Ry": (0.0, 1.0, 0.0), "Rz": (0.0, 0.0, 1.0)}


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
    return {side: pose[:3, 3] for side, pose in _now_poses(document).items()}


def _turn(axis: Array, angle: float) -> Array:
    """Rotation by ``angle`` (rad) about the unit ``axis`` (Rodrigues)."""
    k = np.cross(np.eye(3), axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _forearm_in_base(joint: Mapping[str, Any], handed: float) -> tuple[Array, ...]:
    """Model forearm triad (radial, distal, palmar) and the two wrist axes, base frame.

    Distal runs from the forearm body's origin to the wrist joint; a positive
    first primitive carries it toward radial; ``handed`` fixes palmar.
    """
    base = _matrix(joint["parent_to_base"])
    primitives = joint["primitives"]
    if len(primitives) != 2 or np.linalg.norm(base[:3, 3]) == 0.0:
        raise ValueError("a wrist joint needs two primitives away from its parent")
    deviation, extension = (np.array(_AXES[p["primitive"]]) for p in primitives)
    distal = base[:3, :3].T @ base[:3, 3] / np.linalg.norm(base[:3, 3])
    radial = np.cross(deviation, distal)
    palmar = handed * np.cross(radial, distal)
    return np.column_stack([radial, distal, palmar]), deviation, extension


def _posed_wrist(joint: Mapping[str, Any], hand: str) -> tuple[Array, Array]:
    """Wrist coordinates (rad) of the skinned hand's pose, and the joint's turn there."""
    extension_deg, ulnar_deg = dict(default_settings().wrist_deg)[hand]
    triad, deviation, extension = _forearm_in_base(joint, _HANDS[hand][1])
    # A positive second primitive carries distal toward ``extension x distal``:
    # that is extension when it points away from the palm.
    sign = -np.sign(np.cross(extension, triad[:, 1]) @ triad[:, 2])
    angles = np.radians([-ulnar_deg, float(sign) * extension_deg])
    return angles, _turn(deviation, angles[0]) @ _turn(extension, angles[1])


def skinned_wrist_coordinates_rad(document: Mapping[str, Any]) -> dict[str, float]:
    """Model wrist coordinates (rad) at which the skinned hands are posed.

    From the hand settings' wrist extension and ulnar deviation. On a revised
    document each model forearm then runs along its skinned hand's forearm.
    """
    at: dict[str, float] = {}
    for hand, (suffix, _) in _HANDS.items():
        joint = _wrist_joint(document, suffix)
        names = [p["coordinate"] for p in joint["primitives"]]
        at.update(zip(names, _posed_wrist(joint, hand)[0].tolist(), strict=True))
    return at


def _target_poses(document: Mapping[str, Any], club: Any) -> dict[str, Array]:
    """4x4 pose each wrist joint's follower frame should have in the club frame."""
    centres, axes = wrist_joints_m(club), forearm_axes(club)
    poses = {}
    for hand, (suffix, handed) in _HANDS.items():
        joint = _wrist_joint(document, suffix)
        triad = _forearm_in_base(joint, handed)[0]
        pose = np.eye(4)
        # Follower = base turned by the joint: the forearm triad seen from the
        # follower must be the skinned forearm's triad seen from the club.
        pose[:3, :3] = axes[hand] @ (_posed_wrist(joint, hand)[1].T @ triad).T
        pose[:3, 3] = centres[hand]
        if np.linalg.det(pose[:3, :3]) < 0.0:
            raise ValueError(f"the {hand} hand is not on the side the document has")
        poses[hand] = pose
    return poses


def _now_poses(document: Mapping[str, Any]) -> dict[str, Array]:
    lead = _matrix(_wrist_joint(document, _CLUB_SUFFIX)["child_to_follower"])
    return {"lead": lead, "trail": _trail_wrist_in_club(document)}


def _place(out: dict[str, Any], target: Mapping[str, Array]) -> None:
    _wrist_joint(out, _CLUB_SUFFIX)["child_to_follower"] = target["lead"].tolist()
    # The trail hand body is welded to the club: keep the weld point on the
    # club, and move and turn the weld frame on the hand body so the wrist
    # joint lands on the target.
    follower = _matrix(_wrist_joint(out, _TRAIL_HAND_SUFFIX)["child_to_follower"])
    closure = out["closure"]
    closure["placement_a"] = (
        follower @ np.linalg.inv(target["trail"]) @ _matrix(closure["placement_b"])
    ).tolist()
    for frame in out["frames"]:
        if frame["name"] == "LW" and frame["body"].endswith(_CLUB_SUFFIX):
            placement = _matrix(frame["placement"])
            placement[:3, 3] = target["lead"][:3, 3]
            frame["placement"] = placement.tolist()


def apply_skinned_hand_wrists(document: Mapping[str, Any]) -> dict[str, Any]:
    """New spec revision with the wrist joints at, and turned as, the skinned wrists.

    The lead wrist joint moves and turns on the club body and the ``LW`` frame
    moves with it (its turn is kept); the trail wrist moves and turns through
    ``closure.placement_a`` on the trail hand body.
    ``subject.hand_visual_backend`` becomes ``skinned_human`` and
    ``spec_revision`` records the parent hashes.

    Preconditions: a right-handed document with a club (``ValueError``
    otherwise; the skinned pair is posed right-handed). Postconditions: the
    input is not mutated; bodies, the trail joint and ``placement_b`` are
    unchanged; at :func:`skinned_wrist_coordinates_rad` each forearm runs along
    its hand's; applying it to its own result returns an equal document.
    """
    if subject_handedness(document) == "left":
        raise ValueError(
            "left-handed skinned hand wrists are not implemented: the skinned "
            "pair is posed for a right-handed golfer"
        )
    club = assembly_from_spec(document)
    if club is None:
        raise ValueError("document has no club to hold")
    target, now = _target_poses(document, club), _now_poses(document)
    out = copy.deepcopy(dict(document))
    placed = all(np.allclose(now[side], target[side], atol=1e-9) for side in target)
    if placed and out["subject"].get("hand_visual_backend") == _BACKEND:
        return out
    _place(out, target)
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
        + f"; skinned hand wrist joints revision {version} (#15253, #15297)"
    )
    return out
