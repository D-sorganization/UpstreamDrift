"""One resolved hand specification for every engine emitter (#13995).

Each emitter used to call ``assembly_meshes_with_hands(club)`` with defaults, so
the grip style, grip strength and stature-tied hand length of a spec never
reached a render. :func:`resolved_hand_spec` reads them once from the spec
document; emitters pass the resulting :class:`ResolvedHandSpec` on to
:mod:`grip_hands` and infer no anatomy themselves. Visual only: no mass, inertia
or constraint depends on it.

Hand length: ``subject.hand_length_m`` when set, else the ANSUR II ratio times
``subject.stature_m`` (:func:`hand_size.resolve_hand_length_m`), else the
reference hand. Grip: ``subject.grip_config`` (style, strength, handedness) via
:func:`grip_spec_of`, else the neutral overlap pose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.grip_contact.grip_geometry import GripGeometry, get_grip_preset
from src.shared.python.hand_size import (
    DEFAULT_HAND_LENGTH_M,
    resolve_hand_length_m,
    validate_hand_length_m,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

_GRIP_AXIS = np.array([0.0, 1.0, 0.0], dtype=float)


def wrist_follower_dorsal_axis(
    spec_document: Mapping[str, Any],
    hand: Hand,
    pose: GripPose = DEFAULT_GRIP_POSE,
) -> NDArray[np.float64] | None:
    """Wrist follower dorsal-palmar unit axis in the club frame, projected off the shaft (#13996)."""
    if not isinstance(hand, Hand):
        raise TypeError(f"hand must be a Hand, got {hand!r}")
    joints = spec_document.get("joints")
    if not isinstance(joints, list | tuple) or not joints:
        return None
    side = pose.side(hand)
    if side == "L":
        matches = [
            j for j in joints if str(j.get("child", "")).endswith("Clubface Vector")
        ]
        if not matches:
            return None
        f_rot = np.asarray(matches[0]["child_to_follower"], dtype=float)[:3, :3]
    else:
        closure = spec_document.get("closure")
        matches = [
            j for j in joints if str(j.get("child", "")).endswith("RHandStandoff")
        ]
        if not isinstance(closure, Mapping) or not matches:
            return None
        pa = np.asarray(closure["placement_a"], dtype=float)
        pb = np.asarray(closure["placement_b"], dtype=float)
        c2f_r = np.asarray(matches[0]["child_to_follower"], dtype=float)
        f_rot = (pb @ np.linalg.inv(pa) @ c2f_r)[:3, :3]
    dorsal = f_rot[:, 0] - float(f_rot[:, 0] @ _GRIP_AXIS) * _GRIP_AXIS
    norm = float(np.linalg.norm(dorsal))
    if norm < 1e-9:
        return None
    return dorsal / norm


@dataclass(frozen=True)
class ResolvedHandSpec:
    """Grip pose (style, strength, handedness), hand length and grip geometry.

    Preconditions: ``pose`` a :class:`GripPose`; ``hand_length_m`` inside
    ``hand_size.HAND_LENGTH_RANGE_M`` (``ValueError`` otherwise).
    ``grip_geometry`` is the canonical radius source (``radius_at``).
    """

    pose: GripPose = DEFAULT_GRIP_POSE
    hand_length_m: float = DEFAULT_HAND_LENGTH_M
    grip_geometry: GripGeometry = field(default_factory=get_grip_preset)
    wrist_dorsal_axes: Mapping[Hand, tuple[float, float, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.pose, GripPose):
            raise TypeError(f"pose must be a GripPose, got {type(self.pose).__name__}")
        object.__setattr__(
            self, "hand_length_m", validate_hand_length_m(self.hand_length_m)
        )

    def grip_radius_m(self, distance_below_butt_m: float) -> float:
        """Canonical grip radius at ``distance_below_butt_m`` (``radius_at``)."""
        return self.grip_geometry.radius_at(distance_below_butt_m)


def resolved_hand_spec(spec_document: Mapping[str, Any]) -> ResolvedHandSpec:
    """The :class:`ResolvedHandSpec` a spec document describes.

    ``ValueError`` for an unknown grip style or a hand length outside the
    ``hand_size`` range.
    """
    # Imported here: grip_rotation pulls the motion-matching package.
    from src.shared.python.motion_matching.execution.grip_rotation import (
        grip_spec_of,
    )

    subject = spec_document.get("subject", {})
    pose = (
        grip_spec_of(spec_document).pose()
        if subject.get("grip_config") is not None
        else DEFAULT_GRIP_POSE
    )
    override = subject.get("hand_length_m")
    stature = subject.get("stature_m")
    if override is None and stature is None:
        length = DEFAULT_HAND_LENGTH_M
    else:
        length = resolve_hand_length_m(
            stature, str(subject.get("sex", "unspecified")), override_m=override
        )
    dorsal_axes: dict[Hand, tuple[float, float, float]] = {}
    for hand in (Hand.LEAD, Hand.TRAIL):
        axis = wrist_follower_dorsal_axis(spec_document, hand, pose)
        if axis is not None:
            dorsal_axes[hand] = (float(axis[0]), float(axis[1]), float(axis[2]))
    return ResolvedHandSpec(
        pose=pose, hand_length_m=length, wrist_dorsal_axes=dorsal_axes
    )
