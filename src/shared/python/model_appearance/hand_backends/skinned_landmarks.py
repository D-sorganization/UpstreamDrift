"""Typed landmarks and joint frames of a posed skinned hand (#15272).

The other hand tracks of epic #15150 pose and judge their hands against the
same joints. This module reads them off a :class:`HandVisualResult` so no
track has to know the rig: every joint's centre, hinge axis and bone, and the
named points a grip is described by (wrist, knuckles, tips). Everything is in
the result's own frame, the hand-grip frame, in metres.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.joint_angles import (
    FINGERS,
    HandSide,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

Array = NDArray[np.float64]

#: Digits with a knuckle and a tip, and the segments those are read from.
DIGITS: tuple[str, ...] = (*FINGERS, "thumb")
KNUCKLE_SEGMENT = "prox"
TIP_SEGMENT = "dist"
WRIST_BONE = "wrist"


@dataclass(frozen=True)
class HandJointFrame:
    """One joint of the posed hand and the bone that turns on it.

    ``origin_m`` is the joint's centre. ``hinge_axis`` is the unit axis it
    flexes about: a positive joint angle is a right-handed turn about it, on
    either hand; ``None`` for a bone the rig gives no hinge (the forearm).
    ``bone_axis`` is the unit direction from the joint to the bone's far end,
    ``length_m`` away.
    """

    name: str
    parent: str | None
    origin_m: Array
    hinge_axis: Array | None
    bone_axis: Array
    length_m: float

    @property
    def end_m(self) -> Array:
        """The bone's far end: the next joint's centre, or the digit's tip."""
        return np.asarray(self.origin_m + self.length_m * self.bone_axis)


@dataclass(frozen=True)
class HandLandmarks:
    """Joints (in the rig's order, parents first) and named points of a hand."""

    backend: str
    side: HandSide
    joints: tuple[HandJointFrame, ...]
    points_m: Mapping[str, Array]

    def joint(self, name: str) -> HandJointFrame:
        """The joint called ``name``; ``KeyError`` lists the known names."""
        for joint in self.joints:
            if joint.name == name:
                return joint
        known = [joint.name for joint in self.joints]
        raise KeyError(f"no joint {name!r}; known: {known}")

    def as_mapping(self) -> dict[str, Any]:
        """Plain lists and strings, ready for JSON."""
        return {
            "backend": self.backend,
            "side": self.side,
            "frame": "hand_grip",
            "units": "m",
            "joints": {
                joint.name: {
                    "parent": joint.parent,
                    "origin": joint.origin_m.tolist(),
                    "hinge_axis": _listed(joint.hinge_axis),
                    "bone_axis": joint.bone_axis.tolist(),
                    "length": joint.length_m,
                }
                for joint in self.joints
            },
            "points": {name: point.tolist() for name, point in self.points_m.items()},
        }


def _unit(vector: Array) -> Array:
    return np.asarray(vector / np.linalg.norm(vector))


def _listed(axis: Array | None) -> list[float] | None:
    return None if axis is None else axis.tolist()


def hand_landmarks(result: HandVisualResult) -> HandLandmarks:
    """Joint frames and named points of ``result``, in its own frame.

    Preconditions: ``result`` is a :class:`HandVisualResult` that carries its
    rest asset, bone transforms and rigid frame (the skinned backends do).
    Postconditions: one joint per bone of the rig, in the rig's order; unit
    axes; the hinge axis is turned with the bone and, on a mirrored hand,
    reversed, so positive flexion is right-handed about it on both hands (a
    bone whose rest hinge axis is zero has none);
    ``points_m`` holds the wrist and each digit's knuckle and tip.
    """
    if not isinstance(result, HandVisualResult):
        raise TypeError("hand_landmarks needs a HandVisualResult")
    asset, transforms, frame = (
        result.rest_asset,
        result.bone_transforms,
        result.frame_from_rest,
    )
    if asset is None or transforms is None or frame is None:
        raise ValueError(f"{result.backend!r} hand carries no skinning data")
    # A mirror image turns the other way about the same line.
    handed = 1.0 if float(np.linalg.det(frame[:3, :3])) > 0.0 else -1.0
    joints = []
    for i, name in enumerate(asset.bone_names):
        placed = frame @ transforms[i]
        turn = placed[:3, :3]
        head = turn @ asset.bone_heads[i] + placed[:3, 3]
        bone = turn @ (asset.bone_tails[i] - asset.bone_heads[i])
        parent = int(asset.bone_parents[i])
        hinge = turn @ asset.flex_axes[i]
        joints.append(
            HandJointFrame(
                name=name,
                parent=None if parent < 0 else asset.bone_names[parent],
                origin_m=np.asarray(head),
                hinge_axis=_unit(handed * hinge) if hinge.any() else None,
                bone_axis=_unit(bone),
                length_m=float(np.linalg.norm(bone)),
            )
        )
    marks = HandLandmarks(result.backend, result.side, tuple(joints), {})
    points = {WRIST_BONE: marks.joint(WRIST_BONE).origin_m}
    for digit in DIGITS:
        points[f"{digit}_knuckle"] = marks.joint(f"{digit}_{KNUCKLE_SEGMENT}").origin_m
        points[f"{digit}_tip"] = marks.joint(f"{digit}_{TIP_SEGMENT}").end_m
    return HandLandmarks(result.backend, result.side, tuple(joints), points)


__all__ = [
    "DIGITS",
    "HandJointFrame",
    "HandLandmarks",
    "hand_landmarks",
]
