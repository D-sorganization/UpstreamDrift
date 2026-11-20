"""The posable hand of the skinned backend (moved from ``skinned_human``, #15329).

A sized right hand whose joints the shared posing module can curl: hinge
angles, rotations fixed on top of them, sideways swings at the knuckles and
the thumb's press. Appearance only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.knuckle_sideways import (
    held_swing_rad,
)
from src.shared.python.model_appearance.hand_backends.pose_kinematics import (
    PoseKinematics,
)
from src.shared.python.model_appearance.hand_backends.posing import (
    FlatHandLandmarks,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    SkinnedHumanSettings,
)
from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    rotation_about_axis,
)

Array = NDArray[np.float64]

_SEGMENTS = ("prox", "mid", "dist")
_EYE3 = np.eye(3)
#: Virtual hinge at the thumb base that presses the thumb onto the grip.
THUMB_PRESS = "thumb_press"


class _GripHand:
    """A sized right hand that the shared posing module can curl.

    Hinge angles are anatomical flexion (0 = straight digit). ``fixed`` holds
    rotations applied on top of the hinges (finger adduction, thumb aim).
    """

    def __init__(self, asset: SkinnedHandAsset, settings: SkinnedHumanSettings) -> None:
        self.asset = asset
        self.settings = settings
        self.placement = np.eye(4)
        self.press_axis = np.array([1.0, 0.0, 0.0])
        self._index = {name: i for i, name in enumerate(asset.bone_names)}
        self._pose = PoseKinematics(asset)
        self._kept: list[tuple] = [(None,)] * len(self._index)
        self._segment: dict[str, NDArray[np.int64]] = {}
        self.fixed: dict[str, Array] = self._adduction()
        # Rigid moves applied to a bone before its own rotation (palm cup).
        self.offsets: dict[str, Array] = {}
        # Sideways swing of a finger at its knuckle, radians (side closing).
        self.side: dict[str, float] = {}
        self._side_axis: dict[str, Array] = {}
        self._moved: dict[str, NDArray[np.int64]] = {}

    def _adduction(self) -> dict[str, Array]:
        """Bring the rest finger splay toward the middle finger's direction."""
        fraction = self.settings.finger_adduction_fraction
        reference = self.direction("middle_prox")
        palmar = np.array([0.0, 0.0, 1.0])
        out: dict[str, Array] = {}
        for finger in FINGERS:
            d = self.direction(f"{finger}_prox")
            splay = math.atan2(d[0], d[1]) - math.atan2(reference[0], reference[1])
            out[f"{finger}_prox"] = rotation_about_axis(palmar, fraction * splay)
        return out

    def direction(self, bone: str) -> Array:
        i = self.asset.bone_index(bone)
        d = self.asset.bone_tails[i] - self.asset.bone_heads[i]
        return d / np.linalg.norm(d)

    def swing_sideways(self, bone: str | None, swing: float = 0.0) -> None:
        """Swing finger ``bone`` sideways at its knuckle by ``swing`` radians,
        about the axis across its flexion axis; a bent knuckle takes less of
        it (:mod:`knuckle_sideways`). ``None`` clears every swing."""
        if bone is None:
            self.side.clear()
            return
        axis = np.cross(self.asset.flex_axes[self._index[bone]], self.direction(bone))
        self.side[bone] = swing
        self._side_axis[bone] = axis / np.linalg.norm(axis)

    def side_swings(self, angles: Mapping[str, float]) -> dict[str, float]:
        """Each swung knuckle's sideways swing as its flexion lets it be."""
        rest, at = self.asset.rest_flexion_rad, self._index
        return {
            bone: held_swing_rad(swing, angles.get(bone, rest[at[bone]]), self.settings)
            for bone, swing in self.side.items()
        }

    def _rotation(self, name: str, angles: Mapping[str, float]) -> Array | None:
        """A bone's rotation about its head: what is fixed, then its hinge,
        its sideways swing and the thumb press; ``None`` if it has none."""
        i = self._index[name]
        rest = float(self.asset.rest_flexion_rad[i])
        out = self.fixed.get(name)
        if name in angles and angles[name] - rest != 0.0:
            hinge = self._pose.hinge(i, angles[name] - rest)
            out = hinge if out is None else out @ hinge
        if name in self.side:
            lean = held_swing_rad(
                self.side[name], angles.get(name, rest), self.settings
            )
            swing = rotation_about_axis(self._side_axis[name], lean)
            out = swing if out is None else out @ swing
        if name == "thumb_meta" and THUMB_PRESS in angles:
            press = rotation_about_axis(self.press_axis, angles[THUMB_PRESS])
            out = press @ (_EYE3 if out is None else out)
        return out

    def rotations(self, angles: Mapping[str, float]) -> dict[str, Array]:
        every = ((name, self._rotation(name, angles)) for name in self._index)
        return {name: rotation for name, rotation in every if rotation is not None}

    def _world(self, angles: Mapping[str, float]) -> Array:
        """Forward kinematics, in an array that the next pose overwrites. A
        bone's rotation is made again only when an angle or a matrix it is
        made of has changed (matrices are replaced, never changed in place)."""
        get, fixed, side = angles.get, self.fixed.get, self.side.get
        axes, kept = self._side_axis.get, self._kept
        press = (get(THUMB_PRESS), id(self.press_axis))
        rotations = []
        for i, name in enumerate(self._index):
            parts = (fixed(name), axes(name), self.press_axis)
            thumb = press if name == "thumb_meta" else None
            key = (get(name), side(name), thumb, id(parts[0]), id(parts[1]))
            if kept[i][0] != key:  # ``parts`` are kept so that no id is reused
                kept[i] = (key, self._rotation(name, angles), parts)
            rotations.append(kept[i][1])
        return self._pose.world(rotations, [self.offsets.get(n) for n in self._index])

    def transforms(self, angles: Mapping[str, float]) -> Array:
        """Forward kinematics; equal to ``skinning.bone_transforms`` of
        :meth:`rotations` (a unit test holds the two together)."""
        return self._world(angles).copy()

    def points(
        self, angles: Mapping[str, float], ids: NDArray[np.int64] | None = None
    ) -> Array:
        """Skinned vertices in the grip frame (same as ``pose_vertices``);
        only vertices ``ids`` when given."""
        local = self._pose.skinned(self._world(angles), ids)
        return local @ self.placement[:3, :3].T + self.placement[:3, 3]

    def segment_ids(self, bone: str) -> NDArray[np.int64]:
        if bone not in self._segment:
            names = ("thumb_prox", "thumb_dist") if bone == THUMB_PRESS else (bone,)
            self._segment[bone] = self.asset.dominant_vertex_ids(names)
        return self._segment[bone]

    def moved_ids(self, bone: str) -> NDArray[np.int64]:
        if bone not in self._moved:
            root = "thumb_meta" if bone == THUMB_PRESS else bone
            names = [self.asset.bone_names[i] for i in self.asset.descendants(root)]
            self._moved[bone] = self.asset.influenced_vertex_ids(names)
        return self._moved[bone]

    def joint_position(
        self, bone: str, angles: Mapping[str, float], *, tail: bool = False
    ) -> Array:
        """Posed head (or tail) of ``bone`` in the grip frame."""
        i = self.asset.bone_index(bone)
        rest = self.asset.bone_tails[i] if tail else self.asset.bone_heads[i]
        t = self._world(angles)[i]
        local = t[:3, :3] @ rest + t[:3, 3]
        return self.placement[:3, :3] @ local + self.placement[:3, 3]


def _straight_angles() -> dict[str, float]:
    angles = {f"{f}_{s}": 0.0 for f in FINGERS for s in _SEGMENTS}
    angles.update({f"meta_{f}": 0.0 for f in FINGERS})
    return angles


def flat_hand_landmarks(hand: _GripHand) -> FlatHandLandmarks:
    """Landmarks of the flat (straight, adducted) hand in its rest frame."""
    t = hand.transforms(_straight_angles())
    asset = hand.asset

    def head(bone: str) -> Array:
        i = asset.bone_index(bone)
        return t[i][:3, :3] @ asset.bone_heads[i] + t[i][:3, 3]

    return FlatHandLandmarks(
        index_mcp=head("index_prox"),
        index_pip=head("index_mid"),
        little_mcp=head("little_prox"),
        radial=np.array([1.0, 0.0, 0.0]),
        distal=np.array([0.0, 1.0, 0.0]),
        palmar=np.array([0.0, 0.0, 1.0]),
    )
