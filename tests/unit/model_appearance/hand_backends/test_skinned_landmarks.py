"""Typed landmarks and joint frames of the skinned hand (#15272, epic #15150)."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends import skinned_human
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    FINGERS,
    HandJointAngles,
)
from src.shared.python.model_appearance.hand_backends.qa import wrist_joint_m
from src.shared.python.model_appearance.hand_backends.skinned_landmarks import (
    HandLandmarks,
    hand_landmarks,
)
from src.shared.python.model_appearance.hand_backends.skinning import (
    rotation_about_axis,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

pytestmark = pytest.mark.unit

_RADIUS = 0.0105
_STEP = math.radians(8.0)
_CACHE: dict[tuple[str, str], HandVisualResult] = {}


def _angles(flexed: str = "") -> dict[str, float]:
    angles = skinned_human._straight_angles()
    for name in angles:
        if not name.startswith("meta_"):
            angles[name] = 0.5
    if flexed:
        angles[flexed] += _STEP
    return angles


def _hand(side: str, flexed: str = "") -> HandVisualResult:
    key = (side, flexed)
    if key not in _CACHE:
        _CACHE[key] = skinned_human.build_hand_visual(
            GripSpec.legacy(),
            side,  # type: ignore[arg-type]
            HandJointAngles.from_mapping(_angles(flexed)),
            grip_radius_m=_RADIUS,
        )
    return _CACHE[key]


@pytest.mark.parametrize("side", ["lead", "trail"])
def test_every_joint_is_exported_with_its_parent_and_unit_axes(side: str) -> None:
    result = _hand(side)
    marks = hand_landmarks(result)
    assert isinstance(marks, HandLandmarks)
    assert (marks.backend, marks.side) == (result.backend, side)
    asset = result.rest_asset
    assert asset is not None
    assert tuple(joint.name for joint in marks.joints) == asset.bone_names
    for i, joint in enumerate(marks.joints):
        parent = int(asset.bone_parents[i])
        assert joint.parent == (None if parent < 0 else asset.bone_names[parent])
        if asset.flex_axes[i].any():
            assert joint.hinge_axis is not None
            assert np.linalg.norm(joint.hinge_axis) == pytest.approx(1.0)
        else:
            assert joint.hinge_axis is None
        assert np.linalg.norm(joint.bone_axis) == pytest.approx(1.0)
        rest_length = np.linalg.norm(asset.bone_tails[i] - asset.bone_heads[i])
        assert joint.length_m == pytest.approx(rest_length)
        np.testing.assert_allclose(
            joint.end_m, joint.origin_m + joint.length_m * joint.bone_axis
        )
    # A chain: each segment starts where the one before it ends.
    np.testing.assert_allclose(
        marks.joint("index_mid").origin_m, marks.joint("index_prox").end_m, atol=1e-9
    )
    hinged = {joint.name for joint in marks.joints if joint.hinge_axis is not None}
    assert {f"{f}_{s}" for f in FINGERS for s in ("prox", "mid", "dist")} <= hinged
    with pytest.raises(KeyError, match="index_prox"):
        marks.joint("elbow")


@pytest.mark.parametrize("side", ["lead", "trail"])
def test_named_points_are_the_wrist_the_knuckles_and_the_tips(side: str) -> None:
    result = _hand(side)
    marks = hand_landmarks(result)
    np.testing.assert_allclose(marks.points_m["wrist"], wrist_joint_m(result))
    for finger in (*FINGERS, "thumb"):
        knuckle, tip = (
            marks.points_m[f"{finger}_knuckle"],
            marks.points_m[f"{finger}_tip"],
        )
        np.testing.assert_allclose(knuckle, marks.joint(f"{finger}_prox").origin_m)
        np.testing.assert_allclose(tip, marks.joint(f"{finger}_dist").end_m)
        # The tip's bone ends inside the fingertip's skin.
        skin = np.linalg.norm(result.mesh.vertices - tip, axis=1).min()
        assert skin < 0.012, f"{finger} tip is {skin * 1e3:.1f} mm from any skin"
    assert set(marks.points_m) == {"wrist"} | {
        f"{finger}_{point}"
        for finger in (*FINGERS, "thumb")
        for point in ("knuckle", "tip")
    }


@pytest.mark.parametrize("side", ["lead", "trail"])
@pytest.mark.parametrize("joint", ["index_prox", "ring_mid", "little_dist"])
def test_the_hinge_axis_says_where_flexing_takes_the_bone(
    side: str, joint: str
) -> None:
    """Positive flexion is a right-handed turn about the exported axis, on both
    hands: the lead hand is a mirror image, so its axis is not the rest axis
    carried by the hand's frame."""
    before = hand_landmarks(_hand(side)).joint(joint)
    after = hand_landmarks(_hand(side, joint)).joint(joint)
    np.testing.assert_allclose(after.origin_m, before.origin_m, atol=1e-9)
    assert before.hinge_axis is not None
    turn = rotation_about_axis(before.hinge_axis, _STEP)
    predicted = before.origin_m + turn @ (before.end_m - before.origin_m)
    assert np.linalg.norm(after.end_m - predicted) < 1e-6
    assert np.linalg.norm(after.end_m - before.end_m) > 1e-3
    np.testing.assert_allclose(after.hinge_axis, before.hinge_axis, atol=1e-9)  # type: ignore[arg-type]


def test_a_hand_without_skinning_data_is_refused() -> None:
    bare = dataclasses.replace(_hand("trail"), rest_asset=None)
    with pytest.raises(ValueError, match="no skinning data"):
        hand_landmarks(bare)
    with pytest.raises(TypeError, match="HandVisualResult"):
        hand_landmarks("trail")  # type: ignore[arg-type]


def test_the_export_is_plain_data_in_metres() -> None:
    marks = hand_landmarks(_hand("lead"))
    plain = json.loads(json.dumps(marks.as_mapping()))
    assert plain["units"] == "m" and plain["frame"] == "hand_grip"
    assert (plain["backend"], plain["side"]) == (marks.backend, "lead")
    joint = plain["joints"]["index_mid"]
    assert joint["parent"] == "index_prox"
    np.testing.assert_allclose(joint["origin"], marks.joint("index_mid").origin_m)
    np.testing.assert_allclose(
        joint["hinge_axis"],
        marks.joint("index_mid").hinge_axis,  # type: ignore[arg-type]
    )
    assert plain["joints"]["forearm"]["hinge_axis"] is None
    np.testing.assert_allclose(
        plain["points"]["thumb_tip"], marks.points_m["thumb_tip"]
    )
