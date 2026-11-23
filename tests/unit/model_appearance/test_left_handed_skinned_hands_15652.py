"""A left-handed golfer's club and skinned hands mirror the right-handed ones (#15652).

The mirror plane is the swing plane: it holds the shaft axis and the top of
the grip, so in the skinned hand frame (``x`` top of grip, ``z`` grip axis) a
left-handed point is the right-handed one with ``y`` negated.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.club_assembly import (
    ADDRESS_SQUARE_FACE_ROLL_DEG,
    ClubAssembly,
    assembly_from_spec,
    assembly_meshes,
    clubface_centre,
    clubface_vector,
)

pytestmark = pytest.mark.unit

#: Exact mirror up to round-off of the frame transforms (metres).
_MIRROR_ATOL_M = 1e-9
_FLIP_Y = np.array([1.0, -1.0, 1.0])


def _club(handedness: str = "right") -> ClubAssembly:
    roll = ADDRESS_SQUARE_FACE_ROLL_DEG["driver"]
    club = ClubAssembly("driver", 1.1176, 0.265, 0.0065, face_roll_deg=roll)
    if handedness == "right":
        return club
    return dataclasses.replace(club, handedness=handedness)


def _in_hand_frame(points: np.ndarray) -> np.ndarray:
    """Club-body points in the right-handed club's skinned hand frame."""
    frame = sch.skinned_hand_frame(_club())
    return (np.asarray(points) - frame[:3, 3]) @ frame[:3, :3]


def _assert_mirrored(left: np.ndarray, right: np.ndarray) -> None:
    np.testing.assert_allclose(
        _in_hand_frame(left), _in_hand_frame(right) * _FLIP_Y, atol=_MIRROR_ATOL_M
    )


def test_left_pair_is_the_mirror_of_the_right_pair_in_its_grip_frame() -> None:
    right, left = sch._pair(handedness="right"), sch._pair(handedness="left")
    for side in ("lead", "trail"):
        r, lt = right[side].mesh, left[side].mesh
        np.testing.assert_allclose(lt.vertices, r.vertices * _FLIP_Y, atol=1e-9)
        np.testing.assert_array_equal(lt.faces, r.faces[:, ::-1])


@pytest.mark.parametrize("cut_at_wrist", (False, True))
def test_left_hand_meshes_mirror_the_right_ones_on_the_club(
    cut_at_wrist: bool,
) -> None:
    right = sch.skinned_hand_meshes(_club(), cut_at_wrist=cut_at_wrist)
    left = sch.skinned_hand_meshes(
        _club("left"), handedness="left", cut_at_wrist=cut_at_wrist
    )
    assert set(left) == set(right) == {"lead_hand", "trail_hand"}
    for label, mesh in left.items():
        _assert_mirrored(mesh.vertices, right[label].vertices)
        assert mesh.volume() > 0.0
        assert mesh.volume() == pytest.approx(right[label].volume(), rel=1e-9)
    if not cut_at_wrist:
        for label, mesh in left.items():
            np.testing.assert_array_equal(mesh.faces, right[label].faces[:, ::-1])


def test_left_wrist_joints_mirror_the_right_ones() -> None:
    right = sch.wrist_joints_m(_club())
    left = sch.wrist_joints_m(_club("left"), handedness="left")
    for side in ("lead", "trail"):
        _assert_mirrored(left[side][None], right[side][None])


def test_left_forearm_reach_mirrors_the_right_one() -> None:
    right = sch.forearm_reach_m(_club())
    left = sch.forearm_reach_m(_club("left"), handedness="left")
    frame = sch.skinned_hand_frame(_club())[:3, :3]
    for side in ("lead", "trail"):
        _assert_mirrored(left[side][0][None], right[side][0][None])
        np.testing.assert_allclose(
            left[side][1] @ frame, (right[side][1] @ frame) * _FLIP_Y, atol=1e-12
        )


@pytest.mark.parametrize("handedness", ("right", "left"))
def test_skinned_hand_frame_is_a_proper_rotation(handedness: str) -> None:
    frame = sch.skinned_hand_frame(_club(handedness), handedness=handedness)
    rotation = frame[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    # The swing plane holds the top of the grip: both clubs share the frame.
    np.testing.assert_allclose(frame, sch.skinned_hand_frame(_club()), atol=1e-12)


def test_left_club_head_is_the_mirror_of_the_right_head() -> None:
    right, left = assembly_meshes(_club()), assembly_meshes(_club("left"))
    assert set(left) == set(right)
    _assert_mirrored(left["head"].vertices, right["head"].vertices)
    np.testing.assert_array_equal(left["head"].faces, right["head"].faces[:, ::-1])
    assert left["head"].volume() > 0.0
    assert left["head"].volume() == pytest.approx(right["head"].volume(), rel=1e-9)
    _assert_mirrored(
        clubface_centre(_club("left"))[None], clubface_centre(_club())[None]
    )
    frame = sch.skinned_hand_frame(_club())[:3, :3]
    np.testing.assert_allclose(
        clubface_vector(_club("left")) @ frame,
        (clubface_vector(_club()) @ frame) * _FLIP_Y,
        atol=1e-12,
    )


def test_assembly_from_spec_reads_the_subject_handedness() -> None:
    grip = np.eye(4)
    grip[1, 3] = -1.0
    body = {
        "name": "Golfer Clubface Vector",
        "solids": [{"name": "Grip", "placement": grip.tolist()}],
    }
    spec = {"bodies": [body]}
    assert assembly_from_spec(spec).handedness == "right"
    left = assembly_from_spec({**spec, "subject": {"handedness": "left"}})
    assert left.handedness == "left"
    assert dataclasses.replace(left, handedness="right") == assembly_from_spec(spec)


def test_right_handed_outputs_are_unchanged() -> None:
    club = _club()
    assert sch.owner_grip_spec() == sch.owner_grip_spec(handedness="right")
    assert sch.owner_grip_spec(handedness="left").handedness == "left"
    np.testing.assert_array_equal(
        sch.skinned_hand_frame(club), sch.skinned_hand_frame(club, handedness="right")
    )
    default, explicit = (
        sch.wrist_joints_m(club),
        sch.wrist_joints_m(club, handedness="right"),
    )
    for side in default:
        np.testing.assert_array_equal(default[side], explicit[side])
    default_m = sch.skinned_hand_meshes(club)
    explicit_m = sch.skinned_hand_meshes(club, handedness="right")
    for label, mesh in default_m.items():
        np.testing.assert_array_equal(mesh.vertices, explicit_m[label].vertices)
        np.testing.assert_array_equal(mesh.faces, explicit_m[label].faces)


def test_invalid_or_mismatched_handedness_is_refused() -> None:
    with pytest.raises(ValueError, match="handedness"):
        sch.owner_grip_spec(handedness="ambidextrous")
    with pytest.raises(ValueError, match="handedness"):
        sch.build_seated_pair(handedness="both")
    with pytest.raises(ValueError, match="handedness"):
        _club("sinister")
    with pytest.raises(ValueError, match="handedness"):
        sch.skinned_hand_frame(_club(), handedness="up")
    with pytest.raises(ValueError, match="right-handed club"):
        sch.wrist_joints_m(_club(), handedness="left")


def test_skinned_arm_refit_still_refuses_a_left_handed_club() -> None:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import skinned_arm_refit

    grip = np.eye(4)
    grip[1, 3] = -1.0
    spec = {
        "coordinate_order": ["a"],
        "bodies": [
            {
                "name": "Golfer Clubface Vector",
                "solids": [{"name": "Grip", "placement": grip.tolist()}],
            }
        ],
        "subject": {"handedness": "left"},
    }
    with pytest.raises(ValueError, match="left-handed skinned arm refit"):
        skinned_arm_refit._checked(spec, spec, np.zeros(1), 1.0, (), {}, 0.0)
