"""Trail-palm-on-lead-thumb contact patch frame (HRG-H1, #13189)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance.grip_contact_patch import (
    ThumbPatch,
    thumb_palm_patch,
)
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_hands import (
    grip_hand_frame,
    grip_radius_at,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SURFACE_TOL_M = 1e-3
PALM_MARGIN_M = 2e-3
MIRROR_TOL = 1e-9


def _club() -> ca.ClubAssembly:
    club = ca.assembly_from_spec(json.loads(SPEC.read_text(encoding="utf-8")))
    assert club is not None
    return club


def _patch(handedness: str = "right", pose: GripPose = DEFAULT_GRIP_POSE):
    club = _club()
    frame = grip_hand_frame(club, Hand.LEAD, pose)
    radius = grip_radius_at(club, pose.distance_below_butt_m(Hand.LEAD))
    return (
        thumb_palm_patch(frame, grip_radius_m=radius, pose=pose, handedness=handedness),
        frame,
        radius,
    )


def _to_hand(frame: np.ndarray, point: np.ndarray) -> np.ndarray:
    return frame[:3, :3].T @ (point - frame[:3, 3])


def test_patch_lies_on_the_lead_thumb_surface() -> None:
    patch, frame, _ = _patch()
    p = _to_hand(frame, np.array(patch.point_club_m))
    params = DEFAULT_HAND_MESH_PARAMS
    z0, _, z2 = params.thumb_stations_m
    centre = np.array(patch.thumb_axis_hand_m)
    z = np.clip(p[2], z0, z2)
    gap = (
        np.linalg.norm(p - np.array([centre[0], centre[1], z])) - params.thumb_radius_m
    )
    assert abs(gap) <= SURFACE_TOL_M


def test_patch_is_inside_the_trail_palm_block_by_the_margin() -> None:
    patch, frame, radius = _patch()
    pose = DEFAULT_GRIP_POSE
    trail = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=radius, pose=pose)
    palm = next(c for c in trail.components if c.finger == "palm").mesh.vertices
    dz = pose.distance_below_butt_m(Hand.TRAIL) - pose.distance_below_butt_m(Hand.LEAD)
    p = _to_hand(frame, np.array(patch.point_club_m))
    rad = np.hypot(p[0], p[1])
    az = np.degrees(np.arctan2(p[1], p[0]))
    prad = np.hypot(palm[:, 0], palm[:, 1])
    paz = np.degrees(np.arctan2(palm[:, 1], palm[:, 0]))
    assert prad.min() + PALM_MARGIN_M <= rad <= prad.max() - PALM_MARGIN_M
    assert paz.min() <= az <= paz.max()
    assert palm[:, 2].min() + dz + PALM_MARGIN_M <= p[2] <= palm[:, 2].max() + dz


def test_normal_is_unit_and_points_from_the_thumb_to_the_palm() -> None:
    patch, frame, _ = _patch()
    n = np.array(patch.normal_club)
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)
    palm_ref = np.array(patch.palm_reference_club_m)
    p = np.array(patch.point_club_m)
    assert (palm_ref - p) @ n > 0.0


def test_left_handed_patch_is_the_reflection_of_the_right_handed_one() -> None:
    right, frame, _ = _patch("right")
    left, _, _ = _patch("left")
    t_hat = frame[:3, 0]  # the plane of the shaft and the face normal has normal t
    origin = frame[:3, 3]

    def mirror_point(x: np.ndarray) -> np.ndarray:
        return x - 2.0 * ((x - origin) @ t_hat) * t_hat

    def mirror_dir(x: np.ndarray) -> np.ndarray:
        return x - 2.0 * (x @ t_hat) * t_hat

    got = np.array(left.point_club_m)
    np.testing.assert_allclose(
        got, mirror_point(np.array(right.point_club_m)), atol=MIRROR_TOL
    )
    np.testing.assert_allclose(
        np.array(left.normal_club),
        mirror_dir(np.array(right.normal_club)),
        atol=MIRROR_TOL,
    )
    assert (left.lead_side, left.trail_side) == ("R", "L")
    assert (right.lead_side, right.trail_side) == ("L", "R")


def test_patch_follows_the_pose_not_a_constant() -> None:
    base, _, _ = _patch()
    wider = dataclasses.replace(DEFAULT_GRIP_POSE, hand_spacing_m=0.05)
    moved, _, _ = _patch(pose=wider)
    assert (
        np.linalg.norm(np.array(base.point_club_m) - np.array(moved.point_club_m))
        > 1e-4
    )


def test_patch_is_a_frozen_value() -> None:
    patch, _, _ = _patch()
    assert isinstance(patch, ThumbPatch)
    with pytest.raises(dataclasses.FrozenInstanceError):
        patch.point_club_m = (0.0, 0.0, 0.0)  # type: ignore[misc]


def test_preconditions_are_checked() -> None:
    _, frame, radius = _patch()
    with pytest.raises(ValueError, match="handedness"):
        thumb_palm_patch(frame, grip_radius_m=radius, handedness="ambi")
    with pytest.raises(ValueError, match="4x4"):
        thumb_palm_patch(np.eye(3), grip_radius_m=radius)
    bad = frame.copy()
    bad[:3, :3] *= 2.0
    with pytest.raises(ValueError, match="rotation"):
        thumb_palm_patch(bad, grip_radius_m=radius)
    with pytest.raises(ValueError, match="grip_radius_m"):
        thumb_palm_patch(frame, grip_radius_m=0.5)
    with pytest.raises(TypeError, match="GripPose"):
        thumb_palm_patch(frame, grip_radius_m=radius, pose="x")  # type: ignore[arg-type]
