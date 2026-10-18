"""Grip hands placed on the club's grip in the club-body frame (OSV-2, #11728).

The curled-finger hand meshes of :mod:`grip_hand_mesh` are built in the
hand-grip frame; :mod:`grip_hands` places them on the visual grip tube of a
:class:`ClubAssembly`, so every engine that draws the club draws the hands
wrapped on its grip. These tests measure the placement in the club-body frame
(origin at the sole point, shaft toward the grip along ``-y``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import grip_hands as gh
from src.shared.python.model_appearance.grip_hand_mesh import (
    MAX_PAD_CLEARANCE_M,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    GripStyle,
    Hand,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPECS = {
    key: ROOT / f"docs/development/full_body_models/full_body_spec_anthro_{key}.json"
    for key in ("driver", "iron7")
}
#: A wrapped hand's vertex centroid lies within this of the grip axis (measured
#: 0.5-3.3 mm); an
#: open (mitten) hand beside the grip sits ~3-5 cm off it.
MAX_CENTROID_OFF_AXIS_M = 0.01
#: Share of the 36 ten-degree azimuth sectors around the grip that the hand
#: covers within 3 cm of the grip surface (measured 0.92-1.0): the fingers,
#: palm and thumb wrap it.
MIN_WRAP_COVERAGE = 0.9


def _club(key: str) -> ca.ClubAssembly:
    club = ca.assembly_from_spec(json.loads(SPECS[key].read_text(encoding="utf-8")))
    assert club is not None
    return club


def _axis(club: ca.ClubAssembly) -> tuple[np.ndarray, np.ndarray]:
    """(butt point, unit grip axis butt -> head) in the club-body frame."""
    butt = np.array([0.0, -club.length_m, club.axis_offset_m])
    return butt, np.array([0.0, 1.0, 0.0])


def _radial(club: ca.ClubAssembly, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distance below the butt, radial distance from the grip axis) per point."""
    butt, g = _axis(club)
    rel = pts - butt
    along = rel @ g
    radial = np.linalg.norm(rel - np.outer(along, g), axis=1)
    return along, radial


@pytest.mark.parametrize("key", sorted(SPECS))
@pytest.mark.parametrize("hand", list(Hand))
def test_hand_frame_origin_is_on_the_grip_axis_at_the_pose_depth(key, hand) -> None:
    club = _club(key)
    frame = gh.grip_hand_frame(club, hand)
    rot, origin = frame[:3, :3], frame[:3, 3]
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    along, radial = _radial(club, origin[None, :])
    assert radial[0] < 1e-9  # on the grip axis
    assert along[0] == pytest.approx(DEFAULT_GRIP_POSE.distance_below_butt_m(hand))
    np.testing.assert_allclose(rot[:, 2], [0.0, 1.0, 0.0], atol=1e-12)  # z = g


@pytest.mark.parametrize("key", sorted(SPECS))
def test_azimuth_zero_is_toe_axis_and_y_is_minus_face_normal(key) -> None:
    """HRG-A1 (#13162): ``x = +t`` (toward the chin), ``y = -n`` (trail side).

    ``t`` is heel-to-toe and ``n`` the face normal, both projected off the
    shaft; the frame is a proper rotation about the shaft.
    """
    club = _club(key)
    rot = gh.grip_hand_frame(club, Hand.LEAD)[:3, :3]
    g = np.array([0.0, 1.0, 0.0])
    normal = ca.clubface_vector(club)
    normal = normal - normal[1] * g
    normal /= np.linalg.norm(normal)
    toe = np.cross(g, normal)  # right-handed (-n, -t, g)  =>  t = g x n
    np.testing.assert_allclose(rot[:, 0], toe, atol=1e-9)
    np.testing.assert_allclose(rot[:, 1], -normal, atol=1e-9)
    np.testing.assert_allclose(np.cross(rot[:, 0], rot[:, 1]), g, atol=1e-9)
    head = ca.assembly_meshes(club)["head"].vertices
    toe_side = (head.mean(axis=0) - np.array([0, 0, club.axis_offset_m])) @ rot[:, 0]
    assert toe_side > 0.0  # the head body lies toward the toe, i.e. along +x


def _club_direction(club, hand, azimuth_deg: float) -> np.ndarray:
    """Unit club-frame direction of a hand-frame azimuth about the shaft."""
    rot = gh.grip_hand_frame(club, hand)[:3, :3]
    a = np.radians(azimuth_deg)
    return rot[:, 0] * np.cos(a) + rot[:, 1] * np.sin(a)


def _face_normal_off_shaft(club) -> np.ndarray:
    n = ca.clubface_vector(club)
    n = n - n[1] * np.array([0.0, 1.0, 0.0])
    return n / np.linalg.norm(n)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _azimuth_of(mesh_part) -> float:
    v = mesh_part.mesh.vertices
    return float(np.degrees(np.arctan2(v[:, 1].mean(), v[:, 0].mean())))


@pytest.mark.parametrize("key", sorted(SPECS))
def test_trail_palm_faces_the_target(key) -> None:
    """HRG-A1: the trail palm's inward normal is within 20 deg of ``+n``."""
    club = _club(key)
    hand = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=gh.grip_radius_at(club, 0.2))
    palm = next(c for c in hand.components if c.name == "palm")
    inward = _club_direction(club, Hand.TRAIL, _azimuth_of(palm) + 180.0)
    assert _angle_deg(inward, _face_normal_off_shaft(club)) < 20.0


@pytest.mark.parametrize("key", sorted(SPECS))
def test_lead_knuckle_faces_the_target_and_dorsum_sits_at_minus_35(key) -> None:
    """HRG-A1: lead index MCP within 15 deg of ``+n``; dorsal midpoint -35 +/- 15."""
    club = _club(key)
    hand = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=gh.grip_radius_at(club, 0.1))
    parts = {c.name: c for c in hand.components}
    mcp = DEFAULT_GRIP_POSE.lead_v_azimuth_deg - 110.0  # index MCP, hand azimuth
    knuckle = _club_direction(club, Hand.LEAD, mcp)
    assert _angle_deg(knuckle, _face_normal_off_shaft(club)) < 15.0
    assert abs(_azimuth_of(parts["palm"]) - (-35.0)) <= 15.0


def test_frame_rejects_bad_inputs() -> None:
    club = _club("driver")
    with pytest.raises(TypeError, match="ClubAssembly"):
        gh.grip_hand_frame("driver", Hand.LEAD)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Hand"):
        gh.grip_hand_frame(club, "lead")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GripPose"):
        gh.grip_hand_meshes(club, pose="overlap")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grip"):
        gh.grip_radius_at(club, -0.01)
    with pytest.raises(ValueError, match="grip"):
        gh.grip_radius_at(club, club.grip_length_m + 0.01)


def test_grip_radius_follows_the_tube_taper() -> None:
    club = _club("driver")
    assert gh.grip_radius_at(club, 0.0) == pytest.approx(ca.GRIP_BUTT_RADIUS_M)
    assert gh.grip_radius_at(club, club.grip_length_m) == pytest.approx(
        ca.GRIP_TIP_RADIUS_M
    )


@pytest.mark.parametrize("key", sorted(SPECS))
@pytest.mark.parametrize("style", list(GripStyle))
def test_hands_wrap_the_grip_tube_without_entering_it(key, style) -> None:
    club = _club(key)
    pose = GripPose(style=style)
    meshes = gh.grip_hand_meshes(club, pose=pose)
    assert set(meshes) == {"lead_hand", "trail_hand"}
    for name, mesh in meshes.items():
        assert mesh.volume() > 0.0, name
        along, radial = _radial(club, mesh.vertices)
        local = np.array(
            [
                gh.grip_radius_at(club, float(np.clip(a, 0, club.grip_length_m)))
                for a in along
            ]
        )
        gap = float(np.min(radial - local))
        assert 0.0 <= gap <= MAX_PAD_CLEARANCE_M, (name, gap)
        assert along.min() > -0.005, name  # nothing past the butt cap
        assert along.max() < club.grip_length_m, name  # stays on the grip
        _, off_axis = _radial(club, mesh.vertices.mean(axis=0)[None, :])
        assert off_axis[0] < MAX_CENTROID_OFF_AXIS_M, (name, off_axis[0])
        near = radial - local < 0.03
        rot = gh.grip_hand_frame(club, Hand.LEAD)[:3, :3]
        rel = mesh.vertices[near] - _axis(club)[0]
        az = np.degrees(np.arctan2(rel @ rot[:, 1], rel @ rot[:, 0])) % 360.0
        coverage = len(np.unique((az // 10).astype(int))) / 36.0
        assert coverage >= MIN_WRAP_COVERAGE, (name, coverage)


def test_lead_hand_is_above_the_trail_hand() -> None:
    club = _club("driver")
    meshes = gh.grip_hand_meshes(club)
    lead, _ = _radial(club, meshes["lead_hand"].vertices)
    trail, _ = _radial(club, meshes["trail_hand"].vertices)
    assert lead.mean() < trail.mean()
    gap = trail.mean() - lead.mean()
    assert gap == pytest.approx(DEFAULT_GRIP_POSE.hand_spacing_m, abs=0.01)


def test_assembly_with_hands_adds_two_glove_meshes_and_keeps_the_club() -> None:
    club = _club("driver")
    plain = ca.assembly_meshes(club)
    full = gh.assembly_meshes_with_hands(club)
    assert list(full) == ["shaft", "grip", "head", "lead_hand", "trail_hand"]
    for part in ("shaft", "grip", "head"):
        np.testing.assert_array_equal(full[part].vertices, plain[part].vertices)
    for part in ("lead_hand", "trail_hand"):
        assert ca.PART_MATERIALS[part] == "glove_white"
