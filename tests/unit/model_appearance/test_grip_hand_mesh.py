"""Curled-finger grip hand mesh wrapped on the grip cylinder (OSV-2, #11728).

Clearances are measured against the analytic grip cylinder (the grip axis is
the hand-grip frame's ``z`` axis), on the mesh vertices and on dense
barycentric samples of every triangle, so a facet that sags into the grip
between two vertices is caught too.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    MAX_PAD_CLEARANCE_M,
    GripHandMesh,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    GripStyle,
    Hand,
)

pytestmark = pytest.mark.unit

#: Grip radii of the equipment set (driver, iron, wedge) and the contact-pad
#: layout's nominal 1-inch grip.
RADII_M = (0.0127, 0.0138, 0.0140, 0.0145)

_BARY = np.array(
    [(a, b, 1.0 - a - b) for a in np.linspace(0, 1, 6) for b in np.linspace(0, 1, 6)]
)
_BARY = _BARY[_BARY[:, 2] >= -1e-12]


def _surface_points(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]  # (M, 3, 3)
    samples = np.einsum("kj,mjd->mkd", _BARY, tri).reshape(-1, 3)
    return np.vstack([mesh.vertices, samples])


def _clearance(mesh: Mesh, radius: float) -> float:
    """Minimum distance from the mesh surface to the grip cylinder surface."""
    pts = _surface_points(mesh)
    return float(np.min(np.hypot(pts[:, 0], pts[:, 1]) - radius))


def _is_watertight(mesh: Mesh) -> bool:
    directed = Counter()
    for a, b, c in mesh.faces.tolist():
        directed.update([(a, b), (b, c), (c, a)])
    if any(n != 1 for n in directed.values()):
        return False  # duplicated or inconsistently wound edge
    return all((b, a) in directed for (a, b) in directed)


def _build(hand: Hand, radius: float, style: GripStyle = GripStyle.OVERLAP):
    pose = GripPose(style=style)
    return build_grip_hand_mesh(hand, grip_radius_m=radius, pose=pose)


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("style", list(GripStyle))
def test_mesh_is_watertight_with_outward_winding(hand: Hand, style: GripStyle):
    result = _build(hand, 0.0140, style)
    assert isinstance(result, GripHandMesh)
    assert _is_watertight(result.mesh)
    for part in result.components:
        assert _is_watertight(part.mesh), part.name
        assert part.mesh.volume() > 0.0, part.name
    assert np.isfinite(result.mesh.vertices).all()


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("radius", RADII_M)
def test_no_surface_point_is_inside_the_grip(hand: Hand, radius: float):
    result = _build(hand, radius)
    radial = np.hypot(result.mesh.vertices[:, 0], result.mesh.vertices[:, 1])
    assert radial.min() >= radius  # no vertex inside the grip cylinder
    assert _clearance(result.mesh, radius) >= 0.0  # nor any facet


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("radius", RADII_M)
def test_every_finger_on_the_grip_rests_within_3_mm(hand: Hand, radius: float):
    result = _build(hand, radius, GripStyle.TEN_FINGER)
    on_grip = [p for p in result.components if p.on_grip]
    fingers = {p.finger for p in on_grip}
    assert fingers == {*FINGER_NAMES, "thumb", "palm"}
    for finger in fingers:
        meshes = [p.mesh for p in on_grip if p.finger == finger]
        gap = min(_clearance(m, radius) for m in meshes)
        assert 0.0 <= gap <= MAX_PAD_CLEARANCE_M == 0.003, (finger, gap)
    pad = _clearance(Mesh(*_stack(on_grip)), radius)
    assert 0.0 <= pad <= 0.003
    assert result.min_pad_clearance_m == pytest.approx(pad, abs=2e-4)


def _stack(parts):
    verts, faces, offset = [], [], 0
    for p in parts:
        verts.append(p.mesh.vertices)
        faces.append(p.mesh.faces + offset)
        offset += len(p.mesh.vertices)
    return np.vstack(verts), np.vstack(faces)


def test_every_phalanx_is_present():
    result = _build(Hand.LEAD, 0.0140)
    names = {p.name for p in result.components}
    for finger in FINGER_NAMES:
        for bone in ("proximal", "middle", "distal"):
            assert f"{finger}_{bone}" in names
    assert {"palm", "thumb_proximal", "thumb_distal"} <= names


@pytest.mark.parametrize("style", [GripStyle.OVERLAP, GripStyle.INTERLOCK])
def test_linked_trail_little_finger_leaves_the_grip(style: GripStyle):
    trail = _build(Hand.TRAIL, 0.0140, style)
    little = [p for p in trail.components if p.finger == "little"]
    assert little and not any(p.on_grip for p in little)
    lead = _build(Hand.LEAD, 0.0140, style)
    assert all(p.on_grip for p in lead.components if p.finger == "little")


def _azimuth_deg(points: np.ndarray) -> float:
    c = points.mean(axis=0)
    return float(np.degrees(np.arctan2(c[1], c[0])))


def _part(result: GripHandMesh, name: str):
    return next(p for p in result.components if p.name == name)


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def test_lead_and_trail_fingers_are_mirrored_about_their_v_azimuth():
    pose = DEFAULT_GRIP_POSE
    lead = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=pose)
    trail = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=0.014, pose=pose)
    assert lead.curl_sign == -trail.curl_sign
    v_lead = pose.placement(Hand.LEAD).v_azimuth_deg
    v_trail = pose.placement(Hand.TRAIL).v_azimuth_deg
    assert lead.v_azimuth_deg == v_lead and trail.v_azimuth_deg == v_trail
    for name in ("palm", "index_proximal", "index_middle", "middle_distal"):
        a = _wrap(_azimuth_deg(_part(lead, name).mesh.vertices) - v_lead)
        b = _wrap(_azimuth_deg(_part(trail, name).mesh.vertices) - v_trail)
        assert a == pytest.approx(-b, abs=1e-6), name
    # The proximal-to-distal wrap runs in the curl direction of each hand.
    for result in (lead, trail):
        prox = _azimuth_deg(_part(result, "index_proximal").mesh.vertices)
        mid = _azimuth_deg(_part(result, "index_middle").mesh.vertices)
        assert np.sign(_wrap(mid - prox)) == result.curl_sign


def test_thumb_sits_at_the_grip_pose_thumb_azimuth():
    pose = DEFAULT_GRIP_POSE
    for hand in Hand:
        result = build_grip_hand_mesh(hand, grip_radius_m=0.014, pose=pose)
        expected = pose.placement(hand).thumb_offset_deg
        thumb = _part(result, "thumb_distal").mesh.vertices
        assert _azimuth_deg(thumb) == pytest.approx(expected, abs=1e-6)


def test_hands_are_ordered_along_the_shaft_like_grip_pose():
    pose = DEFAULT_GRIP_POSE
    lead = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=pose)
    trail = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=0.014, pose=pose)
    assert lead.distance_below_butt_m == pose.distance_below_butt_m(Hand.LEAD)
    assert trail.distance_below_butt_m == pose.distance_below_butt_m(Hand.TRAIL)

    def finger_depth(result: GripHandMesh, finger: str) -> float:
        z = _part(result, f"{finger}_proximal").mesh.vertices[:, 2].mean()
        return result.distance_below_butt_m + float(z)

    for result in (lead, trail):  # little finger toward the butt, index below
        depths = [finger_depth(result, f) for f in reversed(FINGER_NAMES)]
        assert depths == sorted(depths)
    lead_lowest = max(
        result_z for result_z in lead.on_grip_depths_below_butt_m().values()
    )
    trail_highest = min(trail.on_grip_depths_below_butt_m().values())
    assert lead_lowest < trail_highest
    # The default lead hand does not hang off the butt end of the grip.
    assert lead.on_grip_extent_below_butt_m()[0] >= 0.0


def test_overlap_little_finger_lies_over_the_lead_index_middle_gap():
    pose = DEFAULT_GRIP_POSE
    lead = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=pose)
    trail = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=0.014, pose=pose)
    depths = lead.on_grip_depths_below_butt_m()
    gap = 0.5 * (depths["index"] + depths["middle"])
    z = _part(trail, "little_proximal").mesh.vertices[:, 2].mean()
    assert trail.distance_below_butt_m + z == pytest.approx(gap, abs=1e-9)


@pytest.mark.parametrize("radius", [0.0, -0.01, float("nan"), float("inf"), 0.05])
def test_invalid_grip_radius_is_rejected(radius: float):
    with pytest.raises(ValueError, match="grip_radius_m"):
        build_grip_hand_mesh(Hand.LEAD, grip_radius_m=radius)


def test_non_numeric_grip_radius_is_a_type_error():
    with pytest.raises(TypeError, match="grip_radius_m"):
        build_grip_hand_mesh(Hand.LEAD, grip_radius_m="0.014")  # type: ignore[arg-type]


def test_invalid_hand_pose_and_style_are_rejected():
    with pytest.raises(TypeError, match="hand"):
        build_grip_hand_mesh("lead", grip_radius_m=0.014)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pose"):
        build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="style"):
        build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, style="overlap")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="style"):
        GripPose(style="claw")  # type: ignore[arg-type]


def test_style_override_takes_precedence_over_the_pose():
    result = build_grip_hand_mesh(
        Hand.TRAIL, grip_radius_m=0.014, style=GripStyle.TEN_FINGER
    )
    assert result.style is GripStyle.TEN_FINGER
    assert all(p.on_grip for p in result.components)
