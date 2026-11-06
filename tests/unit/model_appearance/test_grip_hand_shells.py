"""#13997: dorsum, thumb-web, 270° shell wrap, tapered-grip clearance."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    MAX_PAD_CLEARANCE_M,
    GripHandMesh,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import GripPose, GripStyle, Hand

pytestmark = pytest.mark.unit

RADII_M = (0.0127, 0.0138, 0.0140, 0.0145)
_GRIP_LENGTH_M, _GRIP_BUTT_RADIUS_M, _GRIP_TIP_RADIUS_M = 0.265, 0.0118, 0.0105


def _grip_radius_at(distance_below_butt_m: float) -> float:
    f = float(distance_below_butt_m) / _GRIP_LENGTH_M
    return _GRIP_BUTT_RADIUS_M + f * (_GRIP_TIP_RADIUS_M - _GRIP_BUTT_RADIUS_M)


def _taper_clearance(mesh: Mesh, *, depth_below_butt_m: float) -> float:
    pts, rho = mesh.vertices, np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    r_surface = np.array(
        [_grip_radius_at(depth_below_butt_m + float(z)) for z in pts[:, 2]]
    )
    return float(np.min(rho - r_surface))


def _rho(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0], points[:, 1])


def _parts(result: GripHandMesh, finger: str):
    return [p for p in result.components if p.finger == finger]


def _span_deg(angles_deg: np.ndarray) -> float:
    a = np.sort(np.mod(angles_deg, 360.0))
    if len(a) < 2:
        return 360.0
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - np.max(gaps))


def test_hand_section_covers_270_deg_around_grip():
    for style in GripStyle:
        pose = GripPose(style=style)
        for hand in Hand:
            r = build_grip_hand_mesh(
                hand,
                grip_radius_m=0.014,
                pose=pose,
                params=DEFAULT_HAND_MESH_PARAMS,
            )
            on = [p for p in r.components if p.on_grip]
            assert on
            pts = np.vstack([p.mesh.vertices for p in on])
            ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            cov = _span_deg(ang - float(r.v_azimuth_deg))
            assert cov >= 270.0 - 0.5, (style, hand, cov)


def _baked_on_tapered_grip(hand: Hand, pose: GripPose) -> GripHandMesh:
    params = DEFAULT_HAND_MESH_PARAMS
    depth = pose.distance_below_butt_m(hand)
    nominal = build_grip_hand_mesh(
        hand, grip_radius_m=_grip_radius_at(depth), pose=pose, params=params
    )
    nearest, _ = nominal.on_grip_extent_below_butt_m()
    radius = _grip_radius_at(min(max(nearest, 0.0), _GRIP_LENGTH_M))
    return build_grip_hand_mesh(hand, grip_radius_m=radius, pose=pose, params=params)


def test_each_contact_patch_gap_within_0_to_3_mm():
    pose = GripPose(style=GripStyle.TEN_FINGER)
    for hand in Hand:
        result = _baked_on_tapered_grip(hand, pose)
        depth = pose.distance_below_butt_m(hand)
        on = [p for p in result.components if p.on_grip and p.finger != "dorsum"]
        assert on
        nearest, farthest = result.on_grip_extent_below_butt_m()
        taper_extra = abs(_grip_radius_at(nearest) - _grip_radius_at(farthest))
        hi = (
            MAX_PAD_CLEARANCE_M
            + taper_extra
            + abs(_grip_radius_at(depth) - float(result.grip_radius_m))
        )
        for part in on:
            lo = _taper_clearance(part.mesh, depth_below_butt_m=depth)
            assert 0.0 <= lo <= hi + 1e-6, (hand, part.name, lo)


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("style", list(GripStyle))
@pytest.mark.parametrize("radius", RADII_M)
def test_dorsum_shell_exists_beyond_mcp_radius(
    hand: Hand, style: GripStyle, radius: float
):
    result = build_grip_hand_mesh(hand, grip_radius_m=radius, style=style)
    dorsum = _parts(result, "dorsum")
    assert dorsum, "no dorsum shell"
    assert not any(p.on_grip for p in dorsum)
    dorsum_rho = _rho(np.vstack([p.mesh.vertices for p in dorsum]))
    assert dorsum_rho.min() > radius + MAX_PAD_CLEARANCE_M
    knuckles = [
        float(_rho(p.mesh.vertices).max())
        for p in result.components
        if p.on_grip and p.name.endswith("_proximal")
    ]
    assert dorsum_rho.max() > max(knuckles)


def _wrapped_deg(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("style", list(GripStyle))
def test_mirrored_hand_matches_handedness(style: GripStyle):
    pose = GripPose(style=style)
    lead = build_grip_hand_mesh(Hand.LEAD, grip_radius_m=0.014, pose=pose)
    trail = build_grip_hand_mesh(Hand.TRAIL, grip_radius_m=0.014, pose=pose)
    for name in ("palm", "dorsum", "middle_proximal", "ring_proximal"):
        a = next(p for p in lead.components if p.name == name).mesh.vertices
        b = next(p for p in trail.components if p.name == name).mesh.vertices
        az_a = _wrapped_deg(
            np.degrees(np.arctan2(a[:, 1], a[:, 0])) - lead.v_azimuth_deg
        )
        az_b = _wrapped_deg(
            np.degrees(np.arctan2(b[:, 1], b[:, 0])) - trail.v_azimuth_deg
        )
        for got, want in ((_rho(a), _rho(b)), (a[:, 2], b[:, 2]), (az_a, -az_b)):
            np.testing.assert_allclose(np.sort(got), np.sort(want), atol=1e-6)
        assert np.sign(np.median(az_a)) == lead.curl_sign, name
        assert np.sign(np.median(az_b)) == trail.curl_sign, name


def test_thumb_web_shell_is_on_grip():
    for style in GripStyle:
        for hand in Hand:
            result = build_grip_hand_mesh(hand, grip_radius_m=0.014, style=style)
            web = [p for p in result.components if p.name == "thumb_web"]
            assert len(web) == 1
            assert web[0].on_grip and web[0].finger == "thumb_web"
