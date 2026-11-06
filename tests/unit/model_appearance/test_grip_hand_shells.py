"""Dorsum, thumb-web and coverage shells for the grip hand mesh (#13997).

The fingers alone read as short and straight: the hand needs a dorsum
(metacarpal block) over the knuckles, a thumb web between index MCP and thumb,
and shells whose on-grip clearance stays inside the pad band even on a tapered
grip tube. These tests measure the shells on the analytic grip cylinder (the
hand-grip frame's ``z`` axis) and on the club's linear taper (same butt-to-tip
radii and length as ``club_assembly``).
"""

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

#: Canonical visual-grip tube (same defaults as ``club_assembly``).
_GRIP_LENGTH_M = 0.265
_GRIP_BUTT_RADIUS_M = 0.0118
_GRIP_TIP_RADIUS_M = 0.0105


def _grip_radius_at(distance_below_butt_m: float) -> float:
    """Linear taper from butt to tip (metres below butt → radius)."""
    f = float(distance_below_butt_m) / _GRIP_LENGTH_M
    return _GRIP_BUTT_RADIUS_M + f * (_GRIP_TIP_RADIUS_M - _GRIP_BUTT_RADIUS_M)


_BARY = np.array(
    [(a, b, 1.0 - a - b) for a in np.linspace(0, 1, 6) for b in np.linspace(0, 1, 6)]
)
_BARY = _BARY[_BARY[:, 2] >= -1e-12]


def _surface_points(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    samples = np.einsum("kj,mjd->mkd", _BARY, tri).reshape(-1, 3)
    return np.vstack([mesh.vertices, samples])


def _taper_clearance(mesh: Mesh, *, depth_below_butt_m: float) -> float:
    """Minimum clearance (m) from mesh surface to the taper surface at each z."""
    pts = _surface_points(mesh)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    r_surface = np.array(
        [_grip_radius_at(depth_below_butt_m + float(z)) for z in pts[:, 2]]
    )
    return float(np.min(rho - r_surface))


def _rho(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0], points[:, 1])


def _parts(result: GripHandMesh, finger: str):
    return [p for p in result.components if p.finger == finger]


def _azimuth_deg(points: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(points[:, 1], points[:, 0]))


def _span_deg(angles: np.ndarray) -> float:
    """Largest gap between adjacent sorted angles on the circle (degrees)."""
    a = np.sort(np.mod(angles, 360.0))
    if len(a) < 2:
        return 360.0
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    return float(360.0 - np.max(gaps))


def test_hand_section_covers_270_deg_around_grip():
    """On-grip shells wrap at least 270° around the shaft for every style."""
    for style in GripStyle:
        pose = GripPose(style=style)
        for hand in Hand:
            result = build_grip_hand_mesh(
                hand,
                grip_radius_m=0.014,
                pose=pose,
                params=DEFAULT_HAND_MESH_PARAMS,
            )
            on = [p for p in result.components if p.on_grip]
            assert on
            angles = _azimuth_deg(np.vstack([p.mesh.vertices for p in on]))
            rel = angles - float(result.v_azimuth_deg)
            cov = _span_deg(rel)
            assert cov >= 270.0 - 0.5, (style, hand, cov)


def _baked_on_tapered_grip(hand: Hand, pose: GripPose) -> GripHandMesh:
    """Same radius choice as :func:`grip_hands._placed_hand` (butter butt-most)."""
    params = DEFAULT_HAND_MESH_PARAMS
    depth = pose.distance_below_butt_m(hand)
    nominal = build_grip_hand_mesh(
        hand,
        grip_radius_m=_grip_radius_at(depth),
        pose=pose,
        params=params,
    )
    nearest, _ = nominal.on_grip_extent_below_butt_m()
    radius = _grip_radius_at(min(max(nearest, 0.0), _GRIP_LENGTH_M))
    return build_grip_hand_mesh(hand, grip_radius_m=radius, pose=pose, params=params)


def test_each_contact_patch_gap_within_0_to_3_mm():
    """Each on-grip patch clears the tapered grip surface by [0, 3] mm (+ taper)."""
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
    assert not any(p.on_grip for p in dorsum)  # it never rests on the grip
    dorsum_rho = _rho(np.vstack([p.mesh.vertices for p in dorsum]))
    # Clear of the grip: it sits on the palm, not on the grip surface.
    assert dorsum_rho.min() > radius + MAX_PAD_CLEARANCE_M
    # Over the knuckles: past the outermost MCP of every finger it holds with.
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
    shells = ("palm", "dorsum", "middle_proximal", "ring_proximal")
    for name in shells:
        a = next(p for p in lead.components if p.name == name).mesh.vertices
        b = next(p for p in trail.components if p.name == name).mesh.vertices
        # Trail is lead reflected about its V azimuth: same radii, axial
        # stations and azimuth offsets from V of opposite sign (as point sets).
        az_a = _wrapped_deg(
            np.degrees(np.arctan2(a[:, 1], a[:, 0])) - lead.v_azimuth_deg
        )
        az_b = _wrapped_deg(
            np.degrees(np.arctan2(b[:, 1], b[:, 0])) - trail.v_azimuth_deg
        )
        for got, want in ((_rho(a), _rho(b)), (a[:, 2], b[:, 2]), (az_a, -az_b)):
            np.testing.assert_allclose(np.sort(got), np.sort(want), atol=1e-6)
        # Curl direction: the shell lies on the hand's curl side of its V.
        assert np.sign(np.median(az_a)) == lead.curl_sign, name
        assert np.sign(np.median(az_b)) == trail.curl_sign, name


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("style", list(GripStyle))
def test_thumb_web_shell_is_on_grip(hand: Hand, style: GripStyle):
    result = build_grip_hand_mesh(hand, grip_radius_m=0.014, style=style)
    web = [p for p in result.components if p.name == "thumb_web"]
    assert len(web) == 1
    assert web[0].on_grip and web[0].finger == "thumb_web"
