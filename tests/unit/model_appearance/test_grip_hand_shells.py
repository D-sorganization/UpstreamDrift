"""Palm, dorsum and thumb-web shells of the grip hand mesh (#13997, slice 1).

The fingers alone read as short and straight: the hand needs a palm under
every finger it holds the grip with, a dorsum (metacarpal block) over the
knuckles and a web between the thumb and the index. These tests measure the
shells on the analytic grip cylinder and, per patch, on the real grip taper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import grip_hands as gh
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    MAX_PAD_CLEARANCE_M,
    GripHandMesh,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import GripPose, GripStyle, Hand

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]

RADII_M = (0.0127, 0.0138, 0.0140, 0.0145)
#: Minimum azimuth the hand section covers around the grip (#13997).
MIN_SECTION_COVERAGE_DEG = 270
#: A section is the mesh within this axial half-width of a finger's centre...
SECTION_HALF_WIDTH_M = 0.001
#: ...and within this radial distance of the grip surface.
SECTION_RADIAL_REACH_M = 0.03

_BARY = np.array(
    [(a, b, 1.0 - a - b) for a in np.linspace(0, 1, 9) for b in np.linspace(0, 1, 9)]
)
_BARY = _BARY[_BARY[:, 2] >= -1e-12]


def _surface_points(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    samples = np.einsum("kj,mjd->mkd", _BARY, tri).reshape(-1, 3)
    return np.vstack([mesh.vertices, samples])


def _rho(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0], points[:, 1])


def _parts(result: GripHandMesh, finger: str):
    return [p for p in result.components if p.finger == finger]


def _section_coverage_deg(result: GripHandMesh, z: float) -> int:
    pts = _surface_points(result.mesh)
    near = (np.abs(pts[:, 2] - z) <= SECTION_HALF_WIDTH_M) & (
        _rho(pts) <= result.grip_radius_m + SECTION_RADIAL_REACH_M
    )
    azimuth = np.degrees(np.arctan2(pts[near, 1], pts[near, 0])) % 360.0
    return int(np.unique(azimuth.astype(int)).size)


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("style", list(GripStyle))
@pytest.mark.parametrize("radius", RADII_M)
def test_hand_section_covers_270_deg_around_grip(
    hand: Hand, style: GripStyle, radius: float
):
    result = build_grip_hand_mesh(hand, grip_radius_m=radius, style=style)
    for finger in FINGER_NAMES:
        proximal = next(p for p in result.components if p.name == f"{finger}_proximal")
        if not proximal.on_grip:
            continue  # intentionally off the grip (overlap, interlock)
        z = float(proximal.mesh.vertices[:, 2].mean())
        coverage = _section_coverage_deg(result, z)
        assert coverage >= MIN_SECTION_COVERAGE_DEG, (finger, coverage)


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


def _club(key: str) -> ca.ClubAssembly:
    spec = ROOT / f"docs/development/full_body_models/full_body_spec_anthro_{key}.json"
    club = ca.assembly_from_spec(json.loads(spec.read_text(encoding="utf-8")))
    assert club is not None
    return club


@pytest.mark.parametrize("key", ["driver", "iron7"])
@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("style", list(GripStyle))
def test_each_contact_patch_gap_within_0_to_3_mm(
    key: str, hand: Hand, style: GripStyle
):
    club = _club(key)
    result = gh.baked_grip_hand(club, hand, GripPose(style=style))
    for part in result.components:
        if not part.on_grip:
            continue
        pts = _surface_points(part.mesh)
        depth = result.distance_below_butt_m + pts[:, 2]
        taper = np.array([gh.grip_radius_at(club, float(d)) for d in depth])
        gap = _rho(pts) - taper
        assert gap.min() >= 0.0, (part.name, gap.min())  # no penetration
        assert gap.min() <= MAX_PAD_CLEARANCE_M, (part.name, gap.min())


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
    # The web sits between each hand's thumb and its V.
    for result in (lead, trail):
        web = next(p for p in result.components if p.name == "thumb_web").mesh
        az = np.degrees(np.arctan2(web.vertices[:, 1], web.vertices[:, 0]))
        offsets = _wrapped_deg(az - result.v_azimuth_deg)
        thumb = _wrapped_deg(
            np.array([result.thumb_azimuth_deg - result.v_azimuth_deg])
        )
        assert offsets.min() <= min(0.0, thumb[0]) and offsets.max() >= max(
            0.0, thumb[0]
        )
