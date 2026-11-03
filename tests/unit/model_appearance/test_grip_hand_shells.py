"""Dorsum shell of the grip hand mesh (#13997, slice 1 of 3).

The fingers alone read as short and straight: the hand needs a dorsum
(metacarpal block) over the knuckles. Later slices add the thumb web and a
palm strip under every topology-placed finger. These tests measure the shells
on the analytic grip cylinder (the hand-grip frame's ``z`` axis).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.grip_hand_mesh import (
    MAX_PAD_CLEARANCE_M,
    GripHandMesh,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import GripPose, GripStyle, Hand

pytestmark = pytest.mark.unit

RADII_M = (0.0127, 0.0138, 0.0140, 0.0145)


def _rho(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[:, 0], points[:, 1])


def _parts(result: GripHandMesh, finger: str):
    return [p for p in result.components if p.finger == finger]


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
