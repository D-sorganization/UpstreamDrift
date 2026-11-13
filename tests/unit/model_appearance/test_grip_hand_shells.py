"""#13997: dorsum, thumb-web, 270° shell wrap, tapered-grip clearance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import grip_hands as gh
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    FINGER_NAMES,
    MAX_PAD_CLEARANCE_M,
    GripHandMesh,
    HandComponent,
    HandMeshParams,
    build_grip_hand_mesh,
)
from src.shared.python.model_appearance.grip_pose import GripPose, GripStyle, Hand

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
RADII_M = (0.0127, 0.0138, 0.0140, 0.0145)
MIN_SECTION_COVERAGE_DEG = 270
MAX_UNCOVERED_AZIMUTH_GAP_DEG = 360.0 - MIN_SECTION_COVERAGE_DEG
SECTION_HALF_WIDTH_M = 0.001
SECTION_RADIAL_REACH_M = 0.03
# Extra clearance while baking on one radius then placing on a tapered club (grip_hands).
_BAKE_TAPER_SLACK_M = 0.0003

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


def _finger_mcp_azimuth_rad(result: GripHandMesh, finger: str, z: float) -> float:
    proximal = next(p for p in result.components if p.name == f"{finger}_proximal")
    pts = proximal.mesh.vertices
    band = pts[np.abs(pts[:, 2] - z) <= SECTION_HALF_WIDTH_M]
    if band.size == 0:
        band = pts
    return math.atan2(float(np.mean(band[:, 1])), float(np.mean(band[:, 0])))


def _in_curl_sector(
    azimuth_rad: np.ndarray, lo: float, hi: float, curl: int
) -> np.ndarray:
    """Whether each azimuth lies in the sector from ``lo`` to ``hi`` in curl direction."""
    two_pi = 2.0 * math.pi
    az = np.mod(azimuth_rad, two_pi)
    lo_m = lo % two_pi
    hi_m = hi % two_pi
    if curl < 0:
        if lo_m >= hi_m:
            return (az <= hi_m) | (az >= lo_m)
        return (az >= lo_m) & (az <= hi_m)
    if hi_m >= lo_m:
        return (az >= lo_m) & (az <= hi_m)
    return (az >= lo_m) | (az <= hi_m)


def _finger_section_surface_points(
    result: GripHandMesh, finger: str, z: float
) -> list[np.ndarray]:
    """Surface samples for one finger's MCP section (no thumb or foreign palm geometry)."""
    wrap_half = math.radians(MIN_SECTION_COVERAGE_DEG / 2.0)
    mcp_az = _finger_mcp_azimuth_rad(result, finger, z)
    curl = result.curl_sign
    lo = mcp_az - curl * wrap_half
    hi = mcp_az + curl * wrap_half
    point_sets: list[np.ndarray] = []
    for part in result.components:
        if not part.on_grip:
            continue
        if part.finger == finger or part.name == f"palm_{finger}":
            point_sets.append(_surface_points(part.mesh))
            continue
        if part.name != "palm" or part.finger != "palm":
            continue
        pts = _surface_points(part.mesh)
        band = np.abs(pts[:, 2] - z) <= SECTION_HALF_WIDTH_M
        radial = _rho(pts) <= result.grip_radius_m + SECTION_RADIAL_REACH_M
        sector = _in_curl_sector(np.arctan2(pts[:, 1], pts[:, 0]), lo, hi, curl)
        pick = band & radial & sector
        if np.any(pick):
            point_sets.append(pts[pick])
    return point_sets


def _max_uncovered_azimuth_gap_deg(
    point_sets: list[np.ndarray], z: float, *, grip_radius_m: float
) -> float:
    """Largest angular gap on the grip-cylinder slice from the given surface samples."""
    if not point_sets:
        return 360.0
    pts = np.vstack(point_sets)
    near = (np.abs(pts[:, 2] - z) <= SECTION_HALF_WIDTH_M) & (
        _rho(pts) <= grip_radius_m + SECTION_RADIAL_REACH_M
    )
    slice_pts = pts[near]
    if slice_pts.size == 0:
        return 360.0
    azimuth = np.degrees(np.arctan2(slice_pts[:, 1], slice_pts[:, 0])) % 360.0
    if azimuth.size == 1:
        return 360.0
    sorted_az = np.sort(azimuth)
    between = np.diff(sorted_az)
    wrap = (360.0 - sorted_az[-1]) + sorted_az[0]
    return float(max(np.max(between), wrap))


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
            continue
        z = float(proximal.mesh.vertices[:, 2].mean())
        point_sets = _finger_section_surface_points(result, finger, z)
        assert point_sets, (finger, z)
        gap_deg = _max_uncovered_azimuth_gap_deg(
            point_sets, z, grip_radius_m=result.grip_radius_m
        )
        assert gap_deg <= MAX_UNCOVERED_AZIMUTH_GAP_DEG + 1e-6, (finger, gap_deg)


def _part_nominal_clearance_m(part: HandComponent, params: HandMeshParams) -> float:
    """Baker nominal pad clearance (m) for patch membership, not acceptance limits."""
    if part.finger in {"palm", "thumb_web"} or part.name.startswith("palm_"):
        return params.palm_clearance_m
    return params.pad_clearance_m


def _taper_slack_m(club: ca.ClubAssembly, distance_below_butt_m: np.ndarray) -> float:
    """Grip-radius change (m) across the axial span of one contact patch."""
    clipped = np.clip(distance_below_butt_m, 0.0, club.grip_length_m)
    radii = np.array([gh.grip_radius_at(club, float(d)) for d in clipped])
    return float(radii.max() - radii.min())


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
    params = DEFAULT_HAND_MESH_PARAMS
    result = gh.baked_grip_hand(club, hand, GripPose(style=style))
    for part in result.components:
        if not part.on_grip:
            continue
        pts = _surface_points(part.mesh)
        depth = result.distance_below_butt_m + pts[:, 2]
        taper = np.array([gh.grip_radius_at(club, float(d)) for d in depth])
        gap = _rho(pts) - taper
        taper_slack = _taper_slack_m(club, depth)
        nominal = _part_nominal_clearance_m(part, params)
        # Touching patch from baker nominal clearance + taper slack only (not 0–3 mm).
        on_patch = gap <= nominal + _BAKE_TAPER_SLACK_M + taper_slack + 1e-6
        assert on_patch.any(), part.name
        patch_gap = gap[on_patch]
        assert patch_gap.min() >= -1e-6, (part.name, patch_gap.min(), patch_gap.max())
        assert patch_gap.max() <= MAX_PAD_CLEARANCE_M + taper_slack + 1e-6, (
            part.name,
            patch_gap.min(),
            patch_gap.max(),
            taper_slack,
        )


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
