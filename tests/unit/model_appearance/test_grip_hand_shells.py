"""#13997: dorsum, thumb-web, 270° shell wrap, tapered-grip clearance."""

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
MIN_SECTION_COVERAGE_DEG = 270
SECTION_HALF_WIDTH_M = 0.001
SECTION_RADIAL_REACH_M = 0.03
# Radial thickness of the on-grip patch sampled from each part's closest approach (not
# the 0–3 mm acceptance cap).
_CONTACT_PATCH_SAMPLE_BAND_M = 0.001

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


def _finger_section_surface_point_sets(
    result: GripHandMesh, z_lo: float, z_hi: float
) -> list[np.ndarray]:
    """Surface samples per finger phalanx (excludes palm, thumb, thumb_web)."""
    point_sets: list[np.ndarray] = []
    reach = result.grip_radius_m + SECTION_RADIAL_REACH_M
    for finger in FINGER_NAMES:
        parts = [p for p in result.components if p.finger == finger]
        if not parts:
            continue
        pts = np.vstack([_surface_points(p.mesh) for p in parts])
        pick = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi) & (_rho(pts) <= reach)
        if np.any(pick):
            point_sets.append(pts[pick])
    return point_sets


def _finger_arc_intervals_deg(
    point_sets: list[np.ndarray],
) -> list[tuple[float, float]]:
    """Azimuth interval on the grip-cylinder section contributed by each finger."""
    intervals: list[tuple[float, float]] = []
    for pts in point_sets:
        if pts.shape[0] < 2:
            continue
        azimuth = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
        intervals.append((float(azimuth.min()), float(azimuth.max())))
    return intervals


def _merged_covered_arc_deg(intervals: list[tuple[float, float]]) -> float:
    """Union of finger arc intervals around the grip, in degrees."""
    if not intervals:
        return 0.0
    segments: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if lo <= hi:
            segments.append((lo, hi))
        else:
            segments.append((lo, 360.0))
            segments.append((0.0, hi))
    segments.sort(key=lambda pair: pair[0])
    merged: list[tuple[float, float]] = []
    for lo, hi in segments:
        if not merged or lo > merged[-1][1] + 1e-9:
            merged.append((lo, hi))
        else:
            prev_lo, prev_hi = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi))
    covered = sum(hi - lo for lo, hi in merged)
    if len(merged) > 1 and merged[0][0] < 1e-9 and merged[-1][1] > 360.0 - 1e-9:
        wrapped = 360.0 - (merged[0][1] - merged[-1][0])
        covered = max(covered, wrapped)
    return float(covered)


def _assert_covered_arc_at_least_deg(covered_deg: float, minimum_deg: float) -> None:
    assert covered_deg >= minimum_deg - 1e-6, (covered_deg, minimum_deg)


def _taper_slack_m(club: ca.ClubAssembly, distance_below_butt_m: np.ndarray) -> float:
    """Grip-radius change (m) across the axial span of one contact patch."""
    clipped = np.clip(distance_below_butt_m, 0.0, club.grip_length_m)
    radii = np.array([gh.grip_radius_at(club, float(d)) for d in clipped])
    return float(radii.max() - radii.min())


def _contact_patch_samples(gap: np.ndarray) -> np.ndarray:
    """Points on the contact patch from closest approach, without the 0–3 mm cut."""
    min_gap = float(gap.min())
    upper = min_gap + _CONTACT_PATCH_SAMPLE_BAND_M
    return gap[gap <= upper + 1e-12]


def _assert_patch_clearance_within_limit(
    patch_gap: np.ndarray, taper_slack: float, *, part_name: str
) -> None:
    assert patch_gap.size, part_name
    assert patch_gap.min() >= -1e-6, (part_name, patch_gap.min(), patch_gap.max())
    limit = MAX_PAD_CLEARANCE_M + taper_slack + 1e-6
    assert patch_gap.max() <= limit, (
        part_name,
        patch_gap.min(),
        patch_gap.max(),
        taper_slack,
        limit,
    )


def _club(key: str) -> ca.ClubAssembly:
    spec = ROOT / f"docs/development/full_body_models/full_body_spec_anthro_{key}.json"
    club = ca.assembly_from_spec(json.loads(spec.read_text(encoding="utf-8")))
    assert club is not None
    return club


@pytest.mark.parametrize("hand", list(Hand))
@pytest.mark.parametrize("style", list(GripStyle))
@pytest.mark.parametrize("radius", RADII_M)
def test_hand_section_covers_270_deg_around_grip(
    hand: Hand, style: GripStyle, radius: float
):
    result = build_grip_hand_mesh(hand, grip_radius_m=radius, style=style)
    proximal_z = [
        float(
            next(p for p in result.components if p.name == f"{finger}_proximal")
            .mesh.vertices[:, 2]
            .mean()
        )
        for finger in FINGER_NAMES
    ]
    z_lo = min(proximal_z) - SECTION_HALF_WIDTH_M
    z_hi = max(proximal_z) + SECTION_HALF_WIDTH_M
    point_sets = _finger_section_surface_point_sets(result, z_lo, z_hi)
    assert point_sets, (hand, style, radius)
    intervals = _finger_arc_intervals_deg(point_sets)
    covered = _merged_covered_arc_deg(intervals)
    _assert_covered_arc_at_least_deg(covered, MIN_SECTION_COVERAGE_DEG)


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
        taper_slack = _taper_slack_m(club, depth)
        patch_gap = _contact_patch_samples(gap)
        assert patch_gap.size, part.name
        _assert_patch_clearance_within_limit(
            patch_gap, taper_slack, part_name=part.name
        )


def test_contact_patch_clearance_rejects_over_limit_sample() -> None:
    taper_slack = 0.0001
    ok = np.array([0.0005, 0.0012, 0.0018])
    _assert_patch_clearance_within_limit(ok, taper_slack, part_name="synthetic_ok")
    bad = np.array([0.0005, 0.0012, 0.0036])
    with pytest.raises(AssertionError):
        _assert_patch_clearance_within_limit(
            bad, taper_slack, part_name="synthetic_bad"
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
