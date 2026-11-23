"""The skinned hand pair posed from the spec's own grip (#15654).

Without ``subject.grip_config`` the pair keeps the owner grip. With one, the
pair takes its hand spacing, grip style, strength and lead thumb, and the
wrists :func:`apply_skinned_hand_wrists` places are those of that pair.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.grip_pose import (
    NATIVE_HAND_SPACING_M,
    GripPose,
)
from src.shared.python.model_appearance.hand_spec import spec_grip_pose
from src.shared.python.motion_matching.execution.grip_rotation import (
    apply_grip_config,
)
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
    wrist_joints_in_club_m,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(900)]
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
WIDE_M = 0.10
#: A weak interlocking grip with the hands spread apart: three picks at once.
PICKED = GripSpec.preset("weak", style=GripStyle.INTERLOCK, hand_spacing_m=WIDE_M)


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def club(base: dict[str, Any]) -> ca.ClubAssembly:
    found = ca.assembly_from_spec(base)
    assert found is not None
    return found


@pytest.fixture(scope="module")
def configured(base: dict[str, Any]) -> dict[str, Any]:
    return apply_grip_config(base, PICKED)


def _station_m(result: Any) -> float:
    """Where a posed hand's frame sits along the grip axis (``z``, toward the
    head), in the pair's grip frame."""
    return float(result.frame_from_rest[2, 3])


def _azimuth_deg(points: np.ndarray) -> float:
    """Mean direction of ``points`` about the grip axis (``z``), degrees."""
    centre = points.mean(axis=0)
    return math.degrees(math.atan2(centre[1], centre[0]))


def test_a_spec_without_grip_config_keeps_the_owner_grip(base, club) -> None:
    assert spec_grip_pose(base) is None
    assert sch.owner_grip_spec(grip=None) == sch.owner_grip_spec()
    owner = sch.wrist_joints_m(club)
    for side, wrist in sch.wrist_joints_m(club, grip=None).items():
        np.testing.assert_array_equal(wrist, owner[side])
    drawn = sch.skinned_hand_meshes(club, grip=None)
    for label, mesh in sch.skinned_hand_meshes(club).items():
        np.testing.assert_array_equal(drawn[label].vertices, mesh.vertices)
        np.testing.assert_array_equal(drawn[label].faces, mesh.faces)


def test_the_configured_grip_reaches_the_pair(configured) -> None:
    grip = spec_grip_pose(configured)
    assert grip == PICKED.pose()
    spec = sch.owner_grip_spec(grip=grip)
    assert spec.hand_spacing_m == WIDE_M
    assert spec.style is GripStyle.INTERLOCK
    assert spec.lead.v_azimuth_deg == pytest.approx(grip.lead_v_azimuth_deg)
    assert spec.lead.thumb_azimuth_deg == pytest.approx(grip.lead_thumb_offset_deg)
    # The owner's seat stays: lead below the cap, trail seat and heel pad.
    owner = sch.owner_grip_spec()
    assert spec.lead_below_butt_m == owner.lead_below_butt_m
    assert spec.trail.club_cross_angle_deg == owner.trail.club_cross_angle_deg
    assert spec.lead.pinky_side_palm_depth_m == owner.lead.pinky_side_palm_depth_m


def test_hand_spacing_along_the_grip_matches_the_spec(configured) -> None:
    grip = spec_grip_pose(configured)
    assert grip is not None
    wide = sch.build_seated_pair(grip=grip)
    wider = sch.build_seated_pair(grip=replace(grip, hand_spacing_m=WIDE_M + 0.02))
    # The lead hand stays where it is; the trail hand moves with the spacing
    # (the seat on the tapered grip shifts it by a fraction of a millimetre).
    assert _station_m(wider["lead"]) == _station_m(wide["lead"])
    moved = _station_m(wider["trail"]) - _station_m(wide["trail"])
    assert moved == pytest.approx(0.02, abs=5e-4)
    # The owner pair is settled against the lead fingers near 0.079 m, so the
    # picked 0.10 m puts the trail hand about two centimetres further down.
    owner = sch.build_seated_pair()
    assert _station_m(wide["trail"]) - _station_m(owner["trail"]) > 0.015


def _native(configured: dict[str, Any], **picks: Any) -> GripPose:
    """The configured grip at the native hand spacing, with ``picks`` changed:
    there the trail little finger reaches the lead hand."""
    grip = spec_grip_pose(configured)
    assert grip is not None
    return replace(grip, hand_spacing_m=NATIVE_HAND_SPACING_M, **picks)


def test_the_grip_style_changes_where_the_trail_little_finger_lies(
    configured,
) -> None:
    interlock = sch.build_seated_pair(grip=_native(configured))["trail"]
    overlap = _native(configured, style=GripStyle.OVERLAP)
    overlapped = sch.build_seated_pair(grip=overlap)["trail"]
    assert interlock.rest_asset is not None
    tip = interlock.rest_asset.dominant_vertex_ids(("little_dist",))
    moved = np.linalg.norm(
        interlock.mesh.vertices[tip].mean(axis=0)
        - overlapped.mesh.vertices[tip].mean(axis=0)
    )
    # Overlapping, it lies in the groove on the lead hand: centimetres away.
    assert moved > 0.005


def test_a_weaker_grip_turns_the_lead_hand_with_its_v(configured) -> None:
    weak = _native(configured, style=GripStyle.OVERLAP)
    neutral = replace(weak, lead_v_azimuth_deg=20.0)
    turn = _azimuth_deg(
        sch.build_seated_pair(grip=weak)["lead"].mesh.vertices
    ) - _azimuth_deg(sch.build_seated_pair(grip=neutral)["lead"].mesh.vertices)
    # The lead V is 15 degrees nearer the target, and in the pair's grip frame
    # the lead hand's azimuth runs with its V: it turns about that far back.
    assert -25.0 < turn < -5.0


def test_apply_skinned_hand_wrists_places_the_wrists_of_the_spec_grip(
    configured, club
) -> None:
    moved = apply_skinned_hand_wrists(configured)
    target = sch.wrist_joints_m(club, grip=spec_grip_pose(configured))
    placed = wrist_joints_in_club_m(moved)
    for side in ("lead", "trail"):
        np.testing.assert_allclose(placed[side], target[side], atol=1e-9)
    owner = sch.wrist_joints_m(club)
    assert float(np.linalg.norm(target["trail"] - owner["trail"])) > 0.005


def test_an_invalid_grip_is_refused(configured, club) -> None:
    broken = json.loads(json.dumps(configured))
    broken["subject"]["grip_config"]["style"] = "baseball_bat"
    with pytest.raises(ValueError):
        spec_grip_pose(broken)
    with pytest.raises(TypeError):
        sch.owner_grip_spec(grip="interlock")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sch.wrist_joints_m(club, grip=PICKED)  # type: ignore[arg-type]
    # A trail V turned past 90 degrees: no hand the skinned pair can pose.
    too_strong = GripPose(lead_v_azimuth_deg=85.0, trail_v_offset_deg=10.0)
    with pytest.raises(ValueError):
        sch.owner_grip_spec(grip=too_strong)
