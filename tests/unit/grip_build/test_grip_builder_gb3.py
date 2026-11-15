"""HRG-GB3 palm wrap, thumb press, trail link, and golden checksums (#15125)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from src.shared.python.grip_build.builder import build_grip, thumb_radial_azimuth_deg
from src.shared.python.grip_build.goldens import (
    GOLDEN_RECEIPT_PATH,
    golden_case_keys,
    load_golden_receipt,
)
from src.shared.python.grip_build.receipt import BuildStatus
from src.shared.python.grip_contact.contact_topology import GripStyle, contact_topology
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.hand_size import HAND_LENGTH_RANGE_M
from src.shared.python.motion_matching.grip_contract import GRIP_STRENGTH_PRESETS_DEG

_THUMB_AZ_TOL_DEG = 2.5
_IP_STRAIGHT_TOL_RAD = math.radians(12.0)
_LIFELINE_COVER_TOL_M = 0.006


def _hand_sizes() -> tuple[float, float, float]:
    lo, hi = HAND_LENGTH_RANGE_M
    return (lo, (lo + hi) * 0.5, hi)


def _strength_spec(strength: str, **kwargs) -> GripSpec:
    delta = GRIP_STRENGTH_PRESETS_DEG[strength]
    lead = replace(HandGripSpec(), v_azimuth_deg=HandGripSpec().v_azimuth_deg + delta)
    trail = replace(
        HandGripSpec(
            v_azimuth_deg=HandGripSpec().v_azimuth_deg + delta,
            thumb_azimuth_deg=45.0,
        ),
    )
    base = replace(GripSpec.neutral(), lead=lead, trail=trail)
    return replace(base, **kwargs) if kwargs else base


@pytest.mark.unit
def test_lead_thumb_azimuth_22_5_away_from_target_within_tol() -> None:
    spec = GripSpec.neutral()
    result = build_grip(spec)
    assert result.status == BuildStatus.OK
    az = thumb_radial_azimuth_deg(result, role="lead")
    target = spec.resolved_thumb_azimuth_deg("lead")
    assert az == pytest.approx(target, abs=_THUMB_AZ_TOL_DEG)


@pytest.mark.unit
def test_trail_thumb_azimuth_45_toward_target() -> None:
    spec = GripSpec.neutral()
    result = build_grip(spec)
    az = thumb_radial_azimuth_deg(result, role="trail")
    target = spec.resolved_thumb_azimuth_deg("trail")
    assert az == pytest.approx(target, abs=_THUMB_AZ_TOL_DEG)


@pytest.mark.unit
def test_thumb_ip_near_straight() -> None:
    spec = GripSpec.neutral()
    result = build_grip(spec)
    for role in ("lead", "trail"):
        ip = result.joint_angles_rad[role]["thumb_ip"]
        assert ip <= _IP_STRAIGHT_TOL_RAD


@pytest.mark.unit
def test_overlap_trail_pinky_on_lead_index_middle_gap() -> None:
    spec = replace(GripSpec.neutral(), style=GripStyle.OVERLAP)
    result = build_grip(spec)
    assert "overlap_little_finger_lead_index" in result.active_contact_pairs
    gap = result.contact_clearances_m["overlap_little_finger_lead_index"]
    assert gap >= -result.receipt.pad_tolerance_m


@pytest.mark.unit
def test_interlock_trail_pinky_between_lead_index_middle_no_penetration() -> None:
    spec = replace(GripSpec.neutral(), style=GripStyle.INTERLOCK)
    result = build_grip(spec)
    for name in (
        "interlock_little_finger_lead_middle",
        "interlock_lead_index_wrap",
    ):
        assert name in result.active_contact_pairs
        assert result.contact_clearances_m[name] >= -result.receipt.bone_tolerance_m


@pytest.mark.unit
def test_interlock_infeasible_small_hand_reports_status() -> None:
    spec = replace(
        GripSpec.neutral(),
        style=GripStyle.INTERLOCK,
        hand_size=HandSize(length_m=HAND_LENGTH_RANGE_M[0]),
    )
    result = build_grip(spec)
    assert result.status in (
        BuildStatus.INFEASIBLE,
        BuildStatus.JOINT_LIMIT,
        BuildStatus.GAP,
    )


@pytest.mark.unit
def test_trail_lifeline_covers_lead_thumb() -> None:
    spec = GripSpec.neutral()
    result = build_grip(spec)
    gap = result.contact_clearances_m.get("trail_palm_lead_thumb")
    assert gap is not None
    assert gap >= -_LIFELINE_COVER_TOL_M


@pytest.mark.unit
def test_contacts_map_to_topology_pair_names() -> None:
    spec = GripSpec.neutral()
    result = build_grip(spec)
    from src.shared.python.motion_matching.grip_contract import (
        GripSpec as TopologyGripSpec,
    )

    topo = contact_topology(
        TopologyGripSpec(
            style=spec.style,
            trigger_finger=spec.trigger_finger,
            handedness=spec.handedness,
        )
    )
    for name in result.active_contact_pairs:
        topo.pair(name)


@pytest.mark.unit
def test_goldens_match_checksums() -> None:
    receipt = load_golden_receipt()
    for case in receipt["cases"]:
        key = case["key"]
        spec = _spec_for_golden_key(key)
        result = build_grip(spec)
        joints_json = json.dumps(
            result.joint_angles_rad, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(joints_json.encode()).hexdigest()
        assert digest == case["joint_sha256"], key
        contact_json = json.dumps(
            sorted(result.active_contact_pairs),
            separators=(",", ":"),
        )
        contact_digest = hashlib.sha256(contact_json.encode()).hexdigest()
        assert contact_digest == case["contact_sha256"], key


@pytest.mark.unit
@pytest.mark.parametrize("style", list(GripStyle))
def test_neutral_grip_ok_reference_hand(style: GripStyle) -> None:
    spec = replace(GripSpec.neutral(), style=style)
    result = build_grip(spec)
    assert result.status == BuildStatus.OK


def _spec_for_golden_key(key: str) -> GripSpec:
    """Parse golden table keys: ``{strength}_{style}_{handedness}_{size}``."""
    parts = key.split("_")
    size_name = parts[-1]
    handedness = parts[-2]
    style_name = "_".join(parts[1:-2])
    strength = parts[0]
    style = GripStyle(style_name)
    sizes = {
        "small": _hand_sizes()[0],
        "medium": _hand_sizes()[1],
        "large": _hand_sizes()[2],
    }
    spec = _strength_spec(strength, handedness=handedness, style=style)  # type: ignore[arg-type]
    return replace(spec, hand_size=HandSize(length_m=sizes[size_name]))


@pytest.mark.unit
def test_golden_receipt_committed() -> None:
    assert GOLDEN_RECEIPT_PATH.is_file()
    keys = golden_case_keys()
    assert len(keys) == 54
