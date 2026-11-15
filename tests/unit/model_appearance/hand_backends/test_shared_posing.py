"""Shared hand poser: flat placement, contact, styles, roll and mirroring."""

from __future__ import annotations

import math
import sys
import types
from dataclasses import replace

import pytest

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.grip_geometry import get_grip_preset
from src.shared.python.grip_contact.pad_layout import GRIP_RADIUS_M
from src.shared.python.grip_spec.types import GripSpec, HandSize
from src.shared.python.hand_size import BREADTH_OVER_LENGTH_UNSPECIFIED
from src.shared.python.model_appearance.grip_hand_mesh import FINGER_NAMES
from src.shared.python.model_appearance.hand_backends import (
    CONTACT_QUALITY_METRIC_KEYS,
    DEFAULT_SETTINGS_PATH,
    GB2_BRANCH,
    JointDescription,
    KinematicChain,
    build_default_chain,
    gb2_curl_status,
    load_pose_settings,
    metrics_from_tables,
    pose_chain,
    pose_grip,
    settings_from_dict,
)
from src.shared.python.model_appearance.hand_backends.grip_solid import GripSolid
from src.shared.python.model_appearance.hand_backends.solve import (
    segment_azimuth_rad,
    surface_gap_m,
)

pytestmark = pytest.mark.unit

_PHALANX = ("proximal", "middle", "distal")


def _settings():
    return load_pose_settings()


def test_settings_file_is_the_contact_contract() -> None:
    settings = load_pose_settings()
    assert DEFAULT_SETTINGS_PATH.is_file()
    assert settings.contact_gap_tolerance_m == pytest.approx(0.002)
    assert settings.penetration_tolerance_m == pytest.approx(0.0005)
    assert settings.base_clearance_m + settings.just_below_pinky_m <= (
        settings.contact_gap_tolerance_m + 1e-12
    )


def test_settings_reject_a_pinky_gap_outside_the_band() -> None:
    import json

    payload = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    payload["just_below_pinky_m"] = 0.01
    with pytest.raises(ValueError, match="just_below"):
        settings_from_dict(payload)


def test_flat_hand_crosses_the_shaft_and_does_not_use_the_full_palm_diagonal() -> None:
    settings = _settings()
    pose = pose_grip(GripSpec.legacy(), settings=settings)
    flat = pose.lead.flat
    assert flat is not None
    radius = pose.lead.segment("index", "proximal").radius_m
    assert flat.index_axis_distance_m == pytest.approx(
        GRIP_RADIUS_M + radius, abs=0.001
    )
    assert flat.index_to_shaft_deg > 60.0
    assert flat.pinky_gap_m == pytest.approx(
        settings.base_clearance_m + settings.just_below_pinky_m
    )
    assert flat.pinky_gap_m > flat.index_gap_m
    span = abs(math.degrees(flat.pinky_azimuth_rad - flat.index_azimuth_rad))
    assert 0.5 < span < 15.0


def test_a_larger_club_cross_opens_the_knuckle_azimuth() -> None:
    narrow_span = _knuckle_span(pose_grip(_with_cross(10.0)))
    wide_span = _knuckle_span(pose_grip(_with_cross(60.0)))
    assert wide_span > narrow_span * 2.0


def test_default_overlap_contacts_without_penetration() -> None:
    settings = _settings()
    pose = pose_grip(GripSpec.legacy(), settings=settings)
    assert pose.metrics["schema"] == "hand_contact_quality_v1"
    assert set(CONTACT_QUALITY_METRIC_KEYS) <= set(pose.metrics)
    assert pose.metrics["contact_gap_tolerance_m"] == settings.contact_gap_tolerance_m
    assert pose.metrics["within_tolerance"] is True
    for finger in FINGER_NAMES:
        _assert_on_grip(pose.lead, finger, settings)
        if finger != "little":
            _assert_on_grip(pose.trail, finger, settings)
    little = pose.trail.segment("little", "proximal")
    assert little.neighbour_gap_m is not None
    assert little.neighbour_gap_m <= settings.contact_gap_tolerance_m
    assert little.grip_gap_m > settings.contact_gap_tolerance_m
    for hand in (pose.lead, pose.trail):
        for segment in hand.segments:
            assert segment.penetration_m <= settings.penetration_tolerance_m
    _assert_neighbours_do_not_penetrate(pose.lead, settings)
    assert pose.metrics["wrap_angle_rad"]["lead_index"] >= settings.min_wrap_rad


def test_arch_keeps_the_flat_cross_and_cups_the_little_more_than_the_index() -> None:
    settings = _settings()
    posed = pose_grip(GripSpec.legacy(), settings=settings)
    flat = posed.lead.flat
    assert flat is not None
    little = posed.lead.segment("little", "proximal")
    assert little.grip_gap_m < flat.pinky_gap_m
    assert posed.lead.angle("lead_little_metacarpal") > posed.lead.angle(
        "lead_index_metacarpal"
    )
    opened = pose_grip(_at_depth(0.0), settings=settings)
    closed = pose_grip(_at_depth(1.0), settings=settings)
    assert opened.lead.segment("little", "proximal").grip_gap_m > (
        opened.lead.segment("index", "proximal").grip_gap_m
    )
    assert closed.lead.palm_contact_area_m2 > opened.lead.palm_contact_area_m2
    assert opened.lead.palm_contact_area_m2 == 0.0
    assert closed.lead.angle("lead_little_metacarpal") > opened.lead.angle(
        "lead_little_metacarpal"
    )


def test_ten_finger_interlock_and_overlap_are_different_contacts() -> None:
    settings = _settings()
    ten = pose_grip(GripSpec.legacy(style=GripStyle.TEN_FINGER), settings=settings)
    interlock = pose_grip(GripSpec.legacy(style=GripStyle.INTERLOCK), settings=settings)
    overlap = pose_grip(GripSpec.legacy(style=GripStyle.OVERLAP), settings=settings)
    ten_little = ten.trail.segment("little", "proximal")
    inter_little = interlock.trail.segment("little", "proximal")
    over_little = overlap.trail.segment("little", "proximal")
    assert ten.metrics["within_tolerance"] is True
    assert interlock.metrics["within_tolerance"] is True
    assert overlap.metrics["within_tolerance"] is True
    assert ten_little.grip_gap_m <= settings.contact_gap_tolerance_m
    assert ten_little.neighbour_gap_m is not None
    assert ten_little.neighbour_gap_m > settings.style_separation_m
    assert inter_little.grip_gap_m <= settings.contact_gap_tolerance_m
    assert inter_little.neighbour_gap_m is not None
    assert inter_little.neighbour_gap_m <= settings.contact_gap_tolerance_m
    assert over_little.neighbour_gap_m is not None
    assert over_little.neighbour_gap_m <= settings.contact_gap_tolerance_m
    assert over_little.grip_gap_m > settings.contact_gap_tolerance_m
    assert (
        over_little.grip_gap_m > inter_little.grip_gap_m + settings.style_separation_m
    )


def test_shaft_roll_preserves_gaps_and_shifts_azimuth() -> None:
    spec = GripSpec.legacy()
    base = pose_grip(spec, shaft_roll_deg=0.0)
    plus = pose_grip(spec, shaft_roll_deg=20.0)
    minus = pose_grip(spec, shaft_roll_deg=-20.0)
    for name, angle in base.lead.angles_rad.items():
        assert plus.lead.angle(name) == pytest.approx(angle)
        assert minus.lead.angle(name) == pytest.approx(angle)
    for finger in FINGER_NAMES:
        for segment in _PHALANX:
            assert plus.lead.segment(finger, segment).grip_gap_m == pytest.approx(
                base.lead.segment(finger, segment).grip_gap_m
            )
    assert _azimuth_delta(base, plus) == pytest.approx(math.radians(20.0), abs=1e-6)
    assert _azimuth_delta(base, minus) == pytest.approx(math.radians(-20.0), abs=1e-6)


def test_left_handed_pose_mirrors_and_flips_shaft_roll() -> None:
    right = pose_grip(GripSpec.legacy())
    left = pose_grip(GripSpec.legacy(handedness="left"))
    for hand_name in ("lead", "trail"):
        right_hand = getattr(right, hand_name)
        left_hand = getattr(left, hand_name)
        for name, angle in right_hand.angles_rad.items():
            assert left_hand.angle(name) == pytest.approx(angle)
        for right_seg, left_seg in zip(
            right_hand.segments, left_hand.segments, strict=True
        ):
            assert left_seg.p0[0] == pytest.approx(right_seg.p0[0])
            assert left_seg.p0[1] == pytest.approx(-right_seg.p0[1])
            assert left_seg.p0[2] == pytest.approx(right_seg.p0[2])
            assert left_seg.p1[1] == pytest.approx(-right_seg.p1[1])
            assert left_seg.grip_gap_m == pytest.approx(right_seg.grip_gap_m)
    base = GripSpec.legacy()
    rolled = replace(
        base,
        handedness="left",
        lead=replace(base.lead, shaft_roll_deg=20.0),
    )
    assert pose_grip(rolled).lead.shaft_roll_deg == pytest.approx(-20.0)
    forced = pose_grip(rolled, shaft_roll_deg=20.0)
    assert forced.lead.shaft_roll_deg == pytest.approx(20.0)
    assert forced.trail.shaft_roll_deg == pytest.approx(20.0)


def test_thumb_stays_on_the_grip_as_it_curls() -> None:
    settings = _settings()
    spec = GripSpec.legacy()
    straight = pose_grip(spec, settings=settings)
    curled = pose_grip(
        replace(spec, lead=replace(spec.lead, thumb_straightness=0.0)),
        settings=settings,
    )
    assert straight.lead.angle("lead_thumb_proximal") == pytest.approx(0.0)
    assert curled.lead.angle("lead_thumb_proximal") > straight.lead.angle(
        "lead_thumb_proximal"
    )
    for pose in (straight, curled):
        thumb = pose.lead.segment("thumb", "proximal")
        assert thumb.grip_gap_m <= settings.contact_gap_tolerance_m
        assert thumb.penetration_m <= settings.penetration_tolerance_m
    thumb = straight.lead.segment("thumb", "proximal")
    azimuth = math.degrees(segment_azimuth_rad(thumb.p0, thumb.p1))
    assert azimuth == pytest.approx(spec.resolved_thumb_azimuth_deg("lead"))


def test_tapered_mens_grip_stays_inside_the_contact_band() -> None:
    geometry = get_grip_preset("mens_standard")
    grip = GripSolid.from_geometry(geometry)
    for station in (0.05, 0.08, 0.12):
        assert grip.radius_at(station) == pytest.approx(geometry.radius_at(station))
    pose = pose_grip(GripSpec.legacy(), grip)
    assert pose.metrics["within_tolerance"] is True


def test_longer_hand_scales_the_phalanx() -> None:
    length = 0.22
    spec = replace(
        GripSpec.legacy(),
        hand_size=HandSize(
            length_m=length,
            breadth_m=length * BREADTH_OVER_LENGTH_UNSPECIFIED,
        ),
    )
    short = build_default_chain(GripSpec.legacy(), "lead")
    long = build_default_chain(spec, "lead")
    assert (
        long.joint("index", "proximal").length_m
        > short.joint("index", "proximal").length_m
    )


def test_clearance_comes_from_settings() -> None:
    settings = _settings()
    posed = pose_grip(GripSpec.legacy(), settings=settings)
    assert posed.lead.flat is not None
    assert posed.lead.flat.index_gap_m == pytest.approx(settings.base_clearance_m)
    tighter = replace(settings, base_clearance_m=0.0002)
    moved = pose_grip(GripSpec.legacy(), settings=tighter)
    assert moved.lead.flat is not None
    assert moved.lead.flat.index_gap_m == pytest.approx(0.0002)


def test_hinge_chain_bisects_to_contact_and_rejects_a_non_shaft_axis() -> None:
    settings = _settings()
    grip = GripSolid.cylinder(GRIP_RADIUS_M)
    pose = pose_chain(
        _hinge_chain(), GripSpec.legacy(), grip, side="lead", settings=settings
    )
    segment = pose.segment("index", "proximal")
    assert -settings.penetration_tolerance_m <= segment.grip_gap_m
    assert segment.grip_gap_m <= settings.contact_gap_tolerance_m
    assert 0.5 < pose.angle("synthetic_proximal") < math.pi
    with pytest.raises(ValueError, match="shaft"):
        pose_chain(
            _hinge_chain((0.0, 1.0, 0.0)),
            GripSpec.legacy(),
            grip,
            side="lead",
            settings=settings,
        )


def test_contracts_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        GripSolid.cylinder(0.0)
    with pytest.raises(TypeError):
        pose_grip("not a spec")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_default_chain(GripSpec.legacy(), "both")
    with pytest.raises(ValueError):
        KinematicChain("empty", "lead", -1, ())


def test_trigger_finger_is_not_silently_applied() -> None:
    pose = pose_grip(GripSpec.legacy(trigger_finger=True))
    assert pose.trigger_finger_applied is False
    assert pose.coupling_overrides_applied is False


def test_gb2_is_unavailable_and_does_not_replace_contact() -> None:
    status = gb2_curl_status(GripSpec.legacy())
    assert status["available"] is False
    assert status["applied"] is False
    assert "15134" in status["note"]
    assert status["branch"] == GB2_BRANCH
    plain = pose_grip(GripSpec.legacy())
    module = types.ModuleType("src.shared.python.grip_build")
    module.BUILDER_VERSION = "gb2-test"

    class _Result:
        joint_angles_rad = {"index": {"mcp": 0.2, "pip": 0.3, "dip": 0.4}}

    module.build_lead_fingers = lambda spec, geometry=None: _Result()
    sys.modules["src.shared.python.grip_build"] = module
    try:
        recorded = pose_grip(GripSpec.legacy())
    finally:
        sys.modules.pop("src.shared.python.grip_build", None)
    assert recorded.gb2["available"] is True
    assert recorded.gb2["applied"] is False
    assert recorded.gb2["angles_rad"]["index"]["pip"] == pytest.approx(0.3)
    for name, angle in plain.lead.angles_rad.items():
        assert recorded.lead.angle(name) == pytest.approx(angle)


def test_backend_metric_table_uses_the_same_schema() -> None:
    settings = _settings()
    good = metrics_from_tables(
        gaps_m={"capsule": 0.0004},
        penetrations_m={"capsule": 0.0},
        wrap_rad={"index": 1.2},
        palm_area_m2={"lead": 0.0},
        settings=settings,
    )
    bad = metrics_from_tables(
        gaps_m={"capsule": 0.01},
        penetrations_m={"capsule": 0.0},
        wrap_rad={"index": 0.0},
        palm_area_m2={"lead": 0.0},
        settings=settings,
    )
    assert good["schema"] == "hand_contact_quality_v1"
    assert good["within_tolerance"] is True
    assert bad["within_tolerance"] is False
    with pytest.raises(ValueError):
        metrics_from_tables(
            gaps_m={"capsule": 0.0},
            penetrations_m={},
            wrap_rad={},
            palm_area_m2={},
            settings=settings,
        )


def _assert_on_grip(hand, finger: str, settings) -> None:
    for segment in ("middle", "distal"):
        item = hand.segment(finger, segment)
        assert item.grip_gap_m <= settings.contact_gap_tolerance_m
        assert item.grip_gap_m >= -settings.penetration_tolerance_m
        assert item.penetration_m <= settings.penetration_tolerance_m


def _assert_neighbours_do_not_penetrate(hand, settings) -> None:
    fingers = list(FINGER_NAMES)
    for left, right in zip(fingers, fingers[1:], strict=False):
        a = hand.segment(left, "proximal")
        b = hand.segment(right, "proximal")
        gap = surface_gap_m(
            a.p0,
            a.p1,
            a.radius_m,
            b.p0,
            b.p1,
            b.radius_m,
            _settings().axis_samples,
        )
        assert gap >= -settings.penetration_tolerance_m


def _knuckle_span(pose) -> float:
    flat = pose.lead.flat
    assert flat is not None
    return abs(flat.pinky_azimuth_rad - flat.index_azimuth_rad)


def _azimuth_delta(base, rolled) -> float:
    before = base.lead.segment("index", "proximal")
    after = rolled.lead.segment("index", "proximal")
    start = segment_azimuth_rad(before.p0, before.p1)
    end = segment_azimuth_rad(after.p0, after.p1)
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _with_cross(angle_deg: float) -> GripSpec:
    spec = GripSpec.legacy()
    return replace(
        spec,
        lead=replace(spec.lead, club_cross_angle_deg=angle_deg),
        trail=replace(spec.trail, club_cross_angle_deg=angle_deg),
    )


def _at_depth(depth: float) -> GripSpec:
    spec = GripSpec.legacy()
    return replace(
        spec,
        lead=replace(spec.lead, palm_finger_depth=depth),
        trail=replace(spec.trail, palm_finger_depth=depth),
    )


def _hinge_chain(axis: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> KinematicChain:
    joint = JointDescription(
        name="synthetic_proximal",
        parent=None,
        finger="index",
        segment="proximal",
        drive="hinge",
        origin_m=(0.0307, 0.0, 0.05),
        axis=axis,
        rest_direction=(1.0, 0.0, 0.0),
        length_m=0.03,
        radius_m=0.008,
        limit_lo_rad=0.0,
        limit_hi_rad=math.pi,
        station_z_m=0.05,
        azimuth_rad=0.0,
        clearance_extra_m=0.0,
        arch_weight=0.0,
    )
    return KinematicChain("synthetic", "lead", -1, (joint,))
