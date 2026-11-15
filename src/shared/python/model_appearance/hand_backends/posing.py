"""Backend-agnostic grip poser.

The hand starts flat: the index proximal tangent point sits on the grip and
the pinky tangent point is ``just_below_pinky_m`` farther out, with only a
fraction of ``club_cross_angle_deg`` used as knuckle azimuth. Each finger
then curls, proximal to distal, until its surface meets the grip or, for
overlap and interlock, the lead index. Metacarpal arch angles cup the palm
and, on the final pose, spend some of the pinky clearance. The flat snapshot
keeps Dieter's cross.

GB2 (#15134) is consulted and not applied. Visual backends should take
``angles_rad`` and the contact metrics; they should not copy this capsule
drawing as skin.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from src.shared.python.contracts import ensure
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.shaft_roll import (
    SHAFT_ROLL_RANGE_DEG,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.grip_hand_mesh import PHALANX_NAMES
from src.shared.python.model_appearance.hand_backends.chain import (
    KinematicChain,
    build_default_chain,
)
from src.shared.python.model_appearance.hand_backends.gb2_adapter import gb2_curl_status
from src.shared.python.model_appearance.hand_backends.grip_solid import (
    GripSolid,
    default_grip,
)
from src.shared.python.model_appearance.hand_backends.metrics import (
    contact_quality_metrics,
    palm_contact_area_m2,
)
from src.shared.python.model_appearance.hand_backends.report import (
    FingerOverride,
    FlatPlacement,
    GripPoseSolution,
    HandPose,
    ParallelDonor,
    SegmentPose,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    PoseSettings,
    load_pose_settings,
)
from src.shared.python.model_appearance.hand_backends.solve import (
    axial_segments,
    bisect_contact,
    blend_segment,
    grip_gap_m,
    hinge_child_endpoints,
    require_shaft_parallel,
    roll_point,
    seat_radius,
    shaft_angle_deg,
    surface_gap_m,
    tangent_finger,
    target_clearance,
)

_Point = tuple[float, float, float]
_LIMIT_SLACK_RAD = 1e-6


class _Raw:
    """Unrolled capsule before shaft roll and neighbour measurement."""

    def __init__(
        self,
        name: str,
        finger: str,
        segment: str,
        p0: _Point,
        p1: _Point,
        angle_rad: float,
        grip_gap_m: float,
        wrap_rad: float,
        radius_m: float,
        seat_radius_m: float,
        tangent_azimuth_rad: float,
        check_grip: bool,
        check_neighbour: bool,
    ) -> None:
        self.name = name
        self.finger = finger
        self.segment = segment
        self.p0 = p0
        self.p1 = p1
        self.angle_rad = angle_rad
        self.grip_gap_m = grip_gap_m
        self.wrap_rad = wrap_rad
        self.radius_m = radius_m
        self.seat_radius_m = seat_radius_m
        self.tangent_azimuth_rad = tangent_azimuth_rad
        self.check_grip = check_grip
        self.check_neighbour = check_neighbour


def pose_grip(
    spec: GripSpec,
    grip: GripSolid | None = None,
    *,
    shaft_roll_deg: float | None = None,
    settings: PoseSettings | None = None,
) -> GripPoseSolution:
    """Pose both hands on ``grip``.

    Preconditions: ``spec`` is a GripSpec. ``shaft_roll_deg``, when given,
    replaces both hands' resolved rolls. Postconditions: joint angles lie in
    their limits; ``gb2['applied']`` is false; ``trigger_finger`` and
    ``joint_coupling_overrides`` are reported as not applied.
    """
    config, solid = _inputs(spec, grip, settings)
    lead_chain = build_default_chain(spec, "lead", config)
    trail_chain = build_default_chain(spec, "trail", config)
    lead = pose_chain(
        lead_chain,
        spec,
        solid,
        side="lead",
        shaft_roll_deg=shaft_roll_deg,
        settings=config,
    )
    overrides = _style_overrides(spec, lead, trail_chain, solid, config)
    neighbours = tuple(
        segment for segment in lead.segments if segment.finger == "index"
    )
    trail = pose_chain(
        trail_chain,
        spec,
        solid,
        side="trail",
        shaft_roll_deg=shaft_roll_deg,
        settings=config,
        finger_overrides=overrides,
        neighbour_segments=neighbours,
        neighbour_finger="little",
    )
    metrics = contact_quality_metrics(lead, trail, config)
    gb2 = gb2_curl_status(spec)
    ensure(gb2["applied"] is False, "GB2 angles must not replace the local pose")
    return GripPoseSolution(
        lead=lead,
        trail=trail,
        metrics=MappingProxyType(metrics),
        gb2=MappingProxyType(gb2),
        trigger_finger_applied=False,
        coupling_overrides_applied=False,
    )


def pose_chain(
    chain: KinematicChain,
    spec: GripSpec,
    grip: GripSolid,
    *,
    side: str,
    shaft_roll_deg: float | None = None,
    settings: PoseSettings | None = None,
    finger_overrides: Mapping[str, FingerOverride] | None = None,
    neighbour_segments: Sequence[SegmentPose] | None = None,
    neighbour_finger: str = "little",
) -> HandPose:
    """Pose one chain.

    Preconditions: ``chain.side`` matches ``side``, the grip radii used by
    the fingers are positive, and overrides name fingers on this chain.
    Postcondition: every joint has a finite angle inside its limits.
    """
    config = _settings(settings)
    _check_chain_call(chain, spec, grip, side, finger_overrides)
    roll = _roll_deg(spec, side, shaft_roll_deg)
    hand = spec.lead if side == "lead" else spec.trail
    overrides = {} if finger_overrides is None else dict(finger_overrides)
    raw: list[_Raw] = []
    angles: dict[str, float] = {}
    for finger in _tangent_fingers(chain):
        posed, finger_angles = _pose_tangent_finger(
            chain, finger, grip, hand.palm_finger_depth, overrides.get(finger), config
        )
        raw.extend(posed)
        angles.update(finger_angles)
    raw.extend(_pose_thumb(chain, spec, side, grip, config, angles))
    _pose_arch(chain, hand.palm_finger_depth, config, angles)
    raw.extend(_pose_hinges(chain, grip, config, angles))
    missing = [joint.name for joint in chain.joints if joint.name not in angles]
    if missing:
        raise ValueError(f"unposed joints on {chain.name}: {missing}")
    flat = _flat_placement(chain, grip, config)
    segments = _finish(raw, roll, neighbour_segments, neighbour_finger, config)
    _ensure_limits(chain, angles)
    area = palm_contact_area_m2(hand.palm_finger_depth, config)
    return HandPose(
        side=side,
        curl_sign=chain.curl_sign,
        shaft_roll_deg=roll,
        angles_rad=MappingProxyType(angles),
        segments=tuple(segments),
        flat=flat,
        palm_contact_area_m2=area,
    )


def _inputs(
    spec: object,
    grip: GripSolid | None,
    settings: PoseSettings | None,
) -> tuple[PoseSettings, GripSolid]:
    if not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be a GripSpec, got {type(spec).__name__}")
    if grip is None:
        solid = default_grip()
    elif isinstance(grip, GripSolid):
        solid = grip
    else:
        raise TypeError(f"grip must be a GripSolid, got {type(grip).__name__}")
    return _settings(settings), solid


def _settings(settings: PoseSettings | None) -> PoseSettings:
    if settings is None:
        return load_pose_settings()
    if not isinstance(settings, PoseSettings):
        raise TypeError(f"settings must be PoseSettings, got {type(settings).__name__}")
    return settings


def _check_chain_call(
    chain: object,
    spec: object,
    grip: object,
    side: str,
    overrides: Mapping[str, FingerOverride] | None,
) -> None:
    if not isinstance(chain, KinematicChain):
        raise TypeError(f"chain must be a KinematicChain, got {type(chain).__name__}")
    if not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be a GripSpec, got {type(spec).__name__}")
    if not isinstance(grip, GripSolid):
        raise TypeError(f"grip must be a GripSolid, got {type(grip).__name__}")
    if chain.side != side:
        raise ValueError(f"chain side {chain.side!r} does not match {side!r}")
    if overrides is None:
        return
    known = {joint.finger for joint in chain.joints}
    for finger, override in overrides.items():
        if finger not in known:
            raise ValueError(f"override finger {finger!r} is not on {chain.name}")
        if not isinstance(override, FingerOverride):
            raise TypeError(f"override for {finger} must be a FingerOverride")
        if override.curl_sign is not None and override.curl_sign not in (-1, 1):
            raise ValueError(f"{finger} curl_sign must be -1 or 1")


def _roll_deg(spec: GripSpec, side: str, shaft_roll_deg: float | None) -> float:
    if shaft_roll_deg is None:
        role: Literal["lead", "trail"] = "lead" if side == "lead" else "trail"
        return float(resolved_shaft_roll_deg(spec, role))
    if isinstance(shaft_roll_deg, bool) or not isinstance(shaft_roll_deg, (int, float)):
        raise TypeError(f"shaft_roll_deg must be a number, got {shaft_roll_deg!r}")
    roll = float(shaft_roll_deg)
    lo, hi = SHAFT_ROLL_RANGE_DEG
    if not math.isfinite(roll) or not lo <= roll <= hi:
        raise ValueError(f"shaft_roll_deg must be in [{lo}, {hi}], got {roll}")
    return roll


def _tangent_fingers(chain: KinematicChain) -> list[str]:
    fingers: list[str] = []
    for joint in chain.joints:
        if joint.drive == "tangent" and joint.finger not in fingers:
            fingers.append(joint.finger)
    return fingers


def _pose_tangent_finger(
    chain: KinematicChain,
    finger: str,
    grip: GripSolid,
    depth: float,
    override: FingerOverride | None,
    settings: PoseSettings,
) -> tuple[list[_Raw], dict[str, float]]:
    try:
        joints = [chain.joint(finger, segment) for segment in PHALANX_NAMES]
    except KeyError as exc:
        raise ValueError(f"tangent finger {finger} needs {PHALANX_NAMES}") from exc
    if any(joint.drive != "tangent" for joint in joints):
        raise ValueError(f"{finger} phalanges must all use drive 'tangent'")
    radii = {joint.radius_m for joint in joints}
    if len(radii) != 1:
        raise ValueError(f"{finger} tangent radii must match, got {sorted(radii)}")
    if override is not None and override.parallel_donors:
        return _pose_parallel_finger(joints, override, grip, settings, finger)
    proximal = joints[0]
    station = _pick(override, "station_z_m", proximal.station_z_m)
    azimuth = _pick(override, "azimuth_rad", proximal.azimuth_rad)
    extra = _pick(override, "clearance_extra_m", proximal.clearance_extra_m)
    curl = (
        chain.curl_sign
        if override is None or override.curl_sign is None
        else override.curl_sign
    )
    if override is not None and override.seat_radius_m is not None:
        seat = float(override.seat_radius_m)
        if seat <= 0.0:
            raise ValueError(f"{finger} seat_radius_m must be positive")
    else:
        target = target_clearance(
            extra, depth, proximal.arch_weight, settings, apply_arch=True
        )
        seat = seat_radius(grip, station, proximal.radius_m, target, settings)
    posed = tangent_finger(
        seat, azimuth, station, [joint.length_m for joint in joints], curl
    )
    grip_on = set(
        PHALANX_NAMES if override is None else override.grip_required_segments
    )
    neighbour_on = set(() if override is None else override.neighbour_required_segments)
    raw: list[_Raw] = []
    angles: dict[str, float] = {}
    for joint, item in zip(joints, posed, strict=True):
        start, end, flexion, wrap, theta = item
        _require_flexion(joint.name, flexion, joint.limit_hi_rad)
        gap = seat - grip.radius_at(station) - joint.radius_m
        raw.append(
            _Raw(
                joint.name,
                finger,
                joint.segment,
                start,
                end,
                flexion,
                gap,
                wrap,
                joint.radius_m,
                seat,
                theta,
                joint.segment in grip_on,
                joint.segment in neighbour_on,
            )
        )
        angles[joint.name] = flexion
    return raw, angles


def _pose_parallel_finger(
    joints,
    override: FingerOverride,
    grip: GripSolid,
    settings: PoseSettings,
    finger: str,
) -> tuple[list[_Raw], dict[str, float]]:
    """Lay a shorter finger on donor bones, offset by the two radii.

    A tangent circle one radius outside the index still collides with the
    index capsule ends, which stand farther off the shaft than the tangent
    point. A centered parallel copy stays one surface-gap of zero away from
    its donor and does not inherit that bulge.
    """
    donors = override.parallel_donors
    if donors is None or len(donors) != len(joints):
        raise ValueError(f"{finger} parallel donors must match its phalanges")
    grip_on = set(override.grip_required_segments)
    neighbour_on = set(override.neighbour_required_segments)
    raw: list[_Raw] = []
    angles: dict[str, float] = {}
    for joint, donor in zip(joints, donors, strict=True):
        # The proximal copy is the contact. Later phalanges sit on the same
        # offset and can clip a neighbouring donor end, so they step outward
        # until the surface gap is no longer negative.
        lift = (
            0.0
            if joint.segment in neighbour_on
            else _parallel_lift(donor, joint, donors, settings)
        )
        start, end, seat = _parallel_capsule(
            donor, joint.length_m, joint.radius_m, lift
        )
        _require_flexion(joint.name, donor.angle_rad, joint.limit_hi_rad)
        gap = grip_gap_m(start, end, joint.radius_m, grip, settings.axis_samples)
        midpoint = (
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]),
            0.5 * (start[2] + end[2]),
        )
        raw.append(
            _Raw(
                joint.name,
                finger,
                joint.segment,
                start,
                end,
                donor.angle_rad,
                gap,
                donor.wrap_rad,
                joint.radius_m,
                seat,
                math.atan2(midpoint[1], midpoint[0]),
                joint.segment in grip_on,
                joint.segment in neighbour_on,
            )
        )
        angles[joint.name] = donor.angle_rad
    return raw, angles


def _parallel_lift(donor, joint, donors, settings: PoseSettings) -> float:
    """Smallest extra outward shift that clears every donor capsule."""
    step = settings.penetration_tolerance_m
    for index in range(1, 41):
        extra = index * step
        start, end, _seat = _parallel_capsule(
            donor, joint.length_m, joint.radius_m, extra
        )
        gap = min(
            surface_gap_m(
                start,
                end,
                joint.radius_m,
                other.p0,
                other.p1,
                other.radius_m,
                settings.axis_samples,
            )
            for other in donors
        )
        if gap >= -1e-9:
            return extra
    raise ValueError(f"{joint.name} cannot clear its donor capsules")


def _parallel_capsule(
    donor: ParallelDonor,
    length_m: float,
    radius_m: float,
    extra_m: float = 0.0,
) -> tuple[_Point, _Point, float]:
    midpoint = (
        0.5 * (donor.p0[0] + donor.p1[0]),
        0.5 * (donor.p0[1] + donor.p1[1]),
        0.5 * (donor.p0[2] + donor.p1[2]),
    )
    radial = math.hypot(midpoint[0], midpoint[1])
    if radial <= 1e-9:
        raise ValueError("parallel donor midpoint is on the shaft")
    outward = (midpoint[0] / radial, midpoint[1] / radial, 0.0)
    shift = donor.radius_m + radius_m + extra_m
    center = (
        midpoint[0] + outward[0] * shift,
        midpoint[1] + outward[1] * shift,
        midpoint[2],
    )
    direction = (
        donor.p1[0] - donor.p0[0],
        donor.p1[1] - donor.p0[1],
        donor.p1[2] - donor.p0[2],
    )
    norm = math.dist(direction, (0.0, 0.0, 0.0))
    if norm <= 1e-9:
        raise ValueError("parallel donor has no length")
    half = length_m / 2.0
    start: _Point = (
        center[0] - direction[0] / norm * half,
        center[1] - direction[1] / norm * half,
        center[2] - direction[2] / norm * half,
    )
    end: _Point = (
        center[0] + direction[0] / norm * half,
        center[1] + direction[1] / norm * half,
        center[2] + direction[2] / norm * half,
    )
    return start, end, math.hypot(center[0], center[1])


def _pose_thumb(
    chain: KinematicChain,
    spec: GripSpec,
    side: str,
    grip: GripSolid,
    settings: PoseSettings,
    angles: dict[str, float],
) -> list[_Raw]:
    joints = [joint for joint in chain.joints if joint.drive == "thumb"]
    if not joints:
        return []
    hand = spec.lead if side == "lead" else spec.trail
    radius = joints[0].radius_m
    station = joints[0].station_z_m
    azimuth = joints[0].azimuth_rad
    lengths = [joint.length_m for joint in joints]
    seat = seat_radius(grip, station, radius, settings.base_clearance_m, settings)
    straight = axial_segments(seat, azimuth, station, lengths)
    curled = tangent_finger(seat, azimuth, station, lengths, chain.curl_sign)
    straightness = hand.thumb_straightness
    posed: list[_Raw] = []
    for joint, axial, curl_item in zip(joints, straight, curled, strict=True):
        if straightness >= 1.0 - 1e-9:
            start, end = axial
            flexion, wrap, theta = 0.0, 0.0, azimuth
        elif straightness <= 1e-9:
            start, end, flexion, wrap, theta = curl_item
        else:
            start, end = blend_segment(
                axial, (curl_item[0], curl_item[1]), straightness, seat
            )
            flexion = (1.0 - straightness) * curl_item[2]
            wrap = (1.0 - straightness) * curl_item[3]
            theta = curl_item[4]
        _require_flexion(joint.name, flexion, joint.limit_hi_rad)
        gap = grip_gap_m(start, end, radius, grip, settings.axis_samples)
        posed.append(
            _Raw(
                joint.name,
                "thumb",
                joint.segment,
                start,
                end,
                flexion,
                gap,
                wrap,
                radius,
                seat,
                theta,
                True,
                False,
            )
        )
        angles[joint.name] = flexion
    return posed


def _pose_arch(
    chain: KinematicChain,
    depth: float,
    settings: PoseSettings,
    angles: dict[str, float],
) -> None:
    for joint in chain.joints:
        if joint.drive != "arch":
            continue
        angle = depth * joint.arch_weight * settings.arch_gain_rad
        _require_flexion(joint.name, angle, joint.limit_hi_rad)
        angles[joint.name] = angle


def _pose_hinges(
    chain: KinematicChain,
    grip: GripSolid,
    settings: PoseSettings,
    angles: dict[str, float],
) -> list[_Raw]:
    by_name = {joint.name: joint for joint in chain.joints}
    world_origin: dict[str, _Point] = {}
    world_angle: dict[str, float] = {}
    posed: list[_Raw] = []
    for joint in _ordered(chain):
        if joint.drive != "hinge":
            continue
        require_shaft_parallel(joint.axis, joint.name)
        parent = joint.parent
        if parent is not None and parent in world_angle:
            ancestor = world_angle[parent]
            parent_world = world_origin[parent]
            parent_rest = by_name[parent].origin_m
        else:
            ancestor = 0.0
            parent_world = None
            parent_rest = None
        gap_at = _hinge_gap(joint, ancestor, parent_world, parent_rest, grip, settings)
        angle = bisect_contact(gap_at, joint.limit_hi_rad, settings)
        start, end = hinge_child_endpoints(
            joint.origin_m,
            joint.rest_direction,
            joint.length_m,
            angle,
            ancestor,
            parent_world,
            parent_rest,
        )
        world_origin[joint.name] = start
        world_angle[joint.name] = ancestor + angle
        gap = grip_gap_m(start, end, joint.radius_m, grip, settings.axis_samples)
        posed.append(
            _Raw(
                joint.name,
                joint.finger,
                joint.segment,
                start,
                end,
                angle,
                gap,
                abs(angle),
                joint.radius_m,
                math.hypot(start[0], start[1]),
                math.atan2(start[1], start[0]),
                True,
                False,
            )
        )
        angles[joint.name] = angle
    return posed


def _hinge_gap(joint, ancestor, parent_world, parent_rest, grip, settings):
    def gap_at(angle: float) -> float:
        start, end = hinge_child_endpoints(
            joint.origin_m,
            joint.rest_direction,
            joint.length_m,
            angle,
            ancestor,
            parent_world,
            parent_rest,
        )
        return grip_gap_m(start, end, joint.radius_m, grip, settings.axis_samples)

    return gap_at


def _ordered(chain: KinematicChain):
    children: dict[str, list[str]] = {joint.name: [] for joint in chain.joints}
    pending = {joint.name: 0 for joint in chain.joints}
    by_name = {joint.name: joint for joint in chain.joints}
    for joint in chain.joints:
        if joint.parent is not None:
            children[joint.parent].append(joint.name)
            pending[joint.name] += 1
    ready = [name for name, count in pending.items() if count == 0]
    ordered = []
    while ready:
        name = ready.pop(0)
        ordered.append(by_name[name])
        for child in children[name]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if len(ordered) != len(chain.joints):
        raise ValueError(f"{chain.name} has a joint cycle")
    return ordered


def _flat_placement(
    chain: KinematicChain,
    grip: GripSolid,
    settings: PoseSettings,
) -> FlatPlacement | None:
    try:
        index = chain.joint("index", "proximal")
        little = chain.joint("little", "proximal")
    except KeyError:
        return None
    if index.drive != "tangent" or little.drive != "tangent":
        return None
    index_gap = target_clearance(
        index.clearance_extra_m, 0.0, 0.0, settings, apply_arch=False
    )
    pinky_gap = target_clearance(
        little.clearance_extra_m, 0.0, 0.0, settings, apply_arch=False
    )
    seat = seat_radius(grip, index.station_z_m, index.radius_m, index_gap, settings)
    start, end, *_rest = tangent_finger(
        seat,
        index.azimuth_rad,
        index.station_z_m,
        (index.length_m,),
        chain.curl_sign,
    )[0]
    return FlatPlacement(
        index_gap_m=index_gap,
        pinky_gap_m=pinky_gap,
        index_axis_distance_m=seat,
        index_to_shaft_deg=shaft_angle_deg(start, end),
        index_azimuth_rad=index.azimuth_rad,
        pinky_azimuth_rad=little.azimuth_rad,
    )


def _finish(
    raw: list[_Raw],
    roll: float,
    neighbours: Sequence[SegmentPose] | None,
    neighbour_finger: str,
    settings: PoseSettings,
) -> list[SegmentPose]:
    segments: list[SegmentPose] = []
    for item in raw:
        start = roll_point(item.p0, roll)
        end = roll_point(item.p1, roll)
        neighbour = _neighbour_gap(
            item, start, end, neighbours, neighbour_finger, settings
        )
        if item.check_neighbour and neighbour is None:
            raise ValueError(f"{item.name} needs a neighbour capsule")
        gap = item.grip_gap_m
        if item.check_neighbour and neighbour is not None:
            gap = neighbour if not item.check_grip else max(item.grip_gap_m, neighbour)
        segments.append(
            SegmentPose(
                name=item.name,
                finger=item.finger,
                segment=item.segment,
                p0=start,
                p1=end,
                angle_rad=item.angle_rad,
                gap_m=gap,
                grip_gap_m=item.grip_gap_m,
                penetration_m=max(0.0, -item.grip_gap_m),
                neighbour_gap_m=neighbour,
                wrap_rad=item.wrap_rad,
                radius_m=item.radius_m,
                seat_radius_m=item.seat_radius_m,
                tangent_azimuth_rad=item.tangent_azimuth_rad,
                check_grip=item.check_grip,
                check_neighbour=item.check_neighbour,
            )
        )
    return segments


def _neighbour_gap(
    item: _Raw,
    start: _Point,
    end: _Point,
    neighbours: Sequence[SegmentPose] | None,
    neighbour_finger: str,
    settings: PoseSettings,
) -> float | None:
    if not neighbours or item.finger != neighbour_finger:
        return None
    return min(
        surface_gap_m(
            start,
            end,
            item.radius_m,
            other.p0,
            other.p1,
            other.radius_m,
            settings.axis_samples,
        )
        for other in neighbours
    )


def _style_overrides(
    spec: GripSpec,
    lead: HandPose,
    trail_chain: KinematicChain,
    grip: GripSolid,
    settings: PoseSettings,
) -> dict[str, FingerOverride]:
    if spec.style == GripStyle.TEN_FINGER:
        return {}
    index = lead.segment("index", "proximal")
    little = trail_chain.joint("little", "proximal")
    before = little.length_m / 2.0
    z_index = 0.5 * (index.p0[2] + index.p1[2])
    if spec.style == GripStyle.OVERLAP:
        donors = tuple(
            ParallelDonor(
                p0=roll_point(segment.p0, -lead.shaft_roll_deg),
                p1=roll_point(segment.p1, -lead.shaft_roll_deg),
                radius_m=segment.radius_m,
                angle_rad=segment.angle_rad,
                wrap_rad=segment.wrap_rad,
            )
            for segment in lead.segments
            if segment.finger == "index"
        )
        return {
            "little": FingerOverride(
                grip_required_segments=(),
                neighbour_required_segments=("proximal",),
                parallel_donors=donors,
            )
        }
    if spec.style == GripStyle.INTERLOCK:
        station = z_index + index.radius_m + little.radius_m
        seat = grip.radius_at(station) + little.radius_m + settings.base_clearance_m
        if seat <= 0.0:
            raise ValueError(f"interlock little seat radius {seat} is not positive")
        curl = trail_chain.curl_sign
        azimuth = index.tangent_azimuth_rad - curl * math.atan2(before, seat)
        return {
            "little": FingerOverride(
                station_z_m=station,
                azimuth_rad=azimuth,
                clearance_extra_m=0.0,
                seat_radius_m=seat,
                curl_sign=curl,
                grip_required_segments=PHALANX_NAMES,
                neighbour_required_segments=("proximal",),
            )
        }
    raise ValueError(f"unsupported grip style {spec.style!r}")


def _pick(override: FingerOverride | None, field: str, fallback: float) -> float:
    if override is None:
        return fallback
    value = getattr(override, field)
    return fallback if value is None else float(value)


def _require_flexion(name: str, angle: float, limit: float) -> None:
    if (
        not math.isfinite(angle)
        or angle < -_LIMIT_SLACK_RAD
        or angle > limit + _LIMIT_SLACK_RAD
    ):
        raise ValueError(
            f"{name} flexion {math.degrees(angle):.2f} deg is outside "
            f"[0, {math.degrees(limit):.2f}] deg"
        )


def _ensure_limits(chain: KinematicChain, angles: Mapping[str, float]) -> None:
    for joint in chain.joints:
        angle = angles[joint.name]
        ensure(math.isfinite(angle), f"{joint.name} angle must be finite", angle)
        if angle < joint.limit_lo_rad - _LIMIT_SLACK_RAD:
            raise ValueError(f"{joint.name} angle {angle} is below its limit")
        if angle > joint.limit_hi_rad + _LIMIT_SLACK_RAD:
            raise ValueError(f"{joint.name} angle {angle} is above its limit")


__all__ = ["pose_chain", "pose_grip"]
