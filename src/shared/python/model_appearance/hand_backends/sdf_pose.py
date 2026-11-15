"""Curl-to-contact skeleton for the SDF hand.

Phalanges are straight axes tangent to a circle about the grip, the same
construction as ``grip_hand_mesh._finger``: the pad meets the clearance
circle and consecutive bones share joints. Lead curl decreases azimuth and
trail curl increases it, so the two hands are mirrors when their V and thumb
azimuths match.

If ``hand_backends/posing.py`` defines ``pose_hand_skeleton`` and it returns
a :class:`HandSkeleton`, that shared poser replaces this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends.contract import (
    HAND_ROLES,
    HandJointAngles,
    HandRole,
)
from src.shared.python.model_appearance.hand_backends.sdf_field import Primitive
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    SdfHandSettings,
)

_PHALANX = ("proximal", "middle", "distal")
_DIGITS = ("little", "ring", "middle", "index")


@dataclass(frozen=True)
class HandSkeleton:
    """Posed SDF primitives plus the anthropometric measures taken from them."""

    primitives: tuple[Primitive, ...]
    grip_radius_m: float
    side: HandRole
    scale: float
    dorsal_hint: tuple[float, float, float]
    hand_length_m: float
    palm_width_m: float


def _vec(values: np.ndarray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _push_outside(point: np.ndarray, min_rho: float) -> np.ndarray:
    """Move ``point`` radially out to ``min_rho`` without changing z."""
    pushed = np.array(point, dtype=float, copy=True)
    rho = float(np.hypot(pushed[0], pushed[1]))
    if rho >= min_rho:
        return pushed
    if rho < 1e-8:
        pushed[0] = min_rho
        return pushed
    pushed[0] *= min_rho / rho
    pushed[1] *= min_rho / rho
    return pushed


def _clear_of_grip(
    centre: np.ndarray,
    radii: tuple[float, float, float],
    grip_radius_m: float,
    clearance_m: float,
) -> np.ndarray:
    """Keep an ellipsoid's bounding sphere on or outside the clearance cylinder."""
    return _push_outside(centre, grip_radius_m + clearance_m + max(radii))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("cannot normalise a near-zero vector")
    return vector / norm


def _radial(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta), 0.0])


def _tangential(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta), 0.0])


def _cone(
    name: str,
    group: str,
    kind: str,
    start: np.ndarray,
    end: np.ndarray,
    radius0: float,
    radius1: float,
    palmar: np.ndarray,
    *,
    blend: str = "joint",
    operation: str = "union",
    aspect: float = 1.0,
) -> Primitive:
    return Primitive(
        name=name,
        group=group,
        kind=kind,
        operation=operation,
        shape="cone",
        start=_vec(start),
        end=_vec(end),
        radius0=float(radius0),
        radius1=float(radius1),
        palmar=_vec(palmar),
        aspect=aspect,
        blend=blend,
        radii=(0.0, 0.0, 0.0),
    )


def _sphere(
    name: str,
    group: str,
    kind: str,
    centre: np.ndarray,
    radius: float,
    palmar: np.ndarray,
    *,
    blend: str,
) -> Primitive:
    return Primitive(
        name=name,
        group=group,
        kind=kind,
        operation="union",
        shape="sphere",
        start=_vec(centre),
        end=_vec(centre),
        radius0=float(radius),
        radius1=float(radius),
        palmar=_vec(palmar),
        aspect=1.0,
        blend=blend,
        radii=(0.0, 0.0, 0.0),
    )


def _ellipsoid(
    name: str,
    group: str,
    kind: str,
    centre: np.ndarray,
    along: np.ndarray,
    palmar: np.ndarray,
    radii: tuple[float, float, float],
    *,
    blend: str,
) -> Primitive:
    return Primitive(
        name=name,
        group=group,
        kind=kind,
        operation="union",
        shape="ellipsoid",
        start=_vec(centre),
        end=_vec(along),
        radius0=0.0,
        radius1=0.0,
        palmar=_vec(palmar),
        aspect=1.0,
        blend=blend,
        radii=radii,
    )


def _tangent_chain(
    lengths: tuple[float, ...],
    tangent_radius: float,
    mcp_azimuth: float,
    z: float,
    curl: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Axes tangent to the circle of radius ``tangent_radius``.

    Returns ``(start, end, outward_radial)`` per bone. The outward radial is
    perpendicular to the axis; palmar is its negation.
    """
    before = lengths[0] / 2.0
    theta = mcp_azimuth + curl * math.atan2(before, tangent_radius)
    lift = np.array([0.0, 0.0, z])
    bones: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for length in lengths:
        after = length - before
        if after <= 0.0:
            raise ValueError(f"phalanx length {length} cannot continue the wrap")
        outward = _radial(theta)
        tangent = curl * _tangential(theta)
        start = tangent_radius * outward - before * tangent + lift
        end = tangent_radius * outward + after * tangent + lift
        bones.append((start, end, outward))
        theta += curl * 2.0 * math.atan2(after, tangent_radius)
        before = after
    return bones


def _scaled_radii(
    radii: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    return (radii[0] * scale, radii[1] * scale, radii[2] * scale)


def _closest_axis_point(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    """Point on the segment closest to the grip axis, and its fraction."""
    delta = end - start
    denom = float(delta[0] ** 2 + delta[1] ** 2)
    if denom < 1e-12:
        return 0.5 * (start + end), 0.5
    fraction = -(start[0] * delta[0] + start[1] * delta[1]) / denom
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return start + fraction * delta, fraction


def _add_pad(
    primitives: list[Primitive],
    name: str,
    group: str,
    start: np.ndarray,
    end: np.ndarray,
    radius0: float,
    radius1: float,
    contact_radius: float,
) -> None:
    """Palmar sphere that brings the tangent point out to ``contact_radius``."""
    point, fraction = _closest_axis_point(start, end)
    radius_here = (1.0 - fraction) * radius0 + fraction * radius1
    if contact_radius - radius_here <= 0.0004:
        return
    radial = np.array([point[0], point[1], 0.0])
    rho = float(np.linalg.norm(radial))
    if rho < 1e-8:
        return
    palmar = -radial / rho
    pad_radius = max(radius_here * 0.85, 0.004)
    shift = contact_radius - pad_radius
    if shift <= 0.0:
        return
    primitives.append(
        _sphere(
            f"{name}_pad",
            group,
            "pad",
            point + palmar * shift,
            pad_radius,
            palmar,
            blend="joint",
        )
    )


def _add_knuckle(
    primitives: list[Primitive],
    name: str,
    group: str,
    centre: np.ndarray,
    palmar: np.ndarray,
    bone_radius: float,
    settings: SdfHandSettings,
    scale: float,
) -> None:
    dorsal = -palmar
    bump = bone_radius * settings.knuckle_radius_fraction
    offset = bone_radius * settings.knuckle_dorsal_fraction
    primitives.append(
        _sphere(
            name,
            group,
            "knuckle",
            centre + dorsal * offset,
            max(bump, 0.002 * scale),
            palmar,
            blend="joint",
        )
    )


def _add_crease(
    primitives: list[Primitive],
    name: str,
    centre: np.ndarray,
    axis: np.ndarray,
    palmar: np.ndarray,
    bone_radius: float,
    settings: SdfHandSettings,
) -> None:
    side = _unit(np.cross(axis, palmar))
    half = side * bone_radius * settings.crease_span_fraction
    depth = palmar * bone_radius * settings.crease_depth_fraction
    radius = max(bone_radius * settings.crease_radius_fraction, 0.001)
    primitives.append(
        _cone(
            name,
            "crease",
            "crease",
            centre + depth - half,
            centre + depth + half,
            radius,
            radius,
            palmar,
            blend="crease",
            operation="subtract",
        )
    )


def _measure(primitives: tuple[Primitive, ...]) -> tuple[float, float]:
    """Middle-finger chord length and palm width, both in metres."""
    phalanx = {
        prim.name: prim
        for prim in primitives
        if prim.group == "middle" and prim.kind == "phalanx"
    }
    mc_base = next(prim for prim in primitives if prim.name == "middle_metacarpal_base")
    mc_head = next(prim for prim in primitives if prim.name == "middle_metacarpal_head")
    chord = float(np.linalg.norm(np.asarray(mc_head.end) - np.asarray(mc_base.start)))
    length = chord
    for name in ("middle_proximal", "middle_middle", "middle_distal"):
        prim = phalanx[name]
        length += float(np.linalg.norm(np.asarray(prim.end) - np.asarray(prim.start)))
    index = next(prim for prim in primitives if prim.name == "index_proximal")
    little = next(prim for prim in primitives if prim.name == "little_proximal")
    width = abs(index.start[2] - little.start[2]) + index.radius0 + little.radius0
    return length, width


def pose_curl_to_contact(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: SdfHandSettings,
) -> HandSkeleton:
    """Pose one hand's SDF skeleton curled onto the grip cylinder.

    Preconditions: ``side`` is lead or trail, ``grip_radius_m`` is positive,
    and ``joint_angles`` is a :class:`HandJointAngles`. Postcondition: the
    measured middle-finger length and palm width track the scaled settings,
    and on-grip phalanx axes lie outside the grip.
    """
    if side not in HAND_ROLES:
        raise ValueError(f"side must be one of {HAND_ROLES}, got {side!r}")
    if not isinstance(joint_angles, HandJointAngles):
        raise TypeError(
            f"joint_angles must be HandJointAngles, got {type(joint_angles).__name__}"
        )
    if not math.isfinite(grip_radius_m) or grip_radius_m <= 0.0:
        raise ValueError(f"grip_radius_m must be positive, got {grip_radius_m!r}")
    scale = float(grip_spec.hand_size.length_m) / settings.hand_length_m
    curl = -1 if side == "lead" else 1
    hand = grip_spec.lead if side == "lead" else grip_spec.trail
    v_azimuth = math.radians(hand.v_azimuth_deg)
    depth = joint_angles.palm_finger_depth / settings.full_contact_palm_finger_depth
    contact = min(1.0, max(0.0, depth))
    lift = (1.0 - contact) * settings.open_hand_lift_m * scale
    clearance = settings.pad_clearance_m * scale

    radii_used = {name: settings.finger(name).radii_m[0][0] * scale for name in _DIGITS}
    center_span = (
        settings.palm_width_m * scale - radii_used["index"] - radii_used["little"]
    )
    if center_span <= 0.0:
        raise ValueError("palm_width_m is smaller than the outer finger radii")
    gap = center_span / (len(_DIGITS) - 1)
    z_of = {
        name: (-0.5 * center_span + index * gap) for index, name in enumerate(_DIGITS)
    }

    primitives: list[Primitive] = []
    proximal_starts: dict[str, np.ndarray] = {}
    outward_at: dict[str, np.ndarray] = {}
    palmar_at: dict[str, np.ndarray] = {}
    for finger in _DIGITS:
        profile = settings.finger(finger)
        lengths = tuple(length * scale for length in profile.phalanges_m)
        radii = tuple((pair[0] * scale, pair[1] * scale) for pair in profile.radii_m)
        contact_radius = radii[0][0]
        tangent_radius = grip_radius_m + clearance + contact_radius + lift
        spread = math.radians(settings.finger_spread_deg[finger])
        mcp = v_azimuth + curl * (math.radians(settings.mcp_from_v_deg) + spread)
        bones = _tangent_chain(lengths, tangent_radius, mcp, z_of[finger], curl)
        proximal_starts[finger] = bones[0][0]
        outward_at[finger] = bones[0][2]
        palmar_at[finger] = -bones[0][2]
        for bone_name, (start, end, outward), (radius0, radius1) in zip(
            _PHALANX, bones, radii, strict=True
        ):
            palmar = -outward
            primitives.append(
                _cone(
                    f"{finger}_{bone_name}",
                    finger,
                    "phalanx",
                    start,
                    end,
                    radius0,
                    radius1,
                    palmar,
                )
            )
            _add_pad(
                primitives,
                f"{finger}_{bone_name}",
                finger,
                start,
                end,
                radius0,
                radius1,
                contact_radius,
            )
        joints = (
            ("mcp", bones[0][0], radii[0][0], bones[0][1] - bones[0][0], bones[0][2]),
            ("pip", bones[0][1], radii[0][1], bones[1][1] - bones[1][0], bones[1][2]),
            ("dip", bones[1][1], radii[1][1], bones[2][1] - bones[2][0], bones[2][2]),
        )
        for label, centre, radius, axis, outward in joints:
            palmar = -outward
            _add_knuckle(
                primitives,
                f"{finger}_{label}",
                finger,
                centre,
                palmar,
                radius,
                settings,
                scale,
            )
            if label != "mcp":
                _add_crease(
                    primitives,
                    f"{finger}_{label}_crease",
                    centre,
                    axis,
                    palmar,
                    radius,
                    settings,
                )
        distal_start, distal_end, distal_out = bones[2]
        distal_axis = _unit(distal_end - distal_start)
        nail_centre = (
            distal_end
            - distal_axis * (lengths[2] * settings.nail_along_fraction)
            + distal_out * (radii[2][1] * 0.45)
        )
        primitives.append(
            _ellipsoid(
                f"{finger}_nail",
                finger,
                "nail",
                nail_centre,
                distal_axis,
                -distal_out,
                _scaled_radii(settings.nail_radii_m, scale),
                blend="nail",
            )
        )

    bases: dict[str, np.ndarray] = {}
    heads: dict[str, np.ndarray] = {}
    for finger in _DIGITS:
        profile = settings.finger(finger)
        length = profile.metacarpal_m * scale
        head = proximal_starts[finger]
        radial = outward_at[finger]
        prox = next(prim for prim in primitives if prim.name == f"{finger}_proximal")
        prox_dir = _unit(np.asarray(prox.end) - np.asarray(prox.start))
        wrist_dir = _unit(
            -prox_dir * settings.mc_tangent_weight + radial * settings.mc_radial_weight
        )
        base = head - wrist_dir * length
        bases[finger] = base
        heads[finger] = head
    # Gather the wrist along the shaft only. A full 3D pull toward the
    # centroid swings the metacarpal chord through the grip cylinder.
    centroid_z = float(np.mean([bases[name][2] for name in _DIGITS]))
    for finger in _DIGITS:
        length = settings.finger(finger).metacarpal_m * scale
        gathered = np.array(bases[finger], dtype=float, copy=True)
        gathered[2] = (
            centroid_z + (gathered[2] - centroid_z) * settings.wrist_convergence
        )
        gathered = gathered + outward_at[finger] * (
            settings.transverse_arch_m * scale * settings.transverse_arch_coeff[finger]
        )
        head = heads[finger]
        radial_xy = np.array([head[0], head[1], 0.0])
        radial = (
            _unit(radial_xy)
            if float(np.linalg.norm(radial_xy)) > 1e-8
            else outward_at[finger]
        )
        delta = gathered - head
        outward_component = float(np.dot(delta, radial))
        if outward_component < 0.0:
            delta = delta - outward_component * radial
        if float(np.linalg.norm(delta)) < 1e-8:
            delta = radial
        bases[finger] = head + _unit(delta) * length
    for finger in _DIGITS:
        profile = settings.finger(finger)
        length = profile.metacarpal_m * scale
        base_r, head_r = (
            profile.metacarpal_radii_m[0] * scale,
            profile.metacarpal_radii_m[1] * scale,
        )
        base = bases[finger]
        head = heads[finger]
        bow = outward_at[finger] * (settings.longitudinal_arch_m * scale)
        mid = 0.5 * (base + head) + bow
        mid_r = 0.5 * (base_r + head_r)
        palmar = palmar_at[finger]
        primitives.append(
            _cone(
                f"{finger}_metacarpal_base",
                "palm",
                "metacarpal",
                base,
                mid,
                base_r,
                mid_r,
                palmar,
                blend="palm",
            )
        )
        primitives.append(
            _cone(
                f"{finger}_metacarpal_head",
                "palm",
                "metacarpal",
                mid,
                head,
                mid_r,
                head_r,
                palmar,
                blend="palm",
            )
        )

    thumb = settings.thumb
    thumb_radii = tuple((pair[0] * scale, pair[1] * scale) for pair in thumb.radii_m)
    thumb_contact = max(thumb_radii[0][0], thumb_radii[1][0])
    thumb_azimuth = math.radians(hand.thumb_azimuth_deg)
    outward = _radial(thumb_azimuth)
    palmar = -outward
    tangent_radius = grip_radius_m + clearance + thumb_contact + lift
    centre_xy = tangent_radius * outward
    stations = tuple(station * scale for station in thumb.stations_m)
    straight = [
        (
            centre_xy + np.array([0.0, 0.0, stations[0]]),
            centre_xy + np.array([0.0, 0.0, stations[1]]),
        ),
        (
            centre_xy + np.array([0.0, 0.0, stations[1]]),
            centre_xy + np.array([0.0, 0.0, stations[2]]),
        ),
    ]
    flexed = _tangent_chain(
        tuple(length * scale for length in thumb.phalanges_m),
        tangent_radius,
        thumb_azimuth,
        0.0,
        curl,
    )
    straightness = min(1.0, max(0.0, joint_angles.thumb_straightness))
    thumb_bones: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (start_s, end_s) in enumerate(straight):
        start = straightness * start_s + (1.0 - straightness) * flexed[index][0]
        end = straightness * end_s + (1.0 - straightness) * flexed[index][1]
        thumb_bones.append((start, end))
        radius0, radius1 = thumb_radii[index]
        label = ("proximal", "distal")[index]
        primitives.append(
            _cone(
                f"thumb_{label}",
                "thumb",
                "phalanx",
                start,
                end,
                radius0,
                radius1,
                palmar,
            )
        )
        _add_pad(
            primitives,
            f"thumb_{label}",
            "thumb",
            start,
            end,
            radius0,
            radius1,
            thumb_contact,
        )
    _add_knuckle(
        primitives,
        "thumb_ip",
        "thumb",
        thumb_bones[0][1],
        palmar,
        thumb_radii[0][1],
        settings,
        scale,
    )
    _add_crease(
        primitives,
        "thumb_ip_crease",
        thumb_bones[0][1],
        thumb_bones[1][1] - thumb_bones[1][0],
        palmar,
        thumb_radii[0][1],
        settings,
    )
    index_base = bases["index"]
    thumb_head = thumb_bones[0][0]
    toward_index = index_base - thumb_head
    if float(np.linalg.norm(toward_index)) < 1e-6:
        toward_index = -outward
    mc_dir = _unit(toward_index - palmar * (0.01 * scale))
    mc_length = thumb.metacarpal_m * scale
    mc_base = thumb_head + mc_dir * mc_length
    mc_r0, mc_r1 = (
        thumb.metacarpal_radii_m[0] * scale,
        thumb.metacarpal_radii_m[1] * scale,
    )
    primitives.append(
        _cone(
            "thumb_metacarpal",
            "thumb",
            "metacarpal",
            mc_base,
            thumb_head,
            mc_r0,
            mc_r1,
            palmar,
            blend="joint",
        )
    )
    thumb_mid = 0.5 * (mc_base + thumb_head)
    thenar_radii = _scaled_radii(settings.thenar_radii_m, scale)
    primitives.append(
        _ellipsoid(
            "thenar",
            "thumb",
            "eminence",
            _clear_of_grip(
                thumb_mid + palmar * (0.004 * scale),
                thenar_radii,
                grip_radius_m,
                clearance,
            ),
            mc_dir,
            palmar,
            thenar_radii,
            blend="joint",
        )
    )
    web_radii = _scaled_radii(settings.web_radii_m, scale)
    web_centre = _clear_of_grip(
        0.5 * (thumb_mid + proximal_starts["index"]),
        web_radii,
        grip_radius_m,
        clearance,
    )
    primitives.append(
        _ellipsoid(
            "thumb_web",
            "thumb",
            "web",
            web_centre,
            mc_dir,
            palmar,
            web_radii,
            blend="web",
        )
    )
    little_mid_prim = next(
        prim for prim in primitives if prim.name == "little_metacarpal_head"
    )
    little_mid = 0.5 * (
        np.asarray(little_mid_prim.start) + np.asarray(little_mid_prim.end)
    )
    hypothenar_radii = _scaled_radii(settings.hypothenar_radii_m, scale)
    primitives.append(
        _ellipsoid(
            "hypothenar",
            "palm",
            "eminence",
            _clear_of_grip(
                little_mid
                + palmar_at["little"] * (0.004 * scale)
                + np.array([0.0, 0.0, -0.006 * scale]),
                hypothenar_radii,
                grip_radius_m,
                clearance,
            ),
            outward_at["little"],
            palmar_at["little"],
            hypothenar_radii,
            blend="palm",
        )
    )
    mc_mids = []
    for finger in _DIGITS:
        base_prim = next(
            prim for prim in primitives if prim.name == f"{finger}_metacarpal_base"
        )
        mc_mids.append(0.5 * (np.asarray(base_prim.start) + np.asarray(base_prim.end)))
    palm_along = _unit(sum((heads[name] - bases[name]) for name in _DIGITS))
    palm_radii = _scaled_radii(settings.palm_fill_radii_m, scale)
    wrist_radii = _scaled_radii(settings.wrist_radii_m, scale)
    primitives.append(
        _ellipsoid(
            "palm_fill",
            "palm",
            "palm",
            _clear_of_grip(
                sum(mc_mids) / len(mc_mids), palm_radii, grip_radius_m, clearance
            ),
            palm_along,
            palmar_at["middle"],
            palm_radii,
            blend="palm",
        )
    )
    primitives.append(
        _ellipsoid(
            "wrist",
            "palm",
            "wrist",
            _clear_of_grip(
                sum(bases.values()) / len(bases), wrist_radii, grip_radius_m, clearance
            ),
            palm_along,
            palmar_at["middle"],
            wrist_radii,
            blend="palm",
        )
    )
    dorsal = _unit(sum(outward_at.values()))
    dorsal = np.array([dorsal[0], dorsal[1], 0.0])
    dorsal = _unit(dorsal)
    packed = tuple(primitives)
    if grip_spec.handedness == "left":
        packed, dorsal = _mirror_y(packed, dorsal)
    length_m, width_m = _measure(packed)
    return HandSkeleton(
        primitives=packed,
        grip_radius_m=float(grip_radius_m),
        side=side,
        scale=scale,
        dorsal_hint=_vec(dorsal),
        hand_length_m=length_m,
        palm_width_m=width_m,
    )


def _mirror_y(
    primitives: tuple[Primitive, ...], dorsal: np.ndarray
) -> tuple[tuple[Primitive, ...], np.ndarray]:
    def flip(point: tuple[float, float, float]) -> tuple[float, float, float]:
        return (point[0], -point[1], point[2])

    mirrored = tuple(
        replace(
            prim,
            start=flip(prim.start),
            end=flip(prim.end),
            palmar=flip(prim.palmar),
        )
        for prim in primitives
    )
    dorsal_m = np.array([dorsal[0], -dorsal[1], dorsal[2]])
    return mirrored, dorsal_m


def _shared_poser():
    try:
        from src.shared.python.model_appearance.hand_backends import posing
    except ImportError:
        return None
    return getattr(posing, "pose_hand_skeleton", None)


def pose_hand_skeleton(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float,
    settings: SdfHandSettings,
) -> HandSkeleton:
    """Pose via ``posing.pose_hand_skeleton`` when that module has landed."""
    shared = _shared_poser()
    if shared is not None:
        try:
            posed = shared(
                grip_spec,
                side,
                joint_angles,
                grip_radius_m=grip_radius_m,
                settings=settings,
            )
        except TypeError:
            posed = None
        if isinstance(posed, HandSkeleton):
            return posed
    return pose_curl_to_contact(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=grip_radius_m,
        settings=settings,
    )


def phalanx_contact_points(
    skeleton: HandSkeleton, grip_radius_m: float
) -> dict[str, np.ndarray]:
    """Grip-surface point nearest each phalanx axis.

    The sample is the cylinder point under the axis location closest to the
    grip axis. The hand SDF there is the analytic gap (positive = clearance).
    """
    if grip_radius_m <= 0.0:
        raise ValueError(f"grip_radius_m must be positive, got {grip_radius_m!r}")
    points: dict[str, np.ndarray] = {}
    for prim in skeleton.primitives:
        if prim.kind != "phalanx" or prim.shape != "cone" or prim.operation != "union":
            continue
        start = np.asarray(prim.start, dtype=float)
        end = np.asarray(prim.end, dtype=float)
        delta = end - start
        denom = float(delta[0] ** 2 + delta[1] ** 2)
        if denom < 1e-12:
            fraction = 0.5
        else:
            fraction = -(start[0] * delta[0] + start[1] * delta[1]) / denom
            fraction = float(np.clip(fraction, 0.0, 1.0))
        point = start + fraction * delta
        radial = np.array([point[0], point[1], 0.0])
        rho = float(np.linalg.norm(radial))
        if rho < 1e-8:
            continue
        points[prim.name] = np.array(
            [
                radial[0] / rho * grip_radius_m,
                radial[1] / rho * grip_radius_m,
                point[2],
            ]
        )
    return points


__all__ = [
    "HandSkeleton",
    "phalanx_contact_points",
    "pose_curl_to_contact",
    "pose_hand_skeleton",
]
