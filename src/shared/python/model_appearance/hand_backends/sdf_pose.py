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

import importlib
import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec
from src.shared.python.hand_size import resolve_hand_breadth_m
from src.shared.python.model_appearance.hand_backends.contract import (
    HAND_ROLES,
    GripArticulation,
    HandRole,
)
from src.shared.python.model_appearance.hand_backends.sdf_field import Primitive
from src.shared.python.model_appearance.hand_backends.sdf_primitives import (
    Bone,
    PoseFrame,
    add_crease,
    add_finger_joints,
    add_knuckle,
    add_nail,
    add_pad,
    cone_primitive,
    ellipsoid_primitive,
    measure_skeleton,
    mirror_y,
    radial_dir,
    scaled_radii,
    tangent_chain,
    blend_thumb_bones,
    unit,
    vec3,
)
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    SdfHandSettings,
)

_PHALANX = ("proximal", "middle", "distal")
_DIGITS = ("little", "ring", "middle", "index")
_CROSS_PIVOT_WEIGHTS: dict[str, float] = {
    "little": -1.0,
    "ring": -0.4,
    "middle": 0.0,
    "index": 0.75,
}


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


def _pose_finger(
    primitives: list[Primitive],
    frame: PoseFrame,
    finger: str,
    *,
    mcp: float,
    z: float,
    finger_lift: float,
) -> list[Bone]:
    scale = frame.scale
    profile = frame.settings.finger(finger)
    lengths = tuple(length * scale for length in profile.phalanges_m)
    radii = tuple((pair[0] * scale, pair[1] * scale) for pair in profile.radii_m)
    contact_radius = radii[0][0]
    tangent_radius = (
        frame.grip_radius_m + frame.clearance + contact_radius + finger_lift
    )
    bones = tangent_chain(lengths, tangent_radius, mcp, z, frame.curl)
    for bone_name, (start, end, outward), (radius0, radius1) in zip(
        _PHALANX, bones, radii, strict=True
    ):
        palmar = -outward
        primitives.append(
            cone_primitive(
                f"{finger}_{bone_name}",
                finger,
                "phalanx",
                start,
                end,
                (radius0, radius1),
                palmar,
            )
        )
        add_pad(
            primitives,
            f"{finger}_{bone_name}",
            finger,
            start,
            end,
            radius0,
            radius1,
            contact_radius,
        )
    add_finger_joints(primitives, frame, finger, bones, radii)
    add_nail(primitives, frame, finger, bones[2], lengths[2], radii[2][1])
    return bones


def _pose_fingers(
    primitives: list[Primitive],
    frame: PoseFrame,
    *,
    grip_spec: GripSpec,
    side: HandRole,
    breadth_ratio: float,
    v_azimuth: float,
    cross_delta_deg: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    settings, scale = frame.settings, frame.scale
    radii_used = {name: settings.finger(name).radii_m[0][0] * scale for name in _DIGITS}
    target_width = settings.palm_width_m * scale * breadth_ratio
    center_span = target_width - radii_used["index"] - radii_used["little"]
    if center_span <= 0.0:
        raise ValueError("palm_width_m is smaller than the outer finger radii")
    gap = center_span / (len(_DIGITS) - 1)
    z_of = {name: (-0.5 * center_span + idx * gap) for idx, name in enumerate(_DIGITS)}

    proximal_starts: dict[str, np.ndarray] = {}
    outward_at: dict[str, np.ndarray] = {}
    palmar_at: dict[str, np.ndarray] = {}
    for finger in _DIGITS:
        finger_lift = frame.lift
        if (
            side == "trail"
            and finger == "little"
            and grip_spec.style != GripStyle.TEN_FINGER
            and grip_spec.lead != grip_spec.trail
        ):
            finger_lift += settings.overlap_little_lift_m * scale
        spread = math.radians(settings.finger_spread_deg[finger])
        cross_shift = math.radians(
            _CROSS_PIVOT_WEIGHTS[finger]
            * cross_delta_deg
            * settings.cross_angle_mcp_tilt_gain
        )
        mcp = v_azimuth + frame.curl * (
            math.radians(settings.mcp_from_v_deg) + spread + cross_shift
        )
        bones = _pose_finger(
            primitives, frame, finger, mcp=mcp, z=z_of[finger], finger_lift=finger_lift
        )
        proximal_starts[finger] = bones[0][0]
        outward_at[finger] = bones[0][2]
        palmar_at[finger] = -bones[0][2]
    return proximal_starts, outward_at, palmar_at


def _metacarpal_bases(
    primitives: list[Primitive],
    frame: PoseFrame,
    heads: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    settings, scale = frame.settings, frame.scale
    grip_radius_m, clearance = frame.grip_radius_m, frame.clearance
    bases: dict[str, np.ndarray] = {}
    centroid_z = float(np.mean([heads[name][2] for name in _DIGITS]))
    for finger in _DIGITS:
        profile = settings.finger(finger)
        length = profile.metacarpal_m * scale
        head = heads[finger]
        prox = next(prim for prim in primitives if prim.name == f"{finger}_proximal")
        prox_dir = unit(np.asarray(prox.end) - np.asarray(prox.start))
        curl_sign = (
            1.0 if float(head[0] * prox_dir[1] - head[1] * prox_dir[0]) >= 0.0 else -1.0
        )
        az_head = math.atan2(float(head[1]), float(head[0]))
        az_base = az_head - curl_sign * settings.palm_wrap_arc_rad
        extra_lift = max(
            0.0,
            float(np.hypot(head[0], head[1]))
            - (grip_radius_m + clearance + profile.radii_m[0][0] * scale),
        )
        r_target = (
            grip_radius_m
            + clearance
            + extra_lift
            + (
                settings.palm_base_hover_m
                + settings.transverse_arch_m * settings.transverse_arch_coeff[finger]
            )
            * scale
        )
        z_target = (
            centroid_z
            - settings.palm_axial_drop_m * scale
            + (head[2] - centroid_z) * settings.wrist_convergence
        )
        gathered = np.array(
            [
                r_target * math.cos(az_base),
                r_target * math.sin(az_base),
                z_target,
            ],
            dtype=np.float64,
        )
        bases[finger] = head + unit(gathered - head) * length
    return bases


def _add_metacarpal_cones(
    primitives: list[Primitive],
    frame: PoseFrame,
    bases: dict[str, np.ndarray],
    heads: dict[str, np.ndarray],
    outward_at: dict[str, np.ndarray],
    palmar_at: dict[str, np.ndarray],
) -> None:
    settings, scale = frame.settings, frame.scale
    for finger in _DIGITS:
        profile = settings.finger(finger)
        base_r, head_r = (
            profile.metacarpal_radii_m[0] * scale,
            profile.metacarpal_radii_m[1] * scale,
        )
        base = bases[finger]
        head = heads[finger]
        mid_raw = 0.5 * (base + head)
        radial_xy = np.array([mid_raw[0], mid_raw[1], 0.0], dtype=np.float64)
        radial_mid = (
            unit(radial_xy)
            if float(np.linalg.norm(radial_xy)) > 1e-8
            else outward_at[finger]
        )
        bow = radial_mid * (settings.longitudinal_arch_m * scale)
        mid = mid_raw + bow
        mid_r = 0.5 * (base_r + head_r)
        palmar = palmar_at[finger]
        primitives.append(
            cone_primitive(
                f"{finger}_metacarpal_base",
                "palm",
                "metacarpal",
                base,
                mid,
                (base_r, mid_r),
                palmar,
                blend="palm",
            )
        )
        primitives.append(
            cone_primitive(
                f"{finger}_metacarpal_head",
                "palm",
                "metacarpal",
                mid,
                head,
                (mid_r, head_r),
                palmar,
                blend="palm",
            )
        )


def _add_hypothenar(
    primitives: list[Primitive],
    frame: PoseFrame,
    bases: dict[str, np.ndarray],
    heads: dict[str, np.ndarray],
    palmar_at: dict[str, np.ndarray],
) -> None:
    scale = frame.scale
    little_mid_prim = next(
        prim for prim in primitives if prim.name == "little_metacarpal_head"
    )
    little_mid = 0.5 * (
        np.asarray(little_mid_prim.start) + np.asarray(little_mid_prim.end)
    )
    hypothenar_radii = scaled_radii(frame.settings.hypothenar_radii_m, scale)
    hypothenar_along = unit(heads["little"] - bases["little"])
    primitives.append(
        ellipsoid_primitive(
            "hypothenar",
            "palm",
            "eminence",
            frame.clear(
                little_mid
                + palmar_at["little"] * (0.003 * scale)
                + np.array([0.0, 0.0, -0.005 * scale], dtype=np.float64),
                hypothenar_along,
                palmar_at["little"],
                hypothenar_radii,
            ),
            hypothenar_along,
            palmar_at["little"],
            hypothenar_radii,
            blend="palm",
        )
    )


def _add_palm_and_wrist(
    primitives: list[Primitive],
    frame: PoseFrame,
    bases: dict[str, np.ndarray],
    heads: dict[str, np.ndarray],
    palmar: np.ndarray,
) -> None:
    settings, scale = frame.settings, frame.scale
    mean_heads = sum(
        (heads[name] for name in _DIGITS), np.zeros(3, dtype=np.float64)
    ) / len(_DIGITS)
    mean_bases = sum(
        (bases[name] for name in _DIGITS), np.zeros(3, dtype=np.float64)
    ) / len(_DIGITS)
    palm_along = unit(mean_heads - mean_bases)
    palm_radii = scaled_radii(settings.palm_fill_radii_m, scale)
    wrist_radii = scaled_radii(settings.wrist_radii_m, scale)
    w_head = settings.palm_head_weight
    palm_raw = w_head * mean_heads + (1.0 - w_head) * mean_bases
    primitives.append(
        ellipsoid_primitive(
            "palm_fill",
            "palm",
            "palm",
            frame.clear(palm_raw, palm_along, palmar, palm_radii),
            palm_along,
            palmar,
            palm_radii,
            blend="palm",
        )
    )
    primitives.append(
        ellipsoid_primitive(
            "wrist",
            "palm",
            "wrist",
            frame.clear(
                0.92 * mean_bases + 0.08 * mean_heads,
                palm_along,
                palmar,
                wrist_radii,
            ),
            palm_along,
            palmar,
            wrist_radii,
            blend="palm",
        )
    )


def _pose_metacarpals_and_palm(
    primitives: list[Primitive],
    frame: PoseFrame,
    *,
    proximal_starts: dict[str, np.ndarray],
    outward_at: dict[str, np.ndarray],
    palmar_at: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    heads: dict[str, np.ndarray] = {
        finger: proximal_starts[finger] for finger in _DIGITS
    }
    bases = _metacarpal_bases(primitives, frame, heads)
    _add_metacarpal_cones(primitives, frame, bases, heads, outward_at, palmar_at)
    _add_hypothenar(primitives, frame, bases, heads, palmar_at)
    _add_palm_and_wrist(primitives, frame, bases, heads, palmar_at["middle"])
    return bases, heads


def _add_thumb_palm(
    primitives: list[Primitive],
    frame: PoseFrame,
    *,
    thumb_head: np.ndarray,
    outward: np.ndarray,
    bases: dict[str, np.ndarray],
    proximal_starts: dict[str, np.ndarray],
) -> None:
    settings, scale = frame.settings, frame.scale
    thumb = settings.thumb
    palmar = -outward
    index_base = 0.7 * bases["index"] + 0.3 * bases["middle"]
    toward_index = index_base - thumb_head
    if float(np.linalg.norm(toward_index)) < 1e-6:
        toward_index = -outward
    mc_dir = unit(toward_index - palmar * (0.008 * scale))
    mc_length = thumb.metacarpal_m * scale
    mc_base = thumb_head + mc_dir * mc_length
    mc_r0, mc_r1 = (
        thumb.metacarpal_radii_m[0] * scale,
        thumb.metacarpal_radii_m[1] * scale,
    )
    primitives.append(
        cone_primitive(
            "thumb_metacarpal",
            "thumb",
            "metacarpal",
            mc_base,
            thumb_head,
            (mc_r0, mc_r1),
            palmar,
            blend="joint",
        )
    )
    thumb_mid = 0.5 * (mc_base + thumb_head)
    thenar_radii = scaled_radii(settings.thenar_radii_m, scale)
    primitives.append(
        ellipsoid_primitive(
            "thenar",
            "thumb",
            "eminence",
            frame.clear(
                thumb_mid + palmar * (0.003 * scale), mc_dir, palmar, thenar_radii
            ),
            mc_dir,
            palmar,
            thenar_radii,
            blend="joint",
        )
    )
    web_radii = scaled_radii(settings.web_radii_m, scale)
    web_along = unit(
        0.5 * (proximal_starts["index"] + proximal_starts["middle"]) - thumb_mid
    )
    web_centre = frame.clear(
        0.45 * (0.5 * (thumb_head + proximal_starts["index"]))
        + 0.55 * (0.5 * (thumb_mid + bases["index"])),
        web_along,
        palmar,
        web_radii,
    )
    primitives.append(
        ellipsoid_primitive(
            "thumb_web",
            "thumb",
            "web",
            web_centre,
            web_along,
            palmar,
            web_radii,
            blend="web",
        )
    )


def _pose_thumb(
    primitives: list[Primitive],
    frame: PoseFrame,
    *,
    hand: HandGripSpec,
    joint_angles: GripArticulation,
    bases: dict[str, np.ndarray],
    proximal_starts: dict[str, np.ndarray],
) -> None:
    settings, scale = frame.settings, frame.scale
    thumb = settings.thumb
    thumb_radii = tuple((pair[0] * scale, pair[1] * scale) for pair in thumb.radii_m)
    thumb_contact = max(thumb_radii[0][0], thumb_radii[1][0])
    thumb_azimuth = math.radians(hand.thumb_azimuth_deg)
    outward = radial_dir(thumb_azimuth)
    palmar = -outward
    tangent_radius = frame.grip_radius_m + frame.clearance + thumb_contact + frame.lift
    straightness = min(1.0, max(0.0, joint_angles.thumb_straightness))
    thumb_bones = blend_thumb_bones(
        frame,
        thumb_azimuth=thumb_azimuth,
        outward=outward,
        tangent_radius=tangent_radius,
        straightness=straightness,
    )
    for index, (start, end) in enumerate(thumb_bones):
        radius0, radius1 = thumb_radii[index]
        label = ("proximal", "distal")[index]
        primitives.append(
            cone_primitive(
                f"thumb_{label}",
                "thumb",
                "phalanx",
                start,
                end,
                (radius0, radius1),
                palmar,
            )
        )
        add_pad(
            primitives,
            f"thumb_{label}",
            "thumb",
            start,
            end,
            radius0,
            radius1,
            thumb_contact,
        )
    add_knuckle(
        primitives,
        "thumb_ip",
        "thumb",
        thumb_bones[0][1],
        palmar,
        thumb_radii[0][1],
        settings,
        scale,
    )
    add_crease(
        primitives,
        "thumb_ip_crease",
        thumb_bones[0][1],
        thumb_bones[1][1] - thumb_bones[1][0],
        palmar,
        thumb_radii[0][1],
        settings,
    )
    _add_thumb_palm(
        primitives,
        frame,
        thumb_head=thumb_bones[0][0],
        outward=outward,
        bases=bases,
        proximal_starts=proximal_starts,
    )


def _pose_frame(
    grip_spec: GripSpec,
    joint_angles: GripArticulation,
    *,
    grip_radius_m: float,
    settings: SdfHandSettings,
    curl: int,
) -> PoseFrame:
    scale = float(grip_spec.hand_size.length_m) / settings.hand_length_m
    depth = joint_angles.palm_finger_depth / settings.full_contact_palm_finger_depth
    contact = min(1.0, max(0.0, depth))
    return PoseFrame(
        settings=settings,
        scale=scale,
        grip_radius_m=grip_radius_m,
        clearance=settings.pad_clearance_m * scale,
        lift=(1.0 - contact) * settings.open_hand_lift_m * scale,
        curl=curl,
        margin_m=settings.ellipsoid_support_margin_m * scale,
    )


def pose_curl_to_contact(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: GripArticulation,
    *,
    grip_radius_m: float,
    settings: SdfHandSettings,
) -> HandSkeleton:
    """Pose one hand's SDF skeleton curled onto the grip cylinder.

    Preconditions: ``side`` is lead or trail, ``grip_radius_m`` is positive,
    and ``joint_angles`` is a :class:`GripArticulation`. Postcondition: the
    measured middle-finger length and palm width track the scaled settings,
    and on-grip phalanx axes lie outside the grip.
    """
    if side not in HAND_ROLES:
        raise ValueError(f"side must be one of {HAND_ROLES}, got {side!r}")
    if not isinstance(joint_angles, GripArticulation):
        raise TypeError(
            f"joint_angles must be GripArticulation, got {type(joint_angles).__name__}"
        )
    if not math.isfinite(grip_radius_m) or grip_radius_m <= 0.0:
        raise ValueError(f"grip_radius_m must be positive, got {grip_radius_m!r}")

    frame = _pose_frame(
        grip_spec,
        joint_angles,
        grip_radius_m=grip_radius_m,
        settings=settings,
        curl=-1 if side == "lead" else 1,
    )
    ref_breadth = resolve_hand_breadth_m(float(grip_spec.hand_size.length_m))
    breadth_ratio = float(grip_spec.hand_size.breadth_m) / ref_breadth
    hand = grip_spec.lead if side == "lead" else grip_spec.trail
    v_azimuth = math.radians(hand.v_azimuth_deg)
    cross_delta_deg = (
        float(joint_angles.club_cross_angle_deg) - settings.legacy_club_cross_angle_deg
    )

    primitives: list[Primitive] = []
    proximal_starts, outward_at, palmar_at = _pose_fingers(
        primitives,
        frame,
        grip_spec=grip_spec,
        side=side,
        breadth_ratio=breadth_ratio,
        v_azimuth=v_azimuth,
        cross_delta_deg=cross_delta_deg,
    )
    bases, _heads = _pose_metacarpals_and_palm(
        primitives,
        frame,
        proximal_starts=proximal_starts,
        outward_at=outward_at,
        palmar_at=palmar_at,
    )
    _pose_thumb(
        primitives,
        frame,
        hand=hand,
        joint_angles=joint_angles,
        bases=bases,
        proximal_starts=proximal_starts,
    )

    dorsal_sum = sum(outward_at.values(), np.zeros(3, dtype=np.float64))
    dorsal = unit(np.array([dorsal_sum[0], dorsal_sum[1], 0.0], dtype=np.float64))
    packed = tuple(primitives)
    if grip_spec.handedness == "left":
        packed, dorsal = mirror_y(packed, dorsal)
    length_m, width_m = measure_skeleton(packed)
    return HandSkeleton(
        primitives=packed,
        grip_radius_m=float(grip_radius_m),
        side=side,
        scale=frame.scale,
        dorsal_hint=vec3(dorsal),
        hand_length_m=length_m,
        palm_width_m=width_m,
    )


def _shared_poser():
    try:
        posing = importlib.import_module(
            "src.shared.python.model_appearance.hand_backends.posing"
        )
    except ImportError:
        return None
    return getattr(posing, "pose_hand_skeleton", None)


def pose_hand_skeleton(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: GripArticulation,
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
    """Grip-surface point nearest each phalanx axis."""
    if grip_radius_m <= 0.0:
        raise ValueError(f"grip_radius_m must be positive, got {grip_radius_m!r}")
    points: dict[str, np.ndarray] = {}
    for prim in skeleton.primitives:
        if prim.kind != "phalanx" or prim.shape != "cone" or prim.operation != "union":
            continue
        start = np.asarray(prim.start, dtype=np.float64)
        end = np.asarray(prim.end, dtype=np.float64)
        delta = end - start
        denom = float(delta[0] ** 2 + delta[1] ** 2)
        if denom < 1e-12:
            fraction = 0.5
        else:
            fraction = -(start[0] * delta[0] + start[1] * delta[1]) / denom
            fraction = float(np.clip(fraction, 0.0, 1.0))
        point = start + fraction * delta
        radial = np.array([point[0], point[1], 0.0], dtype=np.float64)
        rho = float(np.linalg.norm(radial))
        if rho < 1e-8:
            continue
        points[prim.name] = np.array(
            [
                radial[0] / rho * grip_radius_m,
                radial[1] / rho * grip_radius_m,
                point[2],
            ],
            dtype=np.float64,
        )
    return points


__all__ = [
    "HandSkeleton",
    "phalanx_contact_points",
    "pose_curl_to_contact",
    "pose_hand_skeleton",
]
