"""Backend-agnostic kinematic chain for one hand.

The default chain is the adult golf hand: four fingers of three phalanges,
metacarpal arch joints, and a two-segment thumb. Origins of tangent joints are
not hinge frames. A backend that drives ``hinge`` joints must supply the real
joint origin, axis and rest direction in the appearance grip frame (+Z shaft).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.hand_size import hand_mesh_scale
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_HAND_MESH_PARAMS,
    FINGER_NAMES,
    PHALANX_NAMES,
    HandMeshParams,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    PoseSettings,
    load_pose_settings,
)

_DRIVES = ("tangent", "hinge", "arch", "thumb")
_SIDES = ("lead", "trail")
_SHAFT = (0.0, 0.0, 1.0)


def joint_name(side: str, finger: str, segment: str) -> str:
    """Canonical angle key ``{side}_{finger}_{segment}``."""
    return f"{side}_{finger}_{segment}"


def curl_sign_for(spec: GripSpec, side: str) -> int:
    """Lead curls toward decreasing azimuth. A left-handed golfer flips it."""
    if side not in _SIDES:
        raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")
    sign = -1 if side == "lead" else 1
    if spec.handedness == "left":
        sign *= -1
    return sign


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{label} must be a 3-tuple, got {value!r}")
    coords = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{label} must be finite numbers, got {value!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite, got {value!r}")
        coords.append(number)
    return (coords[0], coords[1], coords[2])


@dataclass(frozen=True)
class JointDescription:
    """One joint a backend can drive.

    Preconditions: finite geometry, a known ``drive``, ``limit_lo_rad`` not
    above ``limit_hi_rad``, and positive length and radius.
    """

    name: str
    parent: str | None
    finger: str
    segment: str
    drive: str
    origin_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    rest_direction: tuple[float, float, float]
    length_m: float
    radius_m: float
    limit_lo_rad: float
    limit_hi_rad: float
    station_z_m: float
    azimuth_rad: float
    clearance_extra_m: float
    arch_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("joint name must be a non-empty string")
        if self.parent is not None and not isinstance(self.parent, str):
            raise TypeError("parent must be a string or None")
        if self.drive not in _DRIVES:
            raise ValueError(f"drive must be one of {_DRIVES}, got {self.drive!r}")
        _vec3(self.origin_m, f"{self.name}.origin_m")
        axis = _vec3(self.axis, f"{self.name}.axis")
        rest = _vec3(self.rest_direction, f"{self.name}.rest_direction")
        if math.dist(axis, (0.0, 0.0, 0.0)) <= 1e-12:
            raise ValueError(f"{self.name}.axis must be non-zero")
        if math.dist(rest, (0.0, 0.0, 0.0)) <= 1e-12:
            raise ValueError(f"{self.name}.rest_direction must be non-zero")
        for label in (
            "length_m",
            "radius_m",
            "limit_lo_rad",
            "limit_hi_rad",
            "station_z_m",
            "azimuth_rad",
            "clearance_extra_m",
            "arch_weight",
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{self.name}.{label} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{self.name}.{label} must be finite")
        if self.length_m <= 0.0 or self.radius_m <= 0.0:
            raise ValueError(f"{self.name} length and radius must be positive")
        if self.limit_lo_rad > self.limit_hi_rad:
            raise ValueError(f"{self.name} limit_lo_rad exceeds limit_hi_rad")
        if not 0.0 <= self.arch_weight <= 1.0:
            raise ValueError(f"{self.name} arch_weight must be in [0, 1]")


@dataclass(frozen=True)
class KinematicChain:
    """Ordered joints for one hand. Parents must name a joint in this chain."""

    name: str
    side: str
    curl_sign: int
    joints: tuple[JointDescription, ...]

    def __post_init__(self) -> None:
        if self.side not in _SIDES:
            raise ValueError(f"side must be 'lead' or 'trail', got {self.side!r}")
        if self.curl_sign not in (-1, 1):
            raise ValueError(f"curl_sign must be -1 or 1, got {self.curl_sign!r}")
        if not self.joints:
            raise ValueError("a kinematic chain needs at least one joint")
        names = [joint.name for joint in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate joint names in {self.name}")
        known = set(names)
        for joint in self.joints:
            if joint.parent is not None and joint.parent not in known:
                raise ValueError(f"{joint.name} parent {joint.parent!r} is missing")
            if joint.parent == joint.name:
                raise ValueError(f"{joint.name} cannot parent itself")

    def joint(self, finger: str, segment: str) -> JointDescription:
        """Return the joint for ``finger`` and ``segment``."""
        for item in self.joints:
            if item.finger == finger and item.segment == segment:
                return item
        raise KeyError(f"no {finger}/{segment} joint on {self.name}")


def _stack_z(scale: float, params: HandMeshParams) -> dict[str, float]:
    """Same axial centres as the grip mesh: index toward the head."""
    radii = {finger: scale * params.finger_radius_m[finger] for finger in FINGER_NAMES}
    width = 2.0 * sum(radii.values()) + params.finger_gap_m * (len(FINGER_NAMES) - 1)
    centres: dict[str, float] = {}
    edge = -0.5 * width
    for finger in reversed(FINGER_NAMES):
        centres[finger] = edge + radii[finger]
        edge += 2.0 * radii[finger] + params.finger_gap_m
    return centres


def _hand(spec: GripSpec, side: str):
    return spec.lead if side == "lead" else spec.trail


def build_default_chain(
    spec: GripSpec,
    side: str,
    settings: PoseSettings | None = None,
    params: HandMeshParams = DEFAULT_HAND_MESH_PARAMS,
) -> KinematicChain:
    """Human golf chain scaled by ``spec.hand_size``.

    Preconditions: ``spec`` is a GripSpec and ``side`` is ``lead`` or ``trail``.
    Postcondition: every finger has a metacarpal plus three phalanges, and the
    thumb has a metacarpal plus two contact segments. Tangent-joint origins
    lie on the shaft at the finger station; they are not MCP hinge frames.
    """
    if not isinstance(spec, GripSpec):
        raise TypeError(f"spec must be a GripSpec, got {type(spec).__name__}")
    if not isinstance(params, HandMeshParams):
        raise TypeError(f"params must be HandMeshParams, got {type(params).__name__}")
    if side not in _SIDES:
        raise ValueError(f"side must be 'lead' or 'trail', got {side!r}")
    config = load_pose_settings() if settings is None else settings
    if not isinstance(config, PoseSettings):
        raise TypeError(f"settings must be PoseSettings, got {type(config).__name__}")
    scale = hand_mesh_scale(spec.hand_size.length_m)
    if not 0.5 <= scale <= 1.5:
        raise ValueError(f"hand scale {scale} is outside the mesh range [0.5, 1.5]")
    curl = curl_sign_for(spec, side)
    hand = _hand(spec, side)
    stack = _stack_z(scale, params)
    origin_z = spec.lead_below_butt_m
    if side == "trail":
        origin_z += spec.hand_spacing_m
    span = stack["index"] - stack["little"]
    if span <= 0.0:
        raise ValueError("finger stack must run from little toward the head")
    base = math.radians(hand.v_azimuth_deg)
    if spec.handedness == "left":
        base = -base
    base += curl * math.radians(params.mcp_from_v_deg)
    cross = math.radians(hand.club_cross_angle_deg) * config.cross_azimuth_scale
    joints: list[JointDescription] = []
    for finger in FINGER_NAMES:
        radius = scale * params.finger_radius_m[finger]
        lengths = tuple(scale * item for item in params.phalanx_lengths_m[finger])
        station = origin_z + stack[finger]
        fraction = (stack["index"] - stack[finger]) / span
        azimuth = base + curl * fraction * cross
        extra = fraction * config.just_below_pinky_m
        weight = config.arch_weight_for(finger)
        rest = (math.cos(azimuth), math.sin(azimuth), 0.0)
        meta_name = joint_name(side, finger, "metacarpal")
        joints.append(
            _joint(
                meta_name,
                None,
                finger,
                "metacarpal",
                "arch",
                (0.0, 0.0, station),
                rest,
                config.metacarpal_length_for(finger) * scale,
                radius,
                config.arch_gain_rad,
                station,
                azimuth,
                extra,
                weight,
            )
        )
        parent = meta_name
        for segment, length in zip(PHALANX_NAMES, lengths, strict=True):
            name = joint_name(side, finger, segment)
            joints.append(
                _joint(
                    name,
                    parent,
                    finger,
                    segment,
                    "tangent",
                    (0.0, 0.0, station),
                    rest,
                    length,
                    radius,
                    config.flexion_limit_rad(segment),
                    station,
                    azimuth,
                    extra,
                    weight,
                )
            )
            parent = name
    role: Literal["lead", "trail"] = "lead" if side == "lead" else "trail"
    thumb_azimuth = math.radians(spec.resolved_thumb_azimuth_deg(role))
    thumb_rest = (math.cos(thumb_azimuth), math.sin(thumb_azimuth), 0.0)
    thumb_radius = scale * params.thumb_radius_m
    thumb_segments = ("metacarpal", "proximal", "distal")
    thumb_parent: str | None = None
    for segment, length in zip(thumb_segments, config.thumb_lengths_m, strict=True):
        drive = "arch" if segment == "metacarpal" else "thumb"
        limit = 0.0 if segment == "metacarpal" else config.thumb_flex_limit_rad
        name = joint_name(side, "thumb", segment)
        joints.append(
            _joint(
                name,
                thumb_parent,
                "thumb",
                segment,
                drive,
                (0.0, 0.0, origin_z),
                thumb_rest,
                length,
                thumb_radius,
                limit if limit > 0.0 else config.arch_gain_rad,
                origin_z,
                thumb_azimuth,
                0.0,
                0.0,
            )
        )
        thumb_parent = name
    return KinematicChain(
        name=f"{side}_human",
        side=side,
        curl_sign=curl,
        joints=tuple(joints),
    )


def _joint(
    name: str,
    parent: str | None,
    finger: str,
    segment: str,
    drive: str,
    origin: tuple[float, float, float],
    rest: tuple[float, float, float],
    length: float,
    radius: float,
    limit_hi: float,
    station: float,
    azimuth: float,
    extra: float,
    weight: float,
) -> JointDescription:
    return JointDescription(
        name=name,
        parent=parent,
        finger=finger,
        segment=segment,
        drive=drive,
        origin_m=origin,
        axis=_SHAFT,
        rest_direction=rest,
        length_m=length,
        radius_m=radius,
        limit_lo_rad=0.0,
        limit_hi_rad=limit_hi,
        station_z_m=station,
        azimuth_rad=azimuth,
        clearance_extra_m=extra,
        arch_weight=weight,
    )


__all__ = [
    "JointDescription",
    "KinematicChain",
    "build_default_chain",
    "curl_sign_for",
    "joint_name",
]
