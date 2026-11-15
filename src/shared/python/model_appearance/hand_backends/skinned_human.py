"""Skinned MakeHuman hand on the grip (epic #15150, #15148).

MakeHuman's CC0 mesh, skeleton and skin weights, posed by joint rotation and
linear blend skinning. The pose follows Dieter's grip spec through the shared
:mod:`posing` module: a flat hand is laid beside the club, with the club under
the index finger's first segment and just below the base of the little finger,
and the digits then curl proximal to distal until the skin touches the grip.

This is the visual model only: contact physics stays on the capsule grip.
Every number comes from ``scripts/config/skinned_human_hand.json``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.hand_backends.hand_asset import (
    load_hand_cc0_asset,
)
from src.shared.python.model_appearance.hand_backends.correctives import (
    FoldRelaxSettings,
    PadConformSettings,
    conform_pads,
    relax_folds,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    FINGERS,
    HandJointAngles,
    HandSide,
)
from src.shared.python.model_appearance.hand_backends.posing import (
    CoupledJoint,
    CurlChain,
    CurlJoint,
    CurlSettings,
    FlatHandLandmarks,
    ObstacleDistance,
    capsule_union_distance,
    club_axis_in_hand,
    curl_to_contact,
    cylinder_distance,
    flat_hand_rotation,
    rigid,
    SeatLimits,
    seat_on_obstacle,
    wrap_to_contact,
)
from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    bone_transforms,
    flipped_face_ids,
    pose_vertices,
    rotation_about_axis,
    scale_asset,
    vertex_normals,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

Array = NDArray[np.float64]

BACKEND_NAME = "skinned_human"
SETTINGS_PATH = (
    Path(__file__).resolve().parents[5]
    / "scripts"
    / "config"
    / "skinned_human_hand.json"
)
_SEGMENTS = ("prox", "mid", "dist")
_JOINT_OF_SEGMENT = {"prox": "mcp", "mid": "pip", "dist": "dip"}
_THUMB_BONES = ("thumb_meta", "thumb_prox", "thumb_dist")
_EYE3 = np.eye(3)
_EYE4 = np.eye(4)
#: Virtual hinge at the thumb base that presses the thumb onto the grip.
THUMB_PRESS = "thumb_press"


@dataclass(frozen=True)
class SkinnedHumanSettings:
    """Everything tunable about the skinned hand (loaded from JSON)."""

    skin_rgb: tuple[float, float, float]
    breadth_band_m: float
    scale_limits: tuple[float, float]
    palm_contact_azimuth_offset_deg: float
    palm_centre_wrap_deg: float
    index_station_at_spec_zero: float
    seat_max_tilt_deg: float
    seat_max_travel_m: float
    seat_iterations: int
    finger_adduction_fraction: float
    curl: CurlSettings
    limits_deg: tuple[tuple[str, tuple[float, float]], ...]
    pip_per_mcp: float
    dip_per_pip: float
    wrap_scan_samples: int
    arch_passes: int
    arch_gain: tuple[tuple[str, float], ...]
    arch_limit_deg: tuple[tuple[str, float], ...]
    thumb_max_swing_deg: float
    thumb_max_twist_deg: float
    thumb_press_range_deg: tuple[float, float]
    thumb_mcp_limits_deg: tuple[float, float]
    thumb_ip_limits_deg: tuple[float, float]
    thumb_straight_ip_limit_deg: float
    pair_capsule_radius_scale: float
    pair_thumb_radius_scale: float
    pair_grip_fingers: tuple[str, ...]
    pair_max_spacing_adjust_m: float
    pair_capsule_margin_m: float
    frame_excluded_bones: tuple[str, ...]
    fold_relax: FoldRelaxSettings
    palm_pad: PadConformSettings
    wrist_deg: tuple[tuple[str, tuple[float, float]], ...]

    def limit_rad(self, joint: str) -> tuple[float, float]:
        low, high = dict(self.limits_deg)[joint]
        return math.radians(low), math.radians(high)


def _pair(raw: Any, name: str) -> tuple[float, float]:
    values = tuple(float(x) for x in raw)
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be two finite numbers")
    if values[0] > values[1]:
        raise ValueError(f"{name} must be ordered low, high")
    return values[0], values[1]


def _unit_interval(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _fingers(raw: Any) -> tuple[str, ...]:
    names = tuple(str(x) for x in raw)
    if not set(names) <= set(FINGERS):
        raise ValueError(f"grip_fingers must be among {FINGERS}")
    return names


def _non_negative(value: Any, name: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number >= 0.0):
        raise ValueError(f"{name} must be non-negative and finite")
    return number


def _fold_relax(raw: Mapping[str, Any]) -> FoldRelaxSettings:
    return FoldRelaxSettings(
        iterations=int(raw["iterations"]),
        rings=int(raw["rings"]),
        step=float(raw["step"]),
    )


def _palm_pad(raw: Mapping[str, Any]) -> PadConformSettings:
    return PadConformSettings(
        bones=tuple(str(bone) for bone in raw["bones"]),
        reach_m=float(raw["reach_m"]),
        clearance_m=float(raw["clearance_m"]),
        facing_full=float(raw["facing_full"]),
        smooth_iterations=int(raw["smooth_iterations"]),
        passes=int(raw["passes"]),
    )


def settings_from_mapping(raw: Mapping[str, Any]) -> SkinnedHumanSettings:
    """Validate a settings mapping. Raises ``ValueError``/``KeyError`` if bad."""
    rgb = tuple(_unit_interval(x, "skin_rgb") for x in raw["skin_rgb"])
    if len(rgb) != 3:
        raise ValueError("skin_rgb must be three numbers in [0, 1]")
    anthro: Mapping = raw["anthropometry"]
    place: Mapping = raw["placement"]
    pre: Mapping = raw["preshape"]
    curl: Mapping = raw["curl"]
    thumb: Mapping = raw["thumb"]
    pair: Mapping = raw["pair"]
    passes = int(curl["arch_passes"])
    if passes < 1:
        raise ValueError("arch_passes must be at least 1")
    for finger in FINGERS:
        if finger not in curl["arch_gain"] or finger not in curl["arch_limit_deg"]:
            raise ValueError(f"arch settings missing finger {finger!r}")
    for joint in _JOINT_OF_SEGMENT.values():
        if joint not in curl["limits_deg"]:
            raise ValueError(f"limits_deg missing joint {joint!r}")
    return SkinnedHumanSettings(
        skin_rgb=(rgb[0], rgb[1], rgb[2]),
        breadth_band_m=float(anthro["breadth_band_m"]),
        scale_limits=_pair(anthro["scale_limits"], "scale_limits"),
        palm_contact_azimuth_offset_deg=float(place["palm_contact_azimuth_offset_deg"]),
        palm_centre_wrap_deg=float(place["palm_centre_wrap_deg"]),
        index_station_at_spec_zero=_unit_interval(
            place["index_station_at_spec_zero"], "index_station_at_spec_zero"
        ),
        seat_max_tilt_deg=float(place["seat_max_tilt_deg"]),
        seat_max_travel_m=float(place["seat_max_travel_m"]),
        seat_iterations=int(place["seat_iterations"]),
        finger_adduction_fraction=_unit_interval(
            pre["finger_adduction_fraction"], "finger_adduction_fraction"
        ),
        curl=CurlSettings(
            step_rad=math.radians(float(curl["step_deg"])),
            bisection_steps=int(curl["bisection_steps"]),
            contact_tolerance_m=float(curl["contact_tolerance_m"]),
            penetration_allowance_m=float(curl["penetration_allowance_m"]),
        ),
        limits_deg=tuple(
            (k, _pair(v, f"limits_deg.{k}")) for k, v in curl["limits_deg"].items()
        ),
        pip_per_mcp=float(curl["synergy"]["pip_per_mcp"]),
        dip_per_pip=float(curl["synergy"]["dip_per_pip"]),
        wrap_scan_samples=int(curl["wrap_scan_samples"]),
        arch_passes=passes,
        arch_gain=tuple((k, float(v)) for k, v in curl["arch_gain"].items()),
        arch_limit_deg=tuple((k, float(v)) for k, v in curl["arch_limit_deg"].items()),
        thumb_max_swing_deg=float(thumb["max_swing_deg"]),
        thumb_max_twist_deg=float(thumb["max_twist_deg"]),
        thumb_press_range_deg=_pair(thumb["press_range_deg"], "press_range_deg"),
        thumb_mcp_limits_deg=_pair(thumb["mcp_limits_deg"], "mcp_limits_deg"),
        thumb_ip_limits_deg=_pair(thumb["ip_limits_deg"], "ip_limits_deg"),
        thumb_straight_ip_limit_deg=float(thumb["straight_ip_limit_deg"]),
        pair_capsule_radius_scale=_unit_interval(
            pair["capsule_radius_scale"], "capsule_radius_scale"
        ),
        pair_thumb_radius_scale=_unit_interval(
            pair["thumb_radius_scale"], "thumb_radius_scale"
        ),
        pair_grip_fingers=_fingers(pair["grip_fingers"]),
        pair_max_spacing_adjust_m=_non_negative(
            pair["max_spacing_adjust_m"], "max_spacing_adjust_m"
        ),
        pair_capsule_margin_m=_non_negative(
            pair["capsule_margin_m"], "capsule_margin_m"
        ),
        frame_excluded_bones=tuple(
            str(bone) for bone in raw["framing"]["excluded_bones"]
        ),
        fold_relax=_fold_relax(raw["correctives"]["fold_relax"]),
        palm_pad=_palm_pad(raw["correctives"]["palm_pad"]),
        wrist_deg=tuple(
            (
                side,
                (
                    float(raw["wrist"][side]["extension_deg"]),
                    float(raw["wrist"][side]["ulnar_deviation_deg"]),
                ),
            )
            for side in ("lead", "trail")
        ),
    )


def load_settings_mapping() -> dict[str, Any]:
    """The raw settings JSON (for callers that override a few values)."""
    if not SETTINGS_PATH.is_file():
        raise ValueError(f"missing settings: {SETTINGS_PATH}")
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def default_settings() -> SkinnedHumanSettings:
    return settings_from_mapping(load_settings_mapping())


# --------------------------------------------------------------------------
# Anthropometry
# --------------------------------------------------------------------------


def asset_hand_length_m(asset: SkinnedHandAsset) -> float:
    """Wrist joint to the tip of the middle finger skin, along the distal axis."""
    tip = asset.dominant_vertex_ids(("middle_dist",))
    return float(asset.rest_vertices[tip, 1].max())


def asset_hand_breadth_m(asset: SkinnedHandAsset, band_m: float) -> float:
    """Skin breadth across the knuckles (index to little), thumb excluded."""
    knuckle_y = float(
        np.mean([asset.bone_heads[asset.bone_index(f"{f}_prox")][1] for f in FINGERS])
    )
    verts = asset.rest_vertices
    band = np.flatnonzero(np.abs(verts[:, 1] - knuckle_y) < band_m)
    band = np.setdiff1d(band, asset.influenced_vertex_ids(_THUMB_BONES))
    if len(band) == 0:
        raise ValueError("no skin found across the knuckles")
    return float(np.ptp(verts[band, 0]))


@lru_cache(maxsize=8)
def _sized_asset(
    length_m: float, breadth_m: float, settings: SkinnedHumanSettings
) -> SkinnedHandAsset:
    base = load_hand_cc0_asset().asset
    distal = length_m / asset_hand_length_m(base)
    radial = breadth_m / asset_hand_breadth_m(base, settings.breadth_band_m)
    low, high = settings.scale_limits
    for name, value in (("length", distal), ("breadth", radial)):
        if not low <= value <= high:
            raise ValueError(f"hand {name} scale {value:.2f} outside {low}..{high}")
    return scale_asset(base, (radial, distal, radial))


# --------------------------------------------------------------------------
# The posable hand
# --------------------------------------------------------------------------


class _GripHand:
    """A sized right hand that the shared posing module can curl.

    Hinge angles are anatomical flexion (0 = straight digit). ``fixed`` holds
    rotations applied on top of the hinges (finger adduction, thumb aim).
    """

    def __init__(self, asset: SkinnedHandAsset, settings: SkinnedHumanSettings) -> None:
        self.asset = asset
        self.settings = settings
        self.placement = np.eye(4)
        self.press_axis = np.array([1.0, 0.0, 0.0])
        self._index = {name: i for i, name in enumerate(asset.bone_names)}
        self._parents = [int(p) for p in asset.bone_parents]
        axes = asset.flex_axes
        k = np.zeros((len(axes), 3, 3))
        k[:, 0, 1], k[:, 0, 2] = -axes[:, 2], axes[:, 1]
        k[:, 1, 0], k[:, 1, 2] = axes[:, 2], -axes[:, 0]
        k[:, 2, 0], k[:, 2, 1] = -axes[:, 1], axes[:, 0]
        self._hinge_k, self._hinge_kk = k, k @ k
        # Linear blend skinning as one matrix product: weights times [v, 1].
        homogeneous = np.hstack(
            [asset.rest_vertices, np.ones((len(asset.rest_vertices), 1))]
        )
        dense = asset.dense_weights()
        self._blend = (dense[:, :, None] * homogeneous[:, None, :]).reshape(
            len(dense), -1
        )
        self._segment: dict[str, NDArray[np.int64]] = {}
        self.fixed: dict[str, Array] = self._adduction()
        self._moved: dict[str, NDArray[np.int64]] = {}

    def _adduction(self) -> dict[str, Array]:
        """Bring the rest finger splay toward the middle finger's direction."""
        fraction = self.settings.finger_adduction_fraction
        reference = self.direction("middle_prox")
        palmar = np.array([0.0, 0.0, 1.0])
        out: dict[str, Array] = {}
        for finger in FINGERS:
            d = self.direction(f"{finger}_prox")
            splay = math.atan2(d[0], d[1]) - math.atan2(reference[0], reference[1])
            out[f"{finger}_prox"] = rotation_about_axis(palmar, fraction * splay)
        return out

    def direction(self, bone: str) -> Array:
        i = self.asset.bone_index(bone)
        d = self.asset.bone_tails[i] - self.asset.bone_heads[i]
        return d / np.linalg.norm(d)

    def _hinge(self, i: int, angle: float) -> Array:
        """Rodrigues rotation about bone ``i``'s flexion axis (precomputed)."""
        k, kk = self._hinge_k[i], self._hinge_kk[i]
        return _EYE3 + math.sin(angle) * k + (1.0 - math.cos(angle)) * kk

    def rotations(self, angles: Mapping[str, float]) -> dict[str, Array]:
        asset = self.asset
        out = dict(self.fixed)
        for name, angle in angles.items():
            if name == THUMB_PRESS:
                continue
            i = self._index[name]
            delta = angle - float(asset.rest_flexion_rad[i])
            if delta == 0.0:
                continue
            hinge = self._hinge(i, delta)
            out[name] = out[name] @ hinge if name in out else hinge
        if THUMB_PRESS in angles:
            press = rotation_about_axis(self.press_axis, angles[THUMB_PRESS])
            out["thumb_meta"] = press @ out.get("thumb_meta", _EYE3)
        return out

    def transforms(self, angles: Mapping[str, float]) -> Array:
        """Forward kinematics; equal to ``skinning.bone_transforms`` of
        :meth:`rotations` (a unit test holds the two together)."""
        rotations = self.rotations(angles)
        heads, parents = self.asset.bone_heads, self._parents
        out = np.empty((len(parents), 4, 4))
        for i, name in enumerate(self.asset.bone_names):
            rot = rotations.get(name)
            if rot is None:
                out[i] = _EYE4 if parents[i] < 0 else out[parents[i]]
                continue
            local = _EYE4.copy()
            local[:3, :3] = rot
            local[:3, 3] = heads[i] - rot @ heads[i]
            out[i] = local if parents[i] < 0 else out[parents[i]] @ local
        return out

    def points(
        self, angles: Mapping[str, float], ids: NDArray[np.int64] | None = None
    ) -> Array:
        """Skinned vertices in the grip frame (same as ``pose_vertices``);
        only vertices ``ids`` when given."""
        stacked = self.transforms(angles)[:, :3, :].transpose(0, 2, 1).reshape(-1, 3)
        blend = self._blend if ids is None else self._blend[ids]
        local = blend @ stacked
        return local @ self.placement[:3, :3].T + self.placement[:3, 3]

    def segment_ids(self, bone: str) -> NDArray[np.int64]:
        if bone not in self._segment:
            names = ("thumb_prox", "thumb_dist") if bone == THUMB_PRESS else (bone,)
            self._segment[bone] = self.asset.dominant_vertex_ids(names)
        return self._segment[bone]

    def moved_ids(self, bone: str) -> NDArray[np.int64]:
        if bone not in self._moved:
            root = "thumb_meta" if bone == THUMB_PRESS else bone
            names = [self.asset.bone_names[i] for i in self.asset.descendants(root)]
            self._moved[bone] = self.asset.influenced_vertex_ids(names)
        return self._moved[bone]

    def joint_position(
        self, bone: str, angles: Mapping[str, float], *, tail: bool = False
    ) -> Array:
        """Posed head (or tail) of ``bone`` in the grip frame."""
        i = self.asset.bone_index(bone)
        rest = self.asset.bone_tails[i] if tail else self.asset.bone_heads[i]
        t = self.transforms(angles)[i]
        local = t[:3, :3] @ rest + t[:3, 3]
        return self.placement[:3, :3] @ local + self.placement[:3, 3]


def _straight_angles() -> dict[str, float]:
    angles = {f"{f}_{s}": 0.0 for f in FINGERS for s in _SEGMENTS}
    angles.update({f"meta_{f}": 0.0 for f in FINGERS})
    return angles


def flat_hand_landmarks(hand: _GripHand) -> FlatHandLandmarks:
    """Landmarks of the flat (straight, adducted) hand in its rest frame."""
    t = hand.transforms(_straight_angles())
    asset = hand.asset

    def head(bone: str) -> Array:
        i = asset.bone_index(bone)
        return t[i][:3, :3] @ asset.bone_heads[i] + t[i][:3, 3]

    return FlatHandLandmarks(
        index_mcp=head("index_prox"),
        index_pip=head("index_mid"),
        little_mcp=head("little_prox"),
        radial=np.array([1.0, 0.0, 0.0]),
        distal=np.array([0.0, 1.0, 0.0]),
        palmar=np.array([0.0, 0.0, 1.0]),
    )


@dataclass(frozen=True)
class _RightHandTargets:
    """Grip-frame targets for the right-hand build of one side."""

    palm_azimuth_rad: float
    thumb_azimuth_rad: float
    cross_angle_deg: float
    index_station: float
    pinky_palm_depth_m: float
    thumb_straightness: float
    mirrored: bool


def _targets(
    spec: GripSpec, side: HandSide, settings: SkinnedHumanSettings
) -> _RightHandTargets:
    """Express one hand's spec as targets for a right hand.

    Azimuth 0 is the top of the grip and increases away from the target. For a
    right-handed golfer the trail hand is the right hand; the lead hand is
    built as its mirror image (``y -> -y``), so its targets are negated here.
    """
    hand = spec.lead if side == "lead" else spec.trail
    v_azimuth = hand.v_azimuth_deg if side == "trail" else -hand.v_azimuth_deg
    is_right = (side == "trail") == (spec.handedness == "right")
    station = hand.index_contact_station or settings.index_station_at_spec_zero
    return _RightHandTargets(
        palm_azimuth_rad=math.radians(
            v_azimuth
            + settings.palm_contact_azimuth_offset_deg
            + settings.palm_centre_wrap_deg
        ),
        thumb_azimuth_rad=math.radians(-hand.thumb_azimuth_deg),
        cross_angle_deg=hand.club_cross_angle_deg,
        index_station=station,
        pinky_palm_depth_m=hand.pinky_side_palm_depth_m,
        thumb_straightness=hand.thumb_straightness,
        mirrored=not is_right,
    )


def _seat_flat_hand(
    hand: _GripHand,
    targets: _RightHandTargets,
    obstacle: ObstacleDistance,
    arch: Mapping[str, float],
    palm_obstacle: ObstacleDistance | None = None,
) -> None:
    """Lay the flat hand beside the club and rest it on the grip (rigidly).

    The straight fingers only have to clear ``obstacle`` (the grip); the palm
    must also clear ``palm_obstacle`` (the grip and the other hand) if given.
    """
    settings = hand.settings
    marks = flat_hand_landmarks(hand)
    point, direction = club_axis_in_hand(
        marks,
        cross_angle_deg=targets.cross_angle_deg,
        index_station=targets.index_station,
        pinky_palm_depth_m=targets.pinky_palm_depth_m,
    )
    rot = flat_hand_rotation(marks, direction, targets.palm_azimuth_rad)
    centre = 0.5 * (marks.index_mcp + marks.little_mcp)
    on_axis = point + ((centre - point) @ direction) * direction
    hand.placement = rigid(rot, -rot @ on_axis)
    pts = hand.points({**_straight_angles(), **arch})
    keep = np.setdiff1d(
        np.arange(len(pts)), hand.asset.influenced_vertex_ids(_THUMB_BONES)
    )
    c, s = math.cos(targets.palm_azimuth_rad), math.sin(targets.palm_azimuth_rad)
    approach = np.array([-c, -s, 0.0])
    correction, _, _ = seat_on_obstacle(
        pts[keep],
        obstacle,
        approach=approach,
        tilt_axis=np.array([-s, c, 0.0]),
        tilt_pivot=np.zeros(3),
        split_coordinate=pts[keep, 2],
        limits=SeatLimits(
            max_tilt_rad=math.radians(settings.seat_max_tilt_deg),
            max_travel_m=settings.seat_max_travel_m,
            iterations=settings.seat_iterations,
        ),
    )
    hand.placement = correction @ hand.placement
    if palm_obstacle is None:
        return
    # Back the seated hand straight off until its palm clears the other hand.
    palm = hand.points({**_straight_angles(), **arch}, _palm_ids(hand.asset))
    allowance = settings.curl.penetration_allowance_m

    def clear(lift: float) -> bool:
        return float(palm_obstacle(palm - lift * approach).min()) >= -allowance

    low, high = 0.0, settings.seat_max_travel_m
    if clear(low):
        return
    for _ in range(settings.seat_iterations):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if clear(mid) else (mid, high)
    hand.placement = rigid(np.eye(3), -high * approach) @ hand.placement


def _palm_ids(asset: SkinnedHandAsset) -> NDArray[np.int64]:
    """Skin of the palm and back of the hand (wrist and metacarpals)."""
    return asset.dominant_vertex_ids(("wrist",) + tuple(f"meta_{f}" for f in FINGERS))


def _finger_chains(settings: SkinnedHumanSettings) -> tuple[CurlChain, ...]:
    """Finger chains, knuckle first.

    While a joint closes, the joints beyond it follow at the synergy gains, so
    a finger curls as it closes instead of swinging in straight and landing
    on its tip. Each following joint then carries on from where it was left.
    """
    pip_gain, dip_gain = settings.pip_per_mcp, settings.dip_per_pip
    chains = []
    for finger in FINGERS:
        limits = {s: settings.limit_rad(_JOINT_OF_SEGMENT[s]) for s in _SEGMENTS}
        prox, mid, dist = (f"{finger}_{s}" for s in _SEGMENTS)
        joints = (
            CurlJoint(
                prox,
                *limits["prox"],
                limits["prox"][0],
                (
                    CoupledJoint(mid, pip_gain, limits["mid"][1]),
                    CoupledJoint(dist, pip_gain * dip_gain, limits["dist"][1]),
                ),
            ),
            CurlJoint(
                mid,
                *limits["mid"],
                limits["mid"][0],
                (CoupledJoint(dist, dip_gain, limits["dist"][1]),),
            ),
            CurlJoint(dist, *limits["dist"], limits["dist"][0]),
        )
        chains.append(CurlChain(finger, joints, must_touch=(mid, dist)))
    return tuple(chains)


def _arch_from_knuckles(
    angles: Mapping[str, float], settings: SkinnedHumanSettings
) -> dict[str, float]:
    """Metacarpal arch coupled to knuckle flexion, so the palm cups."""
    gains, limits = dict(settings.arch_gain), dict(settings.arch_limit_deg)
    return {
        f"meta_{f}": min(
            math.radians(limits[f]), max(0.0, gains[f] * angles[f"{f}_prox"])
        )
        for f in FINGERS
    }


def _clamped_rotation(source: Array, target: Array, max_rad: float) -> Array:
    """Smallest rotation turning ``source`` toward ``target``, at most ``max_rad``."""
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    axis = np.cross(a, b)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.eye(3)
    return rotation_about_axis(axis, min(max_rad, math.atan2(norm, float(a @ b))))


def _bone_skin_radius(asset: SkinnedHandAsset, bone_name: str) -> float:
    """Median distance from a bone to the skin it mostly moves."""
    i = asset.bone_index(bone_name)
    bone = asset.bone_tails[i] - asset.bone_heads[i]
    rel = asset.rest_vertices[asset.dominant_vertex_ids((bone_name,))]
    rel = rel - asset.bone_heads[i]
    along = np.clip(rel @ bone / (bone @ bone), 0.0, 1.0)
    return float(np.median(np.linalg.norm(rel - along[:, None] * bone, axis=1)))


def _thumb_half_thickness(asset: SkinnedHandAsset) -> float:
    return _bone_skin_radius(asset, "thumb_dist")


def _aim_thumb(
    hand: _GripHand,
    targets: _RightHandTargets,
    angles: Mapping[str, float],
    grip_radius_m: float,
) -> None:
    """Swing and twist the thumb base so the thumb lies along the shaft.

    The thumb tip goes to the thumb azimuth on the grip, as far toward the
    head as the thumb reaches, and the pad is turned to face the grip. Both
    are rotations of the thumb's base joint; nothing else moves.
    """
    settings, asset = hand.settings, hand.asset
    # Aim from the thumb's rest shape, whatever thumb angles come later.
    angles = {k: v for k, v in angles.items() if not k.startswith("thumb")}
    grip_from_rest = hand.placement[:3, :3]
    base = hand.joint_position("thumb_meta", angles)
    tip = hand.joint_position("thumb_dist", angles, tail=True)
    rho = grip_radius_m + _thumb_half_thickness(asset)
    c, s = math.cos(targets.thumb_azimuth_rad), math.sin(targets.thumb_azimuth_rad)
    reach = float(np.linalg.norm(tip - base))
    lateral = float(np.hypot(rho * c - base[0], rho * s - base[1]))
    rise = math.sqrt(max(reach**2 - lateral**2, 0.0))
    goal = np.array([rho * c, rho * s, base[2] + rise])
    swing = _clamped_rotation(
        tip - base, goal - base, math.radians(settings.thumb_max_swing_deg)
    )
    thumb_dir = swing @ (tip - base)
    thumb_dir /= np.linalg.norm(thumb_dir)
    new_tip = base + swing @ (tip - base)

    def perpendicular(v: Array) -> Array:
        w = v - (v @ thumb_dir) * thumb_dir
        return w / max(float(np.linalg.norm(w)), 1e-12)

    # The pad is on the flexion side of the thumb: turn it to face the grip axis.
    i_prox = asset.bone_index("thumb_prox")
    pad_rest = np.cross(asset.flex_axes[i_prox], hand.direction("thumb_prox"))
    pad = perpendicular(swing @ grip_from_rest @ pad_rest)
    inward = perpendicular(-np.array([new_tip[0], new_tip[1], 0.0]))
    twist_angle = math.atan2(
        float(np.cross(pad, inward) @ thumb_dir), float(pad @ inward)
    )
    limit = math.radians(settings.thumb_max_twist_deg)
    twist = rotation_about_axis(thumb_dir, max(-limit, min(limit, twist_angle)))
    aim_rest = grip_from_rest.T @ twist @ swing @ grip_from_rest
    hand.fixed["thumb_meta"] = aim_rest @ hand.fixed.get("thumb_meta", np.eye(3))
    press = np.cross(thumb_dir, inward)
    hand.press_axis = grip_from_rest.T @ (press / np.linalg.norm(press))


def _thumb_chain(
    settings: SkinnedHumanSettings, targets: _RightHandTargets
) -> CurlChain:
    low, high = (math.radians(v) for v in settings.thumb_press_range_deg)
    mcp_low, mcp_high = (math.radians(v) for v in settings.thumb_mcp_limits_deg)
    ip_low, ip_high = (math.radians(v) for v in settings.thumb_ip_limits_deg)
    straight = math.radians(settings.thumb_straight_ip_limit_deg)
    blend = targets.thumb_straightness
    mcp_high = max(mcp_low, blend * straight + (1.0 - blend) * mcp_high)
    ip_high = max(ip_low, blend * straight + (1.0 - blend) * ip_high)
    return CurlChain(
        "thumb",
        (
            CurlJoint(THUMB_PRESS, low, high, low),
            CurlJoint("thumb_prox", mcp_low, mcp_high, mcp_low),
            CurlJoint("thumb_dist", ip_low, ip_high, ip_low),
        ),
        must_touch=("thumb_prox", "thumb_dist"),
    )


def _bend_wrist(hand: _GripHand, side: HandSide) -> None:
    """Turn the forearm about the wrist joint; the hand stays where it is.

    Extension tips the forearm toward the back of the hand, ulnar deviation
    toward the little-finger side (right-hand rest frame: x radial, z palmar).
    """
    extension, ulnar = dict(hand.settings.wrist_deg)[side]
    turn = rotation_about_axis(
        np.array([0.0, 0.0, 1.0]), -math.radians(ulnar)
    ) @ rotation_about_axis(np.array([1.0, 0.0, 0.0]), math.radians(extension))
    asset = hand.asset
    elbow = asset.bone_heads[asset.bone_index("forearm")]
    wrist = asset.bone_heads[asset.bone_index("wrist")]
    hand.fixed["forearm"] = turn
    hand.fixed["wrist"] = turn.T
    # The forearm bone pivots at its elbow end: slide the hand back so the
    # wrist joint, not the elbow, is the point that stays put.
    slip = (elbow - wrist) - turn @ (elbow - wrist)
    hand.placement = hand.placement @ rigid(np.eye(3), -slip)


def _wrap(
    hand: _GripHand,
    chains: Sequence[CurlChain],
    obstacle: ObstacleDistance,
    start: Mapping[str, float],
) -> dict[str, float]:
    """Curl ``chains`` to contact, with the wrap search if it is switched on."""
    settings = hand.settings
    if settings.wrap_scan_samples >= 2:
        return wrap_to_contact(
            hand,
            chains,
            obstacle,
            settings.curl,
            start,
            samples=settings.wrap_scan_samples,
        ).angles
    return curl_to_contact(hand, chains, obstacle, settings.curl, start).angles


@dataclass(frozen=True)
class _SolvedHand:
    hand: _GripHand
    angles: dict[str, float]
    targets: _RightHandTargets
    obstacle: ObstacleDistance


def _segment_gaps(
    hand: _GripHand, skin: Array, obstacle: ObstacleDistance
) -> dict[str, float]:
    """Clearance between the grip and the skin of every phalanx and the palm.

    ``skin`` is every vertex of the hand in the grip frame.
    """
    d = obstacle(skin)
    gaps = {
        f"{f}_{s}": float(d[hand.segment_ids(f"{f}_{s}")].min())
        for f in FINGERS
        for s in _SEGMENTS
    }
    for bone in ("thumb_prox", "thumb_dist"):
        gaps[bone] = float(d[hand.segment_ids(bone)].min())
    gaps["palm"] = float(d[_palm_ids(hand.asset)].min())
    gaps["hand"] = float(d.min())
    return gaps


def _solve_right_hand(
    spec: GripSpec,
    side: HandSide,
    grip_radius_m: float,
    settings: SkinnedHumanSettings,
    given: Mapping[str, float] | None,
    obstacle: ObstacleDistance | None = None,
) -> _SolvedHand:
    asset = _sized_asset(spec.hand_size.length_m, spec.hand_size.breadth_m, settings)
    hand = _GripHand(asset, settings)
    targets = _targets(spec, side, settings)
    grip = cylinder_distance(grip_radius_m)
    other, obstacle = obstacle, obstacle or grip
    arch = {f"meta_{f}": 0.0 for f in FINGERS}
    if given is not None:
        arch = {k: float(given.get(k, 0.0)) for k in arch}
        _seat_flat_hand(hand, targets, grip, arch, other)
        posed = {**_straight_angles(), **{k: float(v) for k, v in given.items()}}
        _aim_thumb(hand, targets, posed, grip_radius_m)
        _bend_wrist(hand, side)
        return _SolvedHand(hand, posed, targets, obstacle)
    chains = _finger_chains(settings)
    angles: dict[str, float] = {}
    for _ in range(settings.arch_passes):
        _seat_flat_hand(hand, targets, grip, arch, other)
        start = {**_straight_angles(), **arch}
        angles = curl_to_contact(hand, chains, obstacle, settings.curl, start).angles
        arch = _arch_from_knuckles(angles, settings)
    _seat_flat_hand(hand, targets, grip, arch, other)
    start = {**_straight_angles(), **arch}
    angles = _wrap(hand, chains, obstacle, start)
    _aim_thumb(hand, targets, angles, grip_radius_m)
    angles = _wrap(hand, (_thumb_chain(settings, targets),), obstacle, angles)
    _bend_wrist(hand, side)
    return _SolvedHand(hand, angles, targets, obstacle)


def _check_inputs(grip_spec: GripSpec, side: str, grip_radius_m: float) -> None:
    if not isinstance(grip_spec, GripSpec):
        raise TypeError("grip_spec must be a GripSpec")
    if side not in ("lead", "trail"):
        raise ValueError("side must be 'lead' or 'trail'")
    if isinstance(grip_radius_m, bool) or not isinstance(grip_radius_m, (int, float)):
        raise TypeError("grip_radius_m must be a number")
    if not (math.isfinite(grip_radius_m) and grip_radius_m > 0.0):
        raise ValueError("grip_radius_m must be positive and finite")


def solve_grip_joint_angles(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float,
    settings: SkinnedHumanSettings | None = None,
) -> HandJointAngles:
    """Curl-to-contact joint angles of one hand for this grip spec."""
    _check_inputs(grip_spec, side, grip_radius_m)
    solved = _solve_right_hand(
        grip_spec, side, grip_radius_m, settings or default_settings(), None
    )
    return HandJointAngles.from_mapping(solved.angles)


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    settings: SkinnedHumanSettings | None = None,
) -> HandVisualResult:
    """Skinned human hand mesh in the hand-grip frame.

    With ``joint_angles=None`` the hand is curled to contact for this spec;
    explicit angles are applied as given (the hand is still seated flat on the
    grip first, and may then penetrate it if the angles say so).

    Preconditions: ``grip_spec`` a :class:`GripSpec`; ``side`` lead or trail;
    ``grip_radius_m`` positive and finite. Postconditions: finite vertices;
    outward-wound closed mesh; unit vertex normals; the hand is rolled about
    the shaft by ``HandGripSpec.shaft_roll_deg``.
    """
    _check_inputs(grip_spec, side, grip_radius_m)
    if joint_angles is not None and not isinstance(joint_angles, HandJointAngles):
        raise TypeError("joint_angles must be HandJointAngles or None")
    settings = settings or default_settings()
    given = None if joint_angles is None else joint_angles.as_mapping()
    solved = _solve_right_hand(grip_spec, side, grip_radius_m, settings, given)
    return _result(solved, grip_spec, side, grip_radius_m, settings, 0.0)


def _club_from_build(
    spec: GripSpec, side: HandSide, mirrored: bool, z_m: float
) -> Array:
    """Right-hand build frame to the club frame: mirror, shaft roll, station."""
    frame = np.diag([1.0, -1.0 if mirrored else 1.0, 1.0, 1.0])
    roll_deg = resolved_shaft_roll_deg(spec, side)
    roll = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, roll_deg)[:3, :3]
    return rigid(roll, np.array([0.0, 0.0, z_m])) @ frame


def _corrected_skin(
    solved: _SolvedHand, transforms: Array, settings: SkinnedHumanSettings
) -> Array:
    """Skinned vertices in the grip frame, with the corrective shapes applied.

    Linear blend skinning first; then the palm pad settles onto the obstacle
    and the skin around any folded triangle relaxes (see ``correctives``).
    With both switched off this is exactly ``pose_vertices``.
    """
    hand = solved.hand
    asset, placement = hand.asset, hand.placement
    rotation, offset = placement[:3, :3], placement[:3, 3]
    skin = pose_vertices(asset, transforms) @ rotation.T + offset
    unknown = set(settings.palm_pad.bones) - set(asset.bone_names)
    if unknown:
        raise ValueError(f"correctives.palm_pad.bones names no bone: {sorted(unknown)}")
    skin = conform_pads(
        skin,
        asset.faces,
        asset.dominant_vertex_ids(settings.palm_pad.bones),
        solved.obstacle,
        settings.palm_pad,
        lambda v: vertex_normals(v, asset.faces),
    )
    return relax_folds(
        skin,
        asset.faces,
        lambda v: flipped_face_ids(asset, transforms, (v - offset) @ rotation),
        settings.fold_relax,
    )


def _result(
    solved: _SolvedHand,
    spec: GripSpec,
    side: HandSide,
    grip_radius_m: float,
    settings: SkinnedHumanSettings,
    z_m: float,
) -> HandVisualResult:
    hand = solved.hand
    transforms = hand.transforms(solved.angles)
    club_from_grip = _club_from_build(spec, side, solved.targets.mirrored, z_m)
    faces = hand.asset.faces.copy()
    if solved.targets.mirrored:
        faces = faces[:, ::-1].copy()
    skin = _corrected_skin(solved, transforms, settings)
    vertices = skin @ club_from_grip[:3, :3].T + club_from_grip[:3, 3]
    require(bool(np.isfinite(vertices).all()), "skinned vertices must be finite")
    mesh = Mesh(vertices, faces)
    ensure(mesh.volume() > 0.0, "skinned hand must stay outward wound")
    unknown = set(settings.frame_excluded_bones) - set(hand.asset.bone_names)
    if unknown:
        raise ValueError(f"framing.excluded_bones names no bone: {sorted(unknown)}")
    focus = np.ones(len(vertices), dtype=bool)
    focus[hand.asset.dominant_vertex_ids(settings.frame_excluded_bones)] = False
    return HandVisualResult(
        backend=BACKEND_NAME,
        side=side,
        mesh=mesh,
        skin_rgb=settings.skin_rgb,
        vertex_normals=vertex_normals(vertices, faces),
        joint_angles=HandJointAngles.from_mapping(solved.angles),
        contact_gaps_m=_segment_gaps(hand, skin, solved.obstacle),
        grip_radius_m=float(grip_radius_m),
        bone_transforms=transforms,
        frame_from_rest=club_from_grip @ hand.placement,
        rest_asset=hand.asset,
        frame_focus=focus,
    )


def register() -> None:
    from src.shared.python.model_appearance.hand_backends.registry import (
        HandVisualBackend,
        register_hand_visual_backend,
    )

    register_hand_visual_backend(
        HandVisualBackend(name=BACKEND_NAME, builder=build_hand_visual)
    )
