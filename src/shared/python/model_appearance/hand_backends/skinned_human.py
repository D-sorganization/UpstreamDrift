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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
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
    PressSettings,
    lift_out,
    press_flat,
    sharpened_faces,
    relax_folds,
    yield_to_points,
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
    skin_distance,
    wrap_to_contact,
)
from src.shared.python.model_appearance.hand_backends.skinned_targets import (
    RightHandTargets as _RightHandTargets,
    scanned_turn as _scanned_turn,
    targets_for as _targets,
    turned as _turned,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    JOINT_OF_SEGMENT as _JOINT_OF_SEGMENT,
    SETTINGS_PATH as SETTINGS_PATH,
    SkinnedHumanSettings as SkinnedHumanSettings,
    default_settings as default_settings,
    load_settings_mapping as load_settings_mapping,
    settings_from_mapping as settings_from_mapping,
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
from src.shared.python.model_appearance.hand_backends.subdivision import (
    original_resolution,
    sharp_edges,
    subdivide_asset,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

Array = NDArray[np.float64]

BACKEND_NAME = "skinned_human"
_SEGMENTS = ("prox", "mid", "dist")
_BREADTH_FIT_PASSES = 2
_THUMB_BONES = ("thumb_meta", "thumb_prox", "thumb_dist")
#: Side closing: each finger and the neighbour it closes on, middle outward.
_SIDE_ORDER = (("ring", "middle"), ("little", "ring"), ("index", "middle"))
_EYE3 = np.eye(3)
_EYE4 = np.eye(4)
#: Virtual hinge at the thumb base that presses the thumb onto the grip.
THUMB_PRESS = "thumb_press"


def _bone_mask(
    asset: SkinnedHandAsset, bones: Sequence[str], name: str
) -> NDArray[np.bool_]:
    """Per-vertex flag: the strongest influence is one of ``bones``."""
    unknown = set(bones) - set(asset.bone_names)
    if unknown:
        raise ValueError(f"{name} names no bone: {sorted(unknown)}")
    mask = np.zeros(len(asset.rest_vertices), dtype=bool)
    mask[asset.dominant_vertex_ids(bones)] = True
    return mask


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
def _sized_assets(
    length_m: float, breadth_m: float, settings: SkinnedHumanSettings
) -> tuple[SkinnedHandAsset, SkinnedHandAsset]:
    """``(solve, skin)`` assets scaled to the hand size.

    ``skin`` is the mesh that is drawn: subdivided as set, then scaled.
    ``solve`` is the same hand at the source resolution (the skin's original
    vertices), which is what the grip is solved on: a contact found on it is a
    contact of the skin. With no subdivision the two are the same object.
    """
    source = load_hand_cc0_asset().asset
    # Edges that are sharp in the source (the nails) stay sharp when subdivided.
    creases = sharp_edges(
        source.rest_vertices,
        source.faces,
        settings.crease_min_fold_deg,
        _bone_mask(source, settings.crease_bones, "mesh.creases.bones"),
    )
    base = subdivide_asset(source, settings.subdivision_levels, creases)
    distal = length_m / asset_hand_length_m(base)
    radial = breadth_m / asset_hand_breadth_m(base, settings.breadth_band_m)
    low, high = settings.scale_limits
    for name, value in (("length", distal), ("breadth", radial)):
        if not low <= value <= high:
            raise ValueError(f"hand {name} scale {value:.2f} outside {low}..{high}")
    skin = scale_asset(base, (radial, distal, radial))
    # Lengthening the hand moves skin in or out of the knuckle band the
    # breadth is measured in, so the breadth is corrected on the scaled hand.
    for _ in range(_BREADTH_FIT_PASSES):
        radial *= breadth_m / asset_hand_breadth_m(skin, settings.breadth_band_m)
        skin = scale_asset(base, (radial, distal, radial))
    return original_resolution(skin, source), skin


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
        # Rigid moves applied to a bone before its own rotation (palm cup).
        self.offsets: dict[str, Array] = {}
        # Sideways swing of a finger at its knuckle, radians (side closing).
        self.side: dict[str, float] = {}
        self._side_turn: dict[str, Array] = {}
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

    def swing_sideways(self, bone: str | None, swing: float = 0.0) -> None:
        """Swing finger ``bone`` sideways at its knuckle by ``swing`` radians,
        about the axis across its flexion axis. ``None`` clears every swing."""
        if bone is None:
            self.side.clear()
            self._side_turn.clear()
            return
        axis = np.cross(self.asset.flex_axes[self._index[bone]], self.direction(bone))
        self.side[bone] = swing
        self._side_turn[bone] = rotation_about_axis(axis / np.linalg.norm(axis), swing)

    def _hinge(self, i: int, angle: float) -> Array:
        """Rodrigues rotation about bone ``i``'s flexion axis (precomputed)."""
        k, kk = self._hinge_k[i], self._hinge_kk[i]
        return _EYE3 + math.sin(angle) * k + (1.0 - math.cos(angle)) * kk

    def rotations(self, angles: Mapping[str, float]) -> dict[str, Array]:
        asset = self.asset
        out = dict(self.fixed)
        for name, angle in angles.items():
            if name == THUMB_PRESS or name.endswith(_SIDE_SUFFIX):
                continue
            i = self._index[name]
            delta = angle - float(asset.rest_flexion_rad[i])
            if delta == 0.0:
                continue
            hinge = self._hinge(i, delta)
            out[name] = out[name] @ hinge if name in out else hinge
        for name, turn in self._side_turn.items():
            out[name] = out[name] @ turn if name in out else turn
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
            rot, offset = rotations.get(name), self.offsets.get(name)
            if rot is None and offset is None:
                out[i] = _EYE4 if parents[i] < 0 else out[parents[i]]
                continue
            local = _EYE4.copy()
            if rot is not None:
                local[:3, :3] = rot
                local[:3, 3] = heads[i] - rot @ heads[i]
            if offset is not None:
                local = offset @ local
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


def _finger_chains(
    settings: SkinnedHumanSettings, index_knuckle_rad: float = 0.0
) -> tuple[CurlChain, ...]:
    """Finger chains, knuckle first.

    The forefinger's knuckle starts at, and never opens below,
    ``index_knuckle_rad`` (the flexion the hand was seated with).

    While a joint closes, the joints beyond it follow at the synergy gains, so
    a finger curls as it closes instead of swinging in straight and landing
    on its tip. Each following joint then carries on from where it was left.
    """
    pip_gain, dip_gain = settings.pip_per_mcp, settings.dip_per_pip
    chains = []
    for finger in FINGERS:
        limits = {s: settings.limit_rad(_JOINT_OF_SEGMENT[s]) for s in _SEGMENTS}
        prox, mid, dist = (f"{finger}_{s}" for s in _SEGMENTS)
        if finger == "index":
            low = min(max(limits["prox"][0], index_knuckle_rad), limits["prox"][1])
            limits["prox"] = (low, limits["prox"][1])
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
    hand.fixed.pop("thumb_meta", None)  # each aim starts from the rest thumb
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


def _cup_palm(
    hand: _GripHand,
    targets: _RightHandTargets,
    angles: Mapping[str, float],
    obstacle: ObstacleDistance,
    grip_radius_m: float,
) -> None:
    """Swing the thumb ray toward the grip, as a cupping palm does (#15181).

    A flat palm touches the grip along one line and leaves the thumb base
    far out, so the thumb can only reach the grip as a ramp. Here the thumb
    base bone, with the pad of flesh on it, turns about a line parallel to
    the club through a point between the thumb base and the middle of the
    palm. Of the angles up to ``palm_cup_max_deg`` that keep that pad out of
    ``obstacle``, it takes the one that brings the thumb's base joint
    nearest to where the thumb is to lie. Zero switches the cup off.
    """
    settings, asset = hand.settings, hand.asset
    hand.offsets.pop("thumb_meta", None)
    if settings.palm_cup_max_deg == 0.0:
        return
    i_thumb, i_palm = asset.bone_index("thumb_meta"), asset.bone_index("meta_middle")
    base = asset.bone_heads[i_thumb]
    palm = 0.5 * (asset.bone_heads[i_palm] + asset.bone_tails[i_palm])
    pivot = base + settings.palm_cup_pivot_fraction * (palm - base)
    axis = hand.placement[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    angles = {k: v for k, v in angles.items() if not k.startswith("thumb")}
    ids = hand.segment_ids("thumb_meta")
    allowance = settings.curl.penetration_allowance_m
    rho = grip_radius_m + _bone_skin_radius(asset, "thumb_meta")
    goal = rho * np.array(
        [math.cos(targets.thumb_azimuth_rad), math.sin(targets.thumb_azimuth_rad)]
    )

    def cupped(angle: float) -> Array:
        # The joint is carried along the arc; the thumb is aimed afterwards.
        arm = rotation_about_axis(axis, angle) @ (base - pivot)
        return rigid(np.eye(3), pivot + arm - base)

    def miss(angle: float) -> float:
        hand.offsets["thumb_meta"] = cupped(angle)
        if float(obstacle(hand.points(angles, ids)).min()) < -allowance:
            return math.inf
        joint = hand.joint_position("thumb_meta", angles)
        return float(np.linalg.norm(joint[:2] - goal))

    step = math.radians(settings.palm_cup_step_deg)
    count = int(settings.palm_cup_max_deg / settings.palm_cup_step_deg)
    best = min((k * step for k in range(-count, count + 1)), key=miss)
    hand.offsets["thumb_meta"] = cupped(best)


def _finger_ids(
    hand: _GripHand, finger: str, segments: Sequence[str] = _SEGMENTS
) -> NDArray[np.int64]:
    return np.concatenate([hand.segment_ids(f"{finger}_{s}") for s in segments])


def _side_gap(
    hand: _GripHand, angles: Mapping[str, float], finger: str, neighbour: str
) -> float:
    """Least skin distance from the middle and end segments of ``finger`` to
    ``neighbour``. The first segments are left out: the web between two
    fingers is always close, whatever the fingers do beyond it."""
    mine = _finger_ids(hand, finger, _SEGMENTS[1:])
    theirs = _finger_ids(hand, neighbour)
    pts = hand.points(angles, np.concatenate([mine, theirs]))
    return float(np.min(cKDTree(pts[len(mine) :]).query(pts[: len(mine)])[0]))


#: Joint-angle name of a finger's sideways swing at its knuckle: ``ring_side``.
_SIDE_SUFFIX = "_side"


def _close_fingers(hand: _GripHand, angles: Mapping[str, float]) -> None:
    """Swing each finger sideways at its knuckle until it meets its neighbour.

    The middle finger stays; the ring finger closes on it, the little finger
    on the ring finger and the index finger on the middle finger. A finger
    stops ``side_close.gap_m`` from its neighbour (the index finger
    ``index_gap_m``) or at ``max_deg``; zero ``max_deg`` switches this off.
    """
    settings = hand.settings
    limit = math.radians(settings.side_close_max_deg)
    hand.swing_sideways(None)
    for finger, neighbour in _SIDE_ORDER if limit else ():
        bone = f"{finger}_prox"
        index = finger == "index"
        want = settings.side_close_index_gap_m if index else settings.side_close_gap_m

        def gap(
            swing: float, bone: str = bone, pair: tuple = (finger, neighbour)
        ) -> float:
            hand.swing_sideways(bone, swing)
            return _side_gap(hand, angles, *pair)

        def apart(swing: float, bone: str = bone, other: str = neighbour) -> float:
            """Distance between the two fingers' end joints (mesh-independent)."""
            hand.swing_sideways(bone, swing)
            mine = hand.joint_position(bone.replace("_prox", "_dist"), angles)
            theirs = hand.joint_position(f"{other}_dist", angles)
            return float(np.linalg.norm(mine - theirs))

        probe = settings.curl.step_rad
        sign = 1.0 if apart(probe) < apart(-probe) else -1.0
        low, high = 0.0, sign * limit
        if gap(low) <= want or gap(high) > want:
            hand.swing_sideways(bone, low if gap(low) <= want else high)
            continue
        for _ in range(settings.curl.bisection_steps):
            mid = 0.5 * (low + high)
            low, high = (mid, high) if gap(mid) > want else (low, mid)
        hand.swing_sideways(bone, low)


def _overlap_little_finger(
    hand: _GripHand, angles: Mapping[str, float], groove: Callable[[Array], Array]
) -> None:
    """Swing the little finger over the groove between two fingers of the
    other hand, which it then curls into (overlap grip, #15183).

    ``groove`` is signed: zero where a point is as far from one of those
    fingers as from the other. The swing is searched from where the finger
    lies up to ``pair.overlap_max_swing_deg`` either way; zero switches it off.
    """
    settings = hand.settings
    limit = math.radians(settings.pair_overlap_max_swing_deg)
    if not limit:
        return
    bone = "little_prox"

    def offset(swing: float) -> float:
        hand.swing_sideways(bone, swing)
        ends = [
            hand.joint_position("little_dist", angles, tail=t) for t in (False, True)
        ]
        return float(groove(np.array(ends)).mean())

    low = hand.side.get(bone, 0.0)
    at_low = offset(low)
    high = min((-limit, limit), key=lambda swing: abs(offset(swing)))
    if at_low * offset(high) > 0.0:  # the groove is out of reach: go nearest
        hand.swing_sideways(bone, high if abs(offset(high)) < abs(at_low) else low)
        return
    for _ in range(settings.curl.bisection_steps):
        mid = 0.5 * (low + high)
        low, high = (mid, high) if offset(mid) * at_low > 0.0 else (low, mid)
    hand.swing_sideways(bone, 0.5 * (low + high))


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


def _lay_thumb(
    hand: _GripHand,
    targets: _RightHandTargets,
    angles: Mapping[str, float],
    obstacle: ObstacleDistance,
    grip_radius_m: float,
    max_turn_deg: float,
) -> dict[str, float]:
    """Aim the thumb and press it onto the grip, then turn it to its target.

    Pressing a thumb down carries it sideways. ``thumb.toward_index_target``
    names what the turn, of up to ``max_turn_deg`` (the hand's
    ``thumb.toward_index_max_deg``), brings it to: ``"azimuth"`` puts the
    middle of its end bone back on the spec azimuth; ``"knuckle"`` brings its
    end joint within ``thumb.knuckle_distance_m`` of the forefinger's knuckle,
    a tight V; ``"fingertip"`` brings its skin within ``thumb.index_gap_m`` of
    the fingertip. The last two take the least such turn toward the hand's own
    forefinger and never leave the grip to ride on the finger or run into the
    other hand. The angle is found in ``thumb.toward_index_steps`` steps; a
    turned thumb may bend as far as ``thumb.toward_index_straightness`` allows
    and the forefinger's skin is in its way. The turn is returned as the
    ``thumb_side`` angle; a maximum of zero leaves the thumb as first aimed.
    """
    settings, key = hand.settings, f"thumb{_SIDE_SUFFIX}"
    pinch = settings.thumb_toward_index_target == "knuckle"
    skin = hand.points(angles)
    tip = skin[hand.segment_ids("index_dist")]
    knuckle = hand.joint_position("index_prox", angles)
    goal = knuckle if pinch else tip.mean(axis=0)
    toward = math.remainder(
        math.atan2(goal[1], goal[0]) - targets.thumb_azimuth_rad, math.tau
    )
    reach = math.copysign(min(abs(toward), math.radians(max_turn_deg)), toward)
    ids = _finger_ids(hand, "index", _SEGMENTS[1:])
    normals = vertex_normals(skin, hand.asset.faces)[ids]
    finger = skin_distance(skin[ids], normals, settings.pair_skin_reach_m)
    tree, thumb = cKDTree(tip), hand.segment_ids(THUMB_PRESS)
    bend = min(targets.thumb_straightness, settings.thumb_toward_index_straightness)
    gap_m = settings.thumb_index_gap_m
    lifted = math.radians(settings.thumb_press_range_deg[0]) + 1e-9

    def around(points: Array) -> Array:
        return np.minimum(obstacle(points), finger(points))

    def there(pose: Mapping[str, float], skin: Array) -> bool:
        if pinch:
            joint = hand.joint_position("thumb_dist", pose)
            apart = float(np.linalg.norm(joint - knuckle))
            return apart <= settings.thumb_knuckle_distance_m
        return float(np.min(np.asarray(tree.query(skin)[0]))) <= gap_m

    def laid(turn: float, seat: bool = False) -> tuple[bool, bool, dict[str, float]]:
        """The thumb turned by ``turn``: is it at its target, has it left
        the grip (to ride on the finger, or stuck in the other hand), and
        its pose. ``seat`` lays it with the wrap search, as an unturned
        thumb is laid, so its first segment sits as low."""
        share = min(abs(turn / reach), 1.0) if reach else 0.0
        straight = (1.0 - share) * targets.thumb_straightness + share * bend
        aimed = _turned(targets, turn, straight)
        _aim_thumb(hand, aimed, angles, grip_radius_m)
        chain = (_thumb_chain(settings, aimed),)
        if turn and not seat:  # a plain curl: the gap closes steadily
            pose = curl_to_contact(hand, chain, around, settings.curl, angles).angles
        else:
            pose = _wrap(hand, chain, around if turn else obstacle, angles)
        skin = hand.points(pose, thumb)
        # Off the grip it rides on the finger; deep in, or never pressed
        # down from where it starts, something is in its way.
        depth = float(np.min(obstacle(skin)))
        off = abs(depth) > gap_m or pose[THUMB_PRESS] <= lifted
        return there(pose, skin), off, {**pose, key: turn}

    if settings.thumb_toward_index_target == "azimuth":

        def missed(turn: float) -> tuple[float, dict[str, float]]:
            pose = laid(turn, True)[2]  # pressing it down carries it sideways
            ends = [
                hand.joint_position("thumb_dist", pose, tail=t) for t in (False, True)
            ]
            pad = 0.5 * (ends[0] + ends[1])
            miss = targets.thumb_azimuth_rad - math.atan2(pad[1], pad[0])
            return math.remainder(miss, math.tau), pose

        most, close = math.radians(max_turn_deg), settings.thumb_azimuth_close_rad
        steps = settings.thumb_toward_index_steps
        chosen = _scanned_turn(missed, most, steps, close)
        return laid(chosen[key], True)[2]  # leaves the hand aimed as chosen
    near, _, pose = laid(0.0)
    if not reach or near:
        return pose
    low, high = 0.0, reach
    for _ in range(settings.thumb_toward_index_steps):
        mid = 0.5 * (low + high)
        near, riding, _ = laid(mid)
        low, high = (low, mid) if near or riding else (mid, high)
    turn = low if laid(high)[1] else high
    _, riding, pose = laid(turn, pinch)
    return laid(turn)[2] if riding else pose  # leaves the hand aimed as chosen


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
    samples: int | None = None,
    target: ObstacleDistance | None = None,
) -> dict[str, float]:
    """Curl ``chains`` to contact, with the wrap search if it is switched on
    (``samples`` overrides ``curl.wrap_scan_samples``; ``target`` as in
    :func:`posing.wrap_to_contact`)."""
    if not chains:
        return dict(start)
    settings = hand.settings
    samples = settings.wrap_scan_samples if samples is None else samples
    if samples >= 2:
        return wrap_to_contact(
            hand,
            chains,
            obstacle,
            settings.curl,
            start,
            samples=samples,
            target=target,
        ).angles
    return curl_to_contact(hand, chains, obstacle, settings.curl, start).angles


@dataclass(frozen=True)
class _Scene:
    """What a hand of a pair is posed around, in that hand's build frame.

    ``distance(depth, palm=False)`` is the signed distance to the grip, shrunk
    by ``depth``, and the other hand; ``palm`` asks for the shape the palm
    must clear. ``groove``, if any, is where the little finger overlaps (see
    :func:`_overlap_little_finger`).
    """

    distance: Callable[..., ObstacleDistance]
    groove: Callable[[Array], Array] | None = None
    #: Skin vertices of the other hand, which this hand's skin gives way to.
    skin_points: Array | None = None


@dataclass(frozen=True)
class _SolvedHand:
    hand: _GripHand
    angles: dict[str, float]
    targets: _RightHandTargets
    obstacle: ObstacleDistance
    skin_asset: SkinnedHandAsset
    #: What the palm pad settles onto, if not ``obstacle`` (a pair: the other
    #: hand at ``pair.palm_capsule_radius_scale``).
    palm_obstacle: ObstacleDistance | None = None
    #: Skin vertices of the other hand of a pair (see :class:`_Scene`).
    other_skin: Array | None = None


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
    pair: _Scene | None = None,
) -> _SolvedHand:
    """Pose one hand as a right hand, around ``pair`` if it is one of a pair
    (default: the bare grip)."""
    paired = pair is not None
    size = spec.hand_size
    solve, skin = _sized_assets(size.length_m, size.breadth_m, settings)
    hand = _GripHand(solve, settings.beside_other_hand() if paired else settings)
    targets = _targets(spec, side, settings)
    deepest = max(settings.squeeze_finger_depth_m, settings.squeeze_palm_depth_m)
    if deepest >= grip_radius_m:
        raise ValueError("squeeze depth must be less than the grip radius")

    def bare(depth: float, **_: object) -> ObstacleDistance:
        return cylinder_distance(grip_radius_m - depth, settings.grip_taper)

    scene = pair.distance if pair is not None else bare
    obstacle = scene(0.0)
    pad = scene(0.0, palm=True) if paired else None
    points = pair.skin_points if pair is not None else None
    # Soft pads give: the palm seats, and the digits close, a little way in.
    grip = bare(settings.squeeze_palm_depth_m)
    other = scene(settings.squeeze_palm_depth_m, palm=True) if paired else None
    soft = scene(settings.squeeze_finger_depth_m)
    thumb_radius_m = grip_radius_m - settings.squeeze_finger_depth_m
    arch = {f"meta_{f}": 0.0 for f in FINGERS}
    # The hand is seated with the forefinger knuckle already flexed, so the
    # club lies in the crook of that finger and the finger can wrap (#15182).
    knuckle = math.radians(settings.index_knuckle_flexion_deg)
    crook = {} if spec.trigger_finger else {"index_prox": knuckle}
    if given is not None:
        arch = {k: float(given.get(k, 0.0)) for k in arch}
        _seat_flat_hand(hand, targets, grip, {**arch, **crook}, other)
        posed = {**_straight_angles(), **{k: float(v) for k, v in given.items()}}
        hand.swing_sideways(None)
        for finger in FINGERS:  # sideways swings are applied as given, too
            if f"{finger}{_SIDE_SUFFIX}" in posed:
                hand.swing_sideways(f"{finger}_prox", posed[f"{finger}{_SIDE_SUFFIX}"])
        _cup_palm(hand, targets, posed, soft, thumb_radius_m)
        turn = posed.get(f"thumb{_SIDE_SUFFIX}", 0.0)
        _aim_thumb(hand, _turned(targets, turn), posed, thumb_radius_m)
        _bend_wrist(hand, side)
        return _SolvedHand(hand, posed, targets, obstacle, skin, pad, points)
    chains = _finger_chains(settings, crook.get("index_prox", 0.0))
    # Of a pair, the fingers that hold the grip aim for the grip itself; the
    # other hand is only in their way. The rest may come to rest on it.
    held = settings.pair_grip_fingers if paired else ()
    mine = bare(settings.squeeze_finger_depth_m)

    def wrap(start: Mapping[str, float], samples: int | None = None) -> dict:
        """Wrap the fingers; with ``samples`` given, the first pass, on which
        the sideways closing is fitted: there the grip fingers see only the
        grip, as on a single hand, so the other hand cannot skew the fit."""
        on_grip = [chain for chain in chains if chain.name in held]
        others = [chain for chain in chains if chain.name not in held]
        around = mine if samples is not None else soft
        angles = _wrap(hand, on_grip, around, start, samples, mine)
        return _wrap(hand, others, soft, angles, samples)

    angles: dict[str, float] = {}
    for _ in range(settings.arch_passes):
        _seat_flat_hand(hand, targets, grip, {**arch, **crook}, other)
        start = {**_straight_angles(), **arch, **crook}
        angles = curl_to_contact(hand, chains, soft, settings.curl, start).angles
        arch = _arch_from_knuckles(angles, settings)
    _seat_flat_hand(hand, targets, grip, {**arch, **crook}, other)
    start = {**_straight_angles(), **arch, **crook}
    # Close the fingers sideways on the wrapped pose, then wrap again: the
    # greedy curl hooks the forefinger, and a swing fitted to it is wrong.
    hand.swing_sideways(None)
    if settings.side_close_max_deg:
        coarse = min(settings.wrap_scan_samples, settings.side_close_scan_samples)
        angles = wrap(start, coarse)
        _close_fingers(hand, angles)
    if pair is not None and pair.groove is not None:
        _overlap_little_finger(hand, angles, pair.groove)
    angles = wrap(start)
    _cup_palm(hand, targets, angles, soft, thumb_radius_m)
    most = dict(settings.thumb_toward_index_max_deg)[side]
    angles = _lay_thumb(hand, targets, angles, soft, thumb_radius_m, most)
    _bend_wrist(hand, side)
    for bone, swing in hand.side.items():  # so given angles rebuild this pose
        angles[f"{bone.removesuffix('_prox')}{_SIDE_SUFFIX}"] = swing
    return _SolvedHand(hand, angles, targets, obstacle, skin, pad, points)


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
    solved: _SolvedHand,
    transforms: Array,
    settings: SkinnedHumanSettings,
) -> Array:
    """Skinned vertices in the grip frame, with the corrective shapes applied.

    Linear blend skinning first; then the palm pad settles onto the obstacle,
    creases that skinning pinched between the knuckles relax, skin pressed
    into what the hand holds is laid on its surface and bulges beside it, and
    the skin around any folded triangle relaxes (see ``correctives``). With
    all four switched off this is exactly ``pose_vertices``.
    """
    asset, placement = solved.skin_asset, solved.hand.placement
    rotation, offset = placement[:3, :3], placement[:3, 3]
    skin = pose_vertices(asset, transforms) @ rotation.T + offset
    unknown = set(settings.palm_pad.bones) - set(asset.bone_names)
    if unknown:
        raise ValueError(f"correctives.palm_pad.bones names no bone: {sorted(unknown)}")
    skin = conform_pads(
        skin,
        asset.faces,
        asset.dominant_vertex_ids(settings.palm_pad.bones),
        solved.palm_obstacle or solved.obstacle,
        settings.palm_pad,
        lambda v: vertex_normals(v, asset.faces),
    )
    relaxable_mask = _bone_mask(
        asset, settings.crease_relax_bones, "correctives.crease_relax.bones"
    )
    relaxable = np.asarray(relaxable_mask[asset.faces].all(axis=1))

    def creased(v: Array) -> NDArray[np.int64]:
        limit = settings.crease_relax_max_added_fold_deg
        found = sharpened_faces(v, asset.rest_vertices, asset.faces, limit)
        return found[relaxable[found]]

    skin = relax_folds(skin, asset.faces, creased, settings.crease_relax)
    skin = press_flat(
        skin,
        asset.faces,
        solved.obstacle,
        settings.press,
        lambda v: vertex_normals(v, asset.faces),
    )
    skin = relax_folds(
        skin,
        asset.faces,
        lambda v: flipped_face_ids(asset, transforms, (v - offset) @ rotation),
        settings.fold_relax,
    )
    # Relaxing may have drawn skin back in: leave none inside what is held.
    skin = lift_out(skin, solved.obstacle, max_depth_m=settings.press.max_depth_m)
    if solved.other_skin is None:
        return skin
    return yield_to_points(
        skin,
        lambda v: vertex_normals(v, asset.faces),
        solved.other_skin,
        max_depth_m=settings.pair_yield_max_depth_m,
        passes=settings.pair_yield_passes,
        clearance_m=settings.pair_yield_clearance_m,
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
    asset = solved.skin_asset
    flip = -1 if solved.targets.mirrored else 1
    faces = asset.faces[:, ::flip].copy()
    skin = _corrected_skin(solved, transforms, settings)
    vertices = skin @ club_from_grip[:3, :3].T + club_from_grip[:3, 3]
    require(bool(np.isfinite(vertices).all()), "skinned vertices must be finite")
    mesh = Mesh(vertices, faces)
    ensure(mesh.volume() > 0.0, "skinned hand must stay outward wound")
    unknown = set(settings.frame_excluded_bones) - set(asset.bone_names)
    if unknown:
        raise ValueError(f"framing.excluded_bones names no bone: {sorted(unknown)}")
    focus = np.ones(len(vertices), dtype=bool)
    focus[asset.dominant_vertex_ids(settings.frame_excluded_bones)] = False
    return HandVisualResult(
        backend=BACKEND_NAME,
        side=side,
        mesh=mesh,
        skin_rgb=settings.skin_rgb,
        vertex_normals=vertex_normals(vertices, faces),
        joint_angles=HandJointAngles.from_mapping(solved.angles),
        contact_gaps_m=_segment_gaps(hand, skin, solved.obstacle),
        grip_radius_m=float(grip_radius_m - settings.grip_taper * z_m),
        grip_taper=settings.grip_taper,
        bone_transforms=transforms,
        frame_from_rest=club_from_grip @ hand.placement,
        rest_asset=asset,
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
