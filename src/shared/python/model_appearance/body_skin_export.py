"""Bones of the fitted CC0 body skin for an engine's native skinning (#15299).

An engine skin (MuJoCo ``<skin>``) blends rigid bodies, not the pose module's
group transforms. This module turns the stored skin groups into shares of the
model's bodies and gives each body its bind pose in the fitted rest space:

* ``group_shares``: the bodies a group follows (spine groups share pelvis and
  thorax, neck groups thorax and head, the deltoid girdle and upper arm);
* ``subject_lengths``: the subject's geometry from the body frames at the bind pose;
* ``bind_poses``: each body's pose in the fitted rest space, its proximal joint
  on the fitted joint and its bone along the fitted bone;
* ``skin_bones`` and ``pose_bones``: one layer's per-body vertex lists and
  weights, and linear blend skinning with them (the engine's rule);
* ``blend_wrists``: the forearm's end shared with the body that carries the
  hand, so the skin's end ring stays under a rigid hand's cuff at any wrist bend;
* ``narrow_head_base``: the rigid head model's base held round the neck axis and
  drawn down the neck, so a turned head does not push it out of the skin's neck
  and the collar and no view passes under it into the open neck; its neck stub
  is held the same way up to the jaw line, so it does not lie in the skin's wall.

Body frames are ``Rigid`` poses (body frame to world) keyed by role:
``pelvis``, ``thorax``, ``head`` and, per side, ``girdle``, ``upper_arm``,
``elbow``, ``forearm``, ``hand``, ``thigh``, ``shank`` and ``foot`` (hand and
foot only mark the wrist and the ankle). Settings and sources are in
``assets/body_skin_cc0/export_settings.json``. Appearance only; opt-in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.body_skin_fit import (
    ASSET_DIR,
    SIDES,
    FittedSkin,
    SegmentLengths,
)
from src.shared.python.model_appearance.body_skin_pose import (
    NECK_GROUPS,
    PoseSettings,
    Rigid,
)

Array = NDArray[np.float64]
#: Values of ``subject.body_visual_backend``; the loft body is the default.
LOFT_BACKEND, SKINNED_BACKEND = "loft", "skinned_cc0"
_ARM: tuple[str, ...] = ("girdle", "upper_arm", "elbow", "forearm", "hand")
_LEG: tuple[str, ...] = ("thigh", "shank", "foot")
SPINE_GROUPS = ("spine05", "spine04", "spine03")
#: Side group -> the body role it rides (the deltoid group is shared instead).
_SIDE_BONES = {
    "clavicle": "girdle",
    "upperarm_proximal": "upper_arm",
    "upperarm_distal": "upper_arm",
    "forearm_proximal": "elbow",
    "forearm_distal": "forearm",
    "thigh_proximal": "thigh",
    "thigh_distal": "thigh",
    "shank": "shank",
}
#: Joint centre -> the body role whose origin it is.
_JOINT_BODY = {
    "girdle": "girdle",
    "shoulder": "upper_arm",
    "elbow": "elbow",
    "wrist": "hand",
    "hip": "thigh",
    "knee": "shank",
    "ankle": "foot",
}
#: Bone role -> its proximal and distal joint (the bone runs between them).
_BONES = {
    "thorax": ("lumbar", "neck"),
    "upper_arm": ("shoulder", "elbow"),
    "elbow": ("elbow", "wrist"),
    "forearm": ("elbow", "wrist"),
    "thigh": ("hip", "knee"),
    "shank": ("knee", "ankle"),
}
_ANCHORS = {"pelvis": "hip_mid", "head": "neck", "girdle": "girdle"}
_HEAD_BASE = ("height_m", "front_m", "side_m", "back_m", "drop_m")
_HEAD_BASE_JAW = ("jaw_height_m", "jaw_slope", "flare_m")


@dataclass(frozen=True)
class ExportSettings:
    """Named export settings; sources live in ``export_settings.json``."""

    spine_follow: tuple[float, ...]
    weight_floor: float
    bind_pose: Mapping[str, float]
    bind_stretch_limit_m: float
    covered_parts: tuple[str, ...]
    wrist_blend_length_m: float = 0.0
    wrist_blend_full_m: float = 0.0
    head_base_height_m: float = 0.0
    head_base_front_m: float = 0.0
    head_base_side_m: float = 0.0
    head_base_back_m: float = 0.0
    head_base_drop_m: float = 0.0
    head_base_jaw_height_m: float = 0.0
    head_base_jaw_slope: float = 0.0
    head_base_flare_m: float = 0.0

    def __post_init__(self) -> None:
        self._check_head_base()
        follow = list(self.spine_follow)
        if len(follow) != len(SPINE_GROUPS) or follow != sorted(follow):
            raise ValueError("spine_follow must hold three rising shares")
        if not 0.0 <= follow[0] <= follow[-1] <= 1.0:
            raise ValueError("spine_follow must lie in [0, 1]")
        if not 0.0 <= self.weight_floor < 0.5:
            raise ValueError("weight_floor outside [0, 0.5)")
        if not self.bind_stretch_limit_m > 0.0:
            raise ValueError("bind_stretch_limit_m must be positive")
        length, full = self.wrist_blend_length_m, self.wrist_blend_full_m
        if length < 0.0 or (length > 0.0 and not 0.0 <= full < length):
            raise ValueError("wrist_blend needs 0 <= full_m < length_m, or length_m 0")

    def _check_head_base(self) -> None:
        """``TypeError`` for a head_base value that is no number, ``ValueError``
        unless height and drop are >= 0 and, with a height, each limit > 0."""
        limits = (self.head_base_front_m, self.head_base_side_m, self.head_base_back_m)
        height, drop = self.head_base_height_m, self.head_base_drop_m
        jaw = (self.head_base_jaw_height_m, self.head_base_flare_m)
        for value in (height, drop, *limits, *jaw, self.head_base_jaw_slope):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("head_base values must be numbers (metres)")
        if height < 0.0 or drop < 0.0 or (height > 0.0 and not min(limits) > 0.0):
            raise ValueError(
                "head_base needs height_m >= 0, drop_m >= 0 and, with a height, "
                "front_m, side_m and back_m > 0"
            )
        if min(*jaw, self.head_base_jaw_slope) < 0.0 or (jaw[0] > 0.0) != (
            jaw[1] > 0.0
        ):
            raise ValueError(
                "head_base needs jaw_height_m, jaw_slope and flare_m >= 0, and "
                "jaw_height_m and flare_m together or neither"
            )


@dataclass(frozen=True)
class SkinBone:
    """One body of a skin layer: its bind pose and the vertices it moves."""

    role: str
    bind: Rigid
    vertex_ids: NDArray[np.int64]
    weights: Array


def load_export_settings(path: Path | None = None) -> ExportSettings:
    """Read ``export_settings.json``; raises ``ValueError`` for invalid values."""
    raw = json.loads((path or ASSET_DIR / "export_settings.json").read_text())
    return ExportSettings(
        spine_follow=tuple(raw["spine_follow"]),
        weight_floor=float(raw["weight_floor"]),
        bind_pose={str(k): float(v) for k, v in raw["bind_pose"].items()},
        bind_stretch_limit_m=float(raw["bind_stretch_limit_m"]),
        covered_parts=tuple(raw["covered_parts"]),
        wrist_blend_length_m=float(raw["wrist_blend_length_m"]),
        wrist_blend_full_m=float(raw["wrist_blend_full_m"]),
        **{
            f"head_base_{key}": float(raw["head_base"][key])
            for key in _HEAD_BASE + _HEAD_BASE_JAW
        },
    )


def body_skin_enabled(spec: Mapping[str, Any]) -> bool:
    """Whether ``subject.body_visual_backend`` asks for the skinned body."""
    backend = spec.get("subject", {}).get("body_visual_backend", LOFT_BACKEND)
    if backend not in (LOFT_BACKEND, SKINNED_BACKEND):
        raise ValueError(
            f"body_visual_backend must be {LOFT_BACKEND!r} or {SKINNED_BACKEND!r}"
        )
    return bool(backend == SKINNED_BACKEND)


def body_roles(spec: Mapping[str, Any], origins: Mapping[str, Array]) -> dict[str, str]:
    """Body name per role, from the joint tree and the body classes.

    ``origins`` are the body origins at the bind pose (y to the subject's
    left). The thorax is the body that carries the head; an arm hangs from
    each shoulder link on it and a leg from each thigh on the pelvis. Raises
    ``ValueError`` when the tree lacks one of them.
    """
    children: dict[str, list[str]] = {}
    for joint in spec.get("joints", ()):
        children.setdefault(str(joint["parent"]), []).append(str(joint["child"]))

    def kids(body: str, part: str) -> list[str]:
        return [c for c in children.get(body, ()) if library.classify_body(c) == part]

    try:
        thorax, head = next(
            (b, kids(b, "head")[0]) for b in children if kids(b, "head")
        )
        pelvis = next(b for b in children if library.classify_body(b) == "pelvis")
        roles = {"pelvis": pelvis, "thorax": thorax, "head": head}
        limbs = [(_ARM, b) for b in kids(thorax, "shoulder")]
        limbs += [(_LEG, b) for b in kids(pelvis, "thigh")]
        for names, body in limbs:
            chain = [body]
            while len(chain) < len(names):
                chain.append(children[chain[-1]][0])
            left = origins[chain[-1]][1] > origins[pelvis][1]
            roles |= {
                f"{n}_{'l' if left else 'r'}": b
                for n, b in zip(names, chain, strict=True)
            }
    except (StopIteration, KeyError, IndexError) as missing:
        raise ValueError("the spec lacks a body the skin follows") from missing
    return roles


def group_shares(
    groups: Sequence[str], pose: PoseSettings, settings: ExportSettings
) -> dict[str, dict[str, float]]:
    """Per skin group, the share of each body role it follows (sums to one).

    Raises ``ValueError`` for a group the skin rig does not know.
    """
    central = {name: {name: 1.0} for name in ("pelvis", "thorax", "head")}
    for group, share in zip(SPINE_GROUPS, settings.spine_follow, strict=True):
        central[group] = {"pelvis": 1.0 - share, "thorax": share}
    for group, share in zip(NECK_GROUPS, pose.neck_follow, strict=True):
        central[group] = {"thorax": 1.0 - share, "head": share}
    out: dict[str, dict[str, float]] = {}
    for group in groups:
        name, _, side = group.rpartition("_")
        if group in central:
            out[group] = central[group]
        elif name == "shoulder":
            follow = pose.shoulder_follow
            out[group] = {f"girdle_{side}": 1.0 - follow, f"upper_arm_{side}": follow}
        elif name in _SIDE_BONES:
            out[group] = {f"{_SIDE_BONES[name]}_{side}": 1.0}
        else:
            raise ValueError(f"skin group {group!r} has no body to follow")
    return out


def model_joints(frames: Mapping[str, Rigid]) -> dict[str, Array]:
    """Joint centres of the model from its body frames (body origins).

    Raises ``ValueError`` naming the first body role that is missing.
    """
    try:
        joints = {"lumbar": frames["thorax"].translation}
        joints["neck"] = frames["head"].translation
        for side, _ in SIDES:
            for joint, role in _JOINT_BODY.items():
                joints[f"{joint}_{side}"] = frames[f"{role}_{side}"].translation
    except KeyError as missing:
        raise ValueError(f"missing body frame {missing}") from missing
    joints["hip_mid"] = 0.5 * (joints["hip_l"] + joints["hip_r"])
    return joints


def subject_lengths(frames: Mapping[str, Rigid]) -> SegmentLengths:
    """The subject's segment lengths; each limb also with its two side lengths.

    A limb's one length is the mean of both sides; ``sides`` holds the two, so
    the fit puts every knee, ankle, elbow and wrist on its own model joint.
    """
    joints = model_joints(frames)

    def span(first: str, second: str) -> float:
        return float(np.linalg.norm(joints[first] - joints[second]))

    def limb(first: str, second: str) -> tuple[float, float]:
        left, right = (span(f"{first}_{s}", f"{second}_{s}") for s, _ in SIDES)
        return left, right

    sides = {
        "upper_arm": limb("shoulder", "elbow"),
        "forearm": limb("elbow", "wrist"),
        "thigh": limb("hip", "knee"),
        "shank": limb("knee", "ankle"),
    }
    return SegmentLengths(
        hip_half_spacing=0.5 * span("hip_l", "hip_r"),
        shoulder_half_spacing=0.5 * span("shoulder_l", "shoulder_r"),
        shoulder_height=0.5 * sum(span(f"girdle_{s}", "hip_mid") for s, _ in SIDES),
        neck_height=span("neck", "hip_mid"),
        **{name: 0.5 * sum(pair) for name, pair in sides.items()},
        sides=sides,
    )


def rest_joints(fitted: FittedSkin, model: Mapping[str, Array]) -> dict[str, Array]:
    """The fitted joints plus hip midpoint, lumbar joint and girdle origins.

    The three extra points lie on the fitted trunk axis (hip midpoint to neck
    joint) at the model's distances from the hip midpoint.
    """
    rest = dict(fitted.joints)
    rest["hip_mid"] = middle = 0.5 * (rest["hip_l"] + rest["hip_r"])
    axis = rest["neck"] - middle
    axis = axis / float(np.linalg.norm(axis))
    for name in ("lumbar", *(f"girdle_{side}" for side, _ in SIDES)):
        height = float(np.linalg.norm(model[name] - model["hip_mid"]))
        rest[name] = middle + height * axis
    return rest


def bone_joints(role: str) -> tuple[str, ...]:
    """Joints on the bone of ``role``, proximal first (one for an anchored body)."""
    name, _, side = role.rpartition("_")
    if role in _BONES:
        return _BONES[role]
    if role in _ANCHORS:
        return (_ANCHORS[role],)
    if name in _BONES:
        return tuple(f"{joint}_{side}" for joint in _BONES[name])
    if name in _ANCHORS:
        return (f"{_ANCHORS[name]}_{side}",)
    raise ValueError(f"{role!r} is not a bone of the body skin")


def _turn(start: Array, end: Array) -> Array:
    """The smallest rotation that carries the direction ``start`` onto ``end``."""
    first, second = (v / float(np.linalg.norm(v)) for v in (start, end))
    axis = np.cross(first, second)
    sine = float(np.linalg.norm(axis))
    if sine < 1.0e-12:
        return np.eye(3)
    angle = np.arctan2(sine, float(first @ second))
    return np.asarray(Rotation.from_rotvec(axis / sine * angle).as_matrix())


def bind_poses(frames: Mapping[str, Rigid], fitted: FittedSkin) -> dict[str, Rigid]:
    """Pose of every bone body in the fitted rest space, keyed by role.

    ``frames`` are the model's body frames at the bind pose. Each body is
    turned by the smallest rotation that lays its bone along the fitted bone
    (pelvis: none; head and girdle: the thorax's) and placed so its proximal
    joint is the fitted one. Postcondition: skinning a fitted joint with its
    bone at the bind pose gives the model joint.
    """
    model = model_joints(frames)
    rest = rest_joints(fitted, model)
    roles = ["pelvis", "thorax", "head"]
    roles += [
        f"{name}_{s}" for s, _ in SIDES for name in dict.fromkeys(_SIDE_BONES.values())
    ]
    binds: dict[str, Rigid] = {}
    for role in roles:
        joints = bone_joints(role)
        if len(joints) == 2:
            near, far = joints
            turn = _turn(model[far] - model[near], rest[far] - rest[near])
        else:
            turn = (
                np.eye(3)
                if role == "pelvis"
                else binds["thorax"].rotation @ (frames["thorax"].rotation.T)
            )
        frame, near = frames[role], joints[0]
        origin = rest[near] + turn @ (frame.translation - model[near])
        binds[role] = Rigid(turn @ frame.rotation, origin)
    return binds


def skin_bones(
    weights: Array,
    groups: Sequence[str],
    shares: Mapping[str, Mapping[str, float]],
    binds: Mapping[str, Rigid],
    settings: ExportSettings,
) -> tuple[SkinBone, ...]:
    """The bones of one layer from its ``(vertices, groups)`` skin weights.

    Shares under ``weight_floor`` are dropped and each vertex renormalised;
    a body that moves no vertex is left out. Bones come in ``binds`` order.
    """
    if weights.ndim != 2 or weights.shape[1] != len(groups):
        raise ValueError("weights must be (vertices, groups)")
    column = {role: index for index, role in enumerate(binds)}
    spread = np.zeros((len(groups), len(column)))
    for row, group in enumerate(groups):
        for role, share in shares[group].items():
            spread[row, column[role]] = share
    body = weights @ spread
    body[body < settings.weight_floor] = 0.0
    body /= body.sum(axis=1, keepdims=True)
    bones = []
    for role, index in column.items():
        ids = np.flatnonzero(body[:, index])
        if len(ids):
            bones.append(SkinBone(role, binds[role], ids, body[ids, index]))
    return tuple(bones)


def blend_wrists(
    bones: Sequence[SkinBone],
    vertices: Array,
    binds: Mapping[str, Rigid],
    hands: Mapping[str, Rigid],
    settings: ExportSettings,
) -> tuple[SkinBone, ...]:
    """``bones`` with each forearm's end shared with the body of its hand.

    ``hands`` holds, per side (``l``, ``r``), the bind pose of the body that
    carries the hand, its origin on the wrist joint; ``vertices`` are the
    layer's, in the fitted rest space. Over the last ``wrist_blend_length_m``
    before the wrist joint a vertex of the forearm's two bodies passes
    smoothly to a ``hand_<side>`` bone, which holds it whole from
    ``wrist_blend_full_m`` before the joint. Postcondition: every vertex keeps
    its total weight, and with a zero length (or no hand) the bones are
    returned unchanged. Raises ``ValueError`` for an unknown side.
    """
    length = settings.wrist_blend_length_m
    span = length - settings.wrist_blend_full_m
    out = list(bones)
    for side, hand in hands.items():
        if f"elbow_{side}" not in binds:
            raise ValueError(f"the skin has no forearm on side {side!r}")
        if length == 0.0:
            continue
        axis = hand.translation - binds[f"elbow_{side}"].translation
        before = (hand.translation - vertices) @ axis / np.linalg.norm(axis)
        t = np.clip((length - before) / span, 0.0, 1.0)
        share, taken = t * t * (3.0 - 2.0 * t), np.zeros(len(vertices))
        for index, bone in enumerate(out):
            if bone.role in (f"forearm_{side}", f"elbow_{side}"):
                ids, part = bone.vertex_ids, share[bone.vertex_ids]
                taken[ids] += bone.weights * part
                stays = part < 1.0
                weights = (bone.weights * (1.0 - part))[stays]
                out[index] = SkinBone(bone.role, bone.bind, ids[stays], weights)
        moved = np.flatnonzero(taken)
        out.append(SkinBone(f"hand_{side}", hand, moved, taken[moved]))
    return tuple(bone for bone in out if len(bone.vertex_ids))


def narrow_head_base(vertices: Array, settings: ExportSettings) -> Array:
    """Head-model ``vertices`` with the base held inside the skin's neck.

    ``vertices`` are ``(N, 3)`` in the head frame: origin on the neck joint, x
    forward, z up the neck. The head model is rigid while the skin's neck only
    shares the head's turn, so a wide base comes out of the neck and the collar
    when the head turns; a base held in a small circle leaves an open slot
    behind the skin's rim under the jaw. A vertex below ``head_base_height_m``
    is moved straight towards the neck axis onto an ellipse: half width
    ``head_base_side_m``, depth ``head_base_front_m`` forward and
    ``head_base_back_m`` behind. It is then drawn down the neck by
    ``head_base_drop_m`` on and under the joint plane, falling linearly to
    nothing at that height, so a view over the rim meets the base. The model's
    neck stub above that height is the skin's neck drawn a second time, a few
    millimetres from it, and the two cut each other in a ragged line; so the
    hold onto the ellipse goes on up to the jaw line, the plane
    ``head_base_jaw_height_m`` above the joint on the axis that drops
    ``head_base_jaw_slope`` per metre forward, and lets go smoothly over the
    ``head_base_flare_m`` below it (``head_base_flare_m`` 0: no hold above the
    base). Postcondition: a vertex at or above both that height and the jaw
    line is unchanged, no vertex moves away from the axis, and none leaves its
    vertical plane through the axis. Raises ``ValueError`` for another shape.
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must be (N, 3)")
    out = points.copy()
    height = settings.head_base_height_m
    if not height > 0.0:
        return out
    low = points[:, 2] < height
    hold = low.astype(float)
    if settings.head_base_flare_m > 0.0:
        line = settings.head_base_jaw_height_m
        under = line - settings.head_base_jaw_slope * points[:, 0] - points[:, 2]
        part = np.clip(under / settings.head_base_flare_m, 0.0, 1.0)
        hold = np.maximum(hold, part * part * (3.0 - 2.0 * part))
    forward = points[:, 0] > 0.0
    depth = np.where(forward, settings.head_base_front_m, settings.head_base_back_m)
    reach = np.hypot(points[:, 0] / depth, points[:, 1] / settings.head_base_side_m)
    held = (hold > 0.0) & (reach > 1.0)
    out[held, :2] *= (1.0 + hold[held] * (1.0 / reach[held] - 1.0))[:, None]
    share = 1.0 - np.clip(points[low, 2] / height, 0.0, 1.0)
    out[low, 2] -= settings.head_base_drop_m * share
    return out


def pose_bones(
    vertices: Array, bones: Sequence[SkinBone], frames: Mapping[str, Rigid]
) -> Array:
    """Linear blend skinning of ``vertices`` for the body ``frames``."""
    posed = np.zeros_like(vertices)
    for bone in bones:
        local = bone.bind.inverse().apply(vertices[bone.vertex_ids])
        posed[bone.vertex_ids] += bone.weights[:, None] * frames[bone.role].apply(local)
    return posed
