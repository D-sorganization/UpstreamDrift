"""Fit the stored CC0 body skin to a subject's joint centres (#15299).

Reshapes the stored skin (``assets/body_skin_cc0``: trunk, arms to the wrists,
legs to the ankles) so its landmarks land on a subject's joint centres:

* the trunk is warped along its own axis (hip midpoint to neck joint) by a
  monotone C1 map through the shoulder line and the neck joint;
* the shoulder girdle follows the trunk, is scaled sideways about the midline
  and shifted so the shoulder joint lands exactly;
* each limb segment is stretched along its bone about its proximal joint.

The pieces are blended with the stored skin weights. Bone directions stay
MakeHuman's rest directions; posing is the skinning step's job. The named
corrections ``hip_back_m`` and ``shoulder_anchor_m`` (``fit_settings.json``)
place the skeleton in the flesh.

Frame: hip-midpoint origin, x forward, y left, z up, metres. Appearance only:
nothing here reads or changes a physics model. Opt-in; no default uses it yet.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

Array = NDArray[np.float64]
PointMap = Callable[[Array], Array]
ASSET_DIR = Path(__file__).resolve().parents[4] / "assets" / "body_skin_cc0"
SIDES: tuple[tuple[str, float], ...] = (("l", 1.0), ("r", -1.0))
ARM_GROUPS = ("clavicle", "shoulder", "upperarm", "forearm")
TRUNK_GROUPS = ("pelvis", "spine", "thorax")
#: Groups the chest scale leaves alone (the arm below the deltoid, the legs).
THORAX_EXEMPT_GROUPS = ("upperarm_distal", "forearm", "thigh", "shank")
#: Smallest chest scale accepted: a fifth off MakeHuman's chest depth.
MIN_THORAX_FRONT_SCALE = 0.8
#: Largest correction accepted, metres: a third of the pelvis depth at 1.82 m.
MAX_CORRECTION_M = 0.06


def _positive(owner: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(owner, name)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")


@dataclass(frozen=True)
class SegmentLengths:
    """A subject's joint-centre geometry in metres (heights from the hip midpoint)."""

    hip_half_spacing: float
    shoulder_half_spacing: float
    shoulder_height: float
    neck_height: float
    upper_arm: float
    forearm: float
    thigh: float
    shank: float

    def __post_init__(self) -> None:
        _positive(self, tuple(f.name for f in fields(self)))
        if self.neck_height <= self.shoulder_height:
            raise ValueError("neck_height must exceed shoulder_height")


@dataclass(frozen=True)
class FitSettings:
    """Named fit settings; values and sources live in ``fit_settings.json``."""

    hip_back_m: float
    hip_back_plateau_m: float
    hip_back_rise_m: float
    shoulder_anchor_m: tuple[float, float, float]
    trunk_margin_m: float
    warp_extent_m: float
    thorax_front_scale: float
    thorax_front_start_m: float
    thorax_front_rise_m: float
    thigh_back_m: float
    thigh_back_rise_fraction: float
    thigh_back_below_hip_m: float
    thigh_back_fade_m: float
    thigh_back_front_m: float

    def __post_init__(self) -> None:
        _positive(self, ("hip_back_plateau_m", "hip_back_rise_m", "trunk_margin_m"))
        _positive(self, ("thorax_front_start_m", "thorax_front_rise_m"))
        _positive(self, ("thigh_back_fade_m", "thigh_back_front_m"))
        if not MIN_THORAX_FRONT_SCALE <= self.thorax_front_scale <= 1.0:
            raise ValueError(
                f"thorax_front_scale outside [{MIN_THORAX_FRONT_SCALE}, 1]"
            )
        if not 0.0 <= self.thigh_back_m <= MAX_CORRECTION_M:
            raise ValueError(f"thigh_back_m outside [0, {MAX_CORRECTION_M}]")
        if not 0.0 < self.thigh_back_rise_fraction <= 1.0:
            raise ValueError("thigh_back_rise_fraction outside (0, 1]")
        if not self.thigh_back_below_hip_m >= 0.0:
            raise ValueError("thigh_back_below_hip_m must not be negative")
        if not 0.0 <= self.hip_back_m <= MAX_CORRECTION_M:
            raise ValueError(f"hip_back_m outside [0, {MAX_CORRECTION_M}]")
        anchor = np.asarray(self.shoulder_anchor_m, dtype=float)
        if anchor.shape != (3,) or not np.linalg.norm(anchor) <= MAX_CORRECTION_M:
            raise ValueError(
                f"shoulder_anchor_m must be 3 values within {MAX_CORRECTION_M} m"
            )
        if not self.warp_extent_m > self.trunk_margin_m:
            raise ValueError("warp_extent_m must exceed trunk_margin_m")


@dataclass(frozen=True)
class BodySkin:
    """The stored skin: metres, dense ``(N, groups)`` weights, named landmarks."""

    vertices: Array
    faces: NDArray[np.int64]
    weights: Array
    groups: tuple[str, ...]
    landmarks: dict[str, Array]


@dataclass(frozen=True)
class FittedSkin:
    """Fitted rest vertices, the joint centres and the per-group point maps."""

    vertices: Array
    joints: dict[str, Array]
    maps: dict[str, PointMap]

    def map_points(self, group: str, points: Array) -> Array:
        """Where the fit (before the hip correction) carries ``points`` of ``group``."""
        return self.maps[group](np.asarray(points, dtype=float))


def load_fit_settings(
    path: Path | None = None, profile: str | None = None
) -> FitSettings:
    """Read ``fit_settings.json``, optionally with one named profile applied.

    A profile overrides named settings; without one the stored defaults are
    returned unchanged. Raises ``ValueError`` for an unknown profile, a profile
    entry that is not a fit setting, or invalid values.
    """
    raw = json.loads((path or ASSET_DIR / "fit_settings.json").read_text())
    names = {f.name for f in fields(FitSettings)}
    values = {name: raw[name] for name in names}
    if profile is not None:
        profiles = raw.get("profiles", {})
        if profile not in profiles:
            raise ValueError(f"unknown fit profile {profile!r}")
        extra = sorted(set(profiles[profile]) - names)
        if extra:
            raise ValueError(f"{extra[0]} in profile {profile!r} is not a fit setting")
        values.update(profiles[profile])
    anchor = values.pop("shoulder_anchor_m")
    if len(anchor) != 3:
        raise ValueError("shoulder_anchor_m must hold three values")
    return FitSettings(shoulder_anchor_m=(anchor[0], anchor[1], anchor[2]), **values)


def load_body_skin() -> BodySkin:
    """Load the asset; fails closed (``ValueError``) on a provenance mismatch."""
    settings = json.loads((ASSET_DIR / "skin_settings.json").read_text())
    record = json.loads((ASSET_DIR / "PROVENANCE.json").read_text())
    blob = (ASSET_DIR / record["asset"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != record["asset_sha256"]:
        raise ValueError("body skin asset does not match its provenance record")
    groups = list(settings["central_groups"])
    groups += [
        f"{name}_{side}" for side, _ in SIDES for name in settings["side_groups"]
    ]
    with np.load(ASSET_DIR / record["asset"]) as data:
        quantum = settings["quantum_m"]
        weights = np.zeros((len(data["vertices"]), len(groups)))
        rows = np.arange(len(weights))[:, None]
        np.add.at(weights, (rows, data["weight_index"]), data["weight_value"])
        marks = data["landmarks"] * quantum
        return BodySkin(
            vertices=data["vertices"] * quantum,
            faces=data["faces"].astype(np.int64),
            weights=weights / settings["weight_levels"],
            groups=tuple(groups),
            landmarks=dict(zip(settings["landmarks"], marks, strict=True)),
        )


def _unit(vector: Array) -> Array:
    return vector / float(np.linalg.norm(vector))


def _stretch(
    near: Array, far: Array, target_near: Array, length: float
) -> tuple[PointMap, Array]:
    """Stretch along ``near -> far`` about ``near``: the map and the far joint."""
    axis = _unit(far - near)
    gain = length / float(np.linalg.norm(far - near)) - 1.0

    def apply(points: Array) -> Array:
        local = points - near
        return target_near + local + gain * np.outer(local @ axis, axis)

    return apply, target_near + length * axis


def _smoothstep(t: Array) -> Array:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _group_maps(
    marks: dict[str, Array], lengths: SegmentLengths, settings: FitSettings
) -> tuple[dict[str, PointMap], dict[str, Array]]:
    """Per-group-prefix maps and the fitted joints (see the module docstring)."""
    hip_mid = 0.5 * (marks["hip_l"] + marks["hip_r"])
    axis = _unit(marks["neck"] - hip_mid)
    shoulder_mid = 0.5 * (marks["shoulder_l"] + marks["shoulder_r"])
    u_sh, u_neck = (
        float((shoulder_mid - hip_mid) @ axis),
        float((marks["neck"] - hip_mid) @ axis),
    )
    off_axis = (
        shoulder_mid - hip_mid - u_sh * axis
    )  # shoulder line ahead of the trunk axis
    h_sh = math.sqrt(lengths.shoulder_height**2 - float(off_axis @ off_axis))
    margin, far = settings.trunk_margin_m, settings.warp_extent_m
    grow = lengths.neck_height - u_neck
    warp = PchipInterpolator(
        [-far, -margin, 0.0, u_sh, u_neck, u_neck + margin, u_neck + far],
        [
            -far,
            -margin,
            0.0,
            h_sh,
            u_neck + grow,
            u_neck + margin + grow,
            u_neck + far + grow,
        ],
    )

    def trunk(points: Array) -> Array:
        u = (points - hip_mid) @ axis
        return points + np.outer(warp(u) - u, axis)

    maps: dict[str, PointMap] = {"trunk": trunk}
    joints = {"neck": trunk(marks["neck"][None])[0]}
    for side, sign in SIDES:
        lateral = np.array([0.0, sign, 0.0])
        wide = lengths.shoulder_half_spacing / abs(
            float(marks[f"shoulder_{side}"][1] - hip_mid[1])
        )

        def girdle(points: Array, wide: float = wide) -> Array:
            out = trunk(points)
            out[:, 1] = hip_mid[1] + wide * (out[:, 1] - hip_mid[1])
            return out

        joints[f"shoulder_{side}"] = shoulder = girdle(marks[f"shoulder_{side}"][None])[
            0
        ]
        maps[f"clavicle_{side}"] = maps[f"shoulder_{side}"] = girdle
        joints[f"hip_{side}"] = hip = hip_mid + lateral * lengths.hip_half_spacing
        chain = (
            ("upperarm", "shoulder", "elbow", shoulder, lengths.upper_arm),
            ("forearm", "elbow", "wrist", None, lengths.forearm),
            ("thigh", "hip", "knee", hip, lengths.thigh),
            ("shank", "knee", "ankle", None, lengths.shank),
        )
        end = shoulder
        for group, near, far_name, start, length in chain:
            maps[f"{group}_{side}"], end = _stretch(
                marks[f"{near}_{side}"],
                marks[f"{far_name}_{side}"],
                end if start is None else start,
                length,
            )
            joints[f"{far_name}_{side}"] = end
    return maps, joints


def _group_weight(skin: BodySkin, prefixes: tuple[str, ...]) -> Array:
    columns = [group.startswith(prefixes) for group in skin.groups]
    return np.clip(skin.weights[:, columns].sum(axis=1), 0.0, 1.0)


def _sagittal_corrections(
    vertices: Array, skin: BodySkin, joints: dict[str, Array], settings: FitSettings
) -> None:
    """Move ``vertices`` fore and aft in place: chest, thigh front, then hips.

    Each correction changes x only and leaves the fitted joints where they are.
    """
    hip_mid = 0.5 * (joints["hip_l"] + joints["hip_r"])
    hip_z, knee_z = joints["hip_l"][2], joints["knee_l"][2]
    if settings.thorax_front_scale != 1.0:
        axis = _unit(joints["neck"] - hip_mid)
        along = (vertices - hip_mid) @ axis
        ahead = vertices[:, 0] - (hip_mid[0] + along * axis[0])
        window = _smoothstep(
            (along - settings.thorax_front_start_m) / settings.thorax_front_rise_m
        ) * (1.0 - _group_weight(skin, THORAX_EXEMPT_GROUPS))
        gain = (settings.thorax_front_scale - 1.0) * window
        vertices[:, 0] += gain * np.clip(ahead, 0.0, None)
    if settings.thigh_back_m:
        height, ahead = vertices[:, 2], vertices[:, 0] - joints["hip_l"][0]
        rise = settings.thigh_back_rise_fraction * (hip_z - knee_z)
        fade = (height - hip_z + settings.thigh_back_below_hip_m) / (
            settings.thigh_back_fade_m
        )
        window = _smoothstep((height - knee_z) / rise) * (1.0 - _smoothstep(fade))
        window *= _smoothstep(ahead / settings.thigh_back_front_m)
        vertices[:, 0] -= (
            settings.thigh_back_m * window * _group_weight(skin, ("thigh",))
        )
    if settings.hip_back_m:
        hip_z, knee_z = joints["hip_l"][2], joints["knee_l"][2]
        height = vertices[:, 2]
        window = _smoothstep((height - knee_z) / (hip_z - knee_z)) * (
            1.0
            - _smoothstep(
                (height - hip_z - settings.hip_back_plateau_m)
                / settings.hip_back_rise_m
            )
        )
        arm = _group_weight(skin, ARM_GROUPS)
        vertices[:, 0] -= settings.hip_back_m * window * (1.0 - arm)


def fit_body_skin(
    skin: BodySkin, lengths: SegmentLengths, settings: FitSettings
) -> FittedSkin:
    """Reshape ``skin`` so its landmarks sit on the subject's joint centres.

    Postcondition: every distance between fitted joints equals the subject's
    segment length, and each segment's map carries its two landmarks onto them.
    The faces and weights of ``skin`` apply unchanged to the result.
    """
    marks = {name: point.copy() for name, point in skin.landmarks.items()}
    forward, outward, up = settings.shoulder_anchor_m
    for side, sign in SIDES:
        marks[f"shoulder_{side}"] += np.array([forward, sign * outward, up])
    prefix_maps, joints = _group_maps(marks, lengths, settings)
    maps: dict[str, PointMap] = {}
    vertices = np.zeros_like(skin.vertices)
    for column, group in enumerate(skin.groups):
        trunk = group.startswith(TRUNK_GROUPS)
        key = "trunk" if trunk else f"{group.split('_')[0]}_{group[-1]}"
        maps[group] = prefix_maps[key]
        used = skin.weights[:, column] > 0.0
        vertices[used] += skin.weights[used, column, None] * maps[group](
            skin.vertices[used]
        )
    _sagittal_corrections(vertices, skin, joints, settings)
    return FittedSkin(vertices=vertices, joints=joints, maps=maps)
