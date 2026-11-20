"""Dress the fitted CC0 body skin: skin, shirt and trousers layers (#15299).

The clothes are the body surface itself, cut on smooth level sets (collar,
sleeve end, waist, trouser end) and moved outward along the surface normal by
a named ease, so they inherit the skin weights and follow every joint without
a second rig. Each hem is folded back onto the skin, and the bare skin is kept
``skin_margin_m`` under the cloth so no gap opens when the body bends. The
forearm ends on a plane ``wrist_cut_m`` before the wrist joint: a clean ring
for the separate hand model's cuff to lie over.

``dress_skin`` returns the layers in the fitted rest pose; values and sources
are in ``assets/body_skin_cc0/dress_settings.json``. Appearance only; opt-in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.shared.python.model_appearance.body_skin_mesh import (
    Faces,
    boundary_edges,
    clip_mesh,
    relax,
    vertex_normals,
)

from src.shared.python.model_appearance.body_skin_fit import (
    ASSET_DIR,
    SIDES,
    BodySkin,
    FittedSkin,
)

Array = NDArray[np.float64]
ARM, LEG = ("upperarm", "forearm"), ("thigh", "shank")


@dataclass(frozen=True)
class ClothSettings:
    """One garment: ease over the skin, its floor, and the smoothing passes."""

    ease_m: float
    min_ease_m: float
    smoothing_passes: int
    relax_passes: int

    def __post_init__(self) -> None:
        if not 0.0 < self.min_ease_m <= self.ease_m:
            raise ValueError("need 0 < min_ease_m <= ease_m")
        if self.smoothing_passes < 0 or self.relax_passes < 0:
            raise ValueError("passes must not be negative")


@dataclass(frozen=True)
class DressSettings:
    """Named cut and ease settings; sources live in ``dress_settings.json``."""

    shirt: ClothSettings
    trousers: ClothSettings
    skin_margin_m: float
    waist_height_m: float
    tuck_m: float
    sleeve_fraction: float
    trouser_end_fraction: float
    collar_gap_m: float
    collar_front_drop_m: float
    collar_front_reach_m: float
    neck_hug_m: float
    neck_hug_gain: float
    tuck_ease_m: tuple[float, float]
    tuck_band_m: tuple[float, float]
    waistband_m: float
    waistband_span_m: tuple[float, float]
    wrist_cut_m: float

    def __post_init__(self) -> None:
        positive: tuple[str, ...]
        positive = ("skin_margin_m", "waist_height_m", "tuck_m", "collar_gap_m")
        positive += ("wrist_cut_m",)
        positive += ("collar_front_reach_m", "neck_hug_m", "waistband_m")
        for name in positive:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("sleeve_fraction", "trouser_end_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.collar_front_drop_m < 0.0 or self.neck_hug_gain < 0.0:
            raise ValueError("collar_front_drop_m and neck_hug_gain must be >= 0")
        for name in ("tuck_ease_m", "tuck_band_m", "waistband_span_m"):
            pair = getattr(self, name)
            if len(pair) != 2 or min(pair) <= 0.0:
                raise ValueError(f"{name} must hold two positive values")


@dataclass(frozen=True)
class Layer:
    """One drawn surface in the fitted rest pose, with its skin weights."""

    vertices: Array
    weights: Array
    faces: Faces


def load_dress_settings(path: Path | None = None) -> DressSettings:
    """Read ``dress_settings.json``; raises ``ValueError`` for invalid values."""
    raw = json.loads((path or ASSET_DIR / "dress_settings.json").read_text())
    values = {f.name: raw[f.name] for f in fields(DressSettings)}
    for name in ("shirt", "trousers"):
        values[name] = ClothSettings(**values[name])
    for name in ("tuck_ease_m", "tuck_band_m", "waistband_span_m"):
        values[name] = tuple(values[name])
    return DressSettings(**values)


@dataclass(frozen=True)
class _Fields:
    """Per-vertex fields of the body the cuts and eases are written in."""

    height: Array  # along the trunk axis from the hip midpoint, metres
    trunk: Array  # 1 off the limbs
    shirt: Array  # level sets: positive where the garment covers the skin
    trousers: Array
    collar: Array
    wrist: Array  # positive up the forearm from the clean wrist cut

    def stack(self) -> Array:
        return np.column_stack([getattr(self, f.name) for f in fields(self)])


def _limb(
    skin: BodySkin, fitted: FittedSkin, groups: tuple[str, ...], bone: tuple[str, str]
) -> tuple[NDArray[np.bool_], Array]:
    """Vertices a limb owns and their distance along ``bone`` from its start."""
    owned = np.zeros(len(fitted.vertices), dtype=bool)
    along = np.zeros(len(fitted.vertices))
    for side, _ in SIDES:
        columns = [g.startswith(groups) and g.endswith(side) for g in skin.groups]
        mine = skin.weights[:, columns].sum(axis=1) > 0.5
        start, end = (fitted.joints[f"{name}_{side}"] for name in bone)
        axis = (end - start) / float(np.linalg.norm(end - start))
        along[mine] = ((fitted.vertices - start) @ axis)[mine]
        owned |= mine
    return owned, along


def _collar(skin: BodySkin, fitted: FittedSkin, settings: DressSettings) -> Array:
    """Distance below the collar line: a loop round the neck base, lower in front."""
    high = skin.vertices[skin.faces][:, :, 2].mean(axis=1) >= skin.landmarks["neck"][2]
    upper, lower = (np.zeros(len(skin.vertices), dtype=bool) for _ in range(2))
    upper[skin.faces[high]] = True
    lower[skin.faces[~high]] = True
    ring = fitted.vertices[upper & lower]
    if not len(ring):
        raise ValueError("the skin has no neck above the neck joint")
    distance = np.asarray(cKDTree(ring).query(fitted.vertices)[0], dtype=float)
    distance[upper & ~lower] *= -1.0
    ahead = (fitted.vertices[:, 0] - ring[:, 0].mean()) / settings.collar_front_reach_m
    drop = settings.collar_front_drop_m * np.clip(ahead, 0.0, 1.0)
    return distance - settings.collar_gap_m - drop


def _fields(skin: BodySkin, fitted: FittedSkin, settings: DressSettings) -> _Fields:
    joints = fitted.joints
    hip_mid = 0.5 * (joints["hip_l"] + joints["hip_r"])
    axis = joints["neck"] - hip_mid
    height = (fitted.vertices - hip_mid) @ (axis / float(np.linalg.norm(axis)))
    arm, sleeve = _limb(skin, fitted, ARM, ("shoulder", "elbow"))
    leg, shank = _limb(skin, fitted, LEG, ("knee", "ankle"))
    upper_arm = float(np.linalg.norm(joints["elbow_l"] - joints["shoulder_l"]))
    shank_length = float(np.linalg.norm(joints["ankle_l"] - joints["knee_l"]))
    hand_end, forearm = _limb(skin, fitted, ("forearm",), ("elbow", "wrist"))
    reach = float(np.linalg.norm(joints["wrist_l"] - joints["elbow_l"]))
    wrist = np.where(hand_end, reach - settings.wrist_cut_m - forearm, 1.0)
    trunk = ~arm & ~leg
    waist = settings.waist_height_m
    collar = _collar(skin, fitted, settings)
    shirt = np.where(trunk, height - (waist - settings.tuck_m), -1.0)
    shirt[arm] = (settings.sleeve_fraction * upper_arm - sleeve)[arm]
    trousers = np.where(trunk, waist - height, -1.0)
    trousers[leg] = (settings.trouser_end_fraction * shank_length - shank)[leg]
    shirt = np.minimum(shirt, collar)
    return _Fields(height, trunk.astype(float), shirt, trousers, collar, wrist)


def _garment(
    body: Array, cut: _Fields, faces: Faces, cloth: ClothSettings, bounds: Array
) -> Layer:
    """Offset the cut body by ``bounds`` (ease, floor, cap) and fold the hem in."""
    points, width = body[:, :3], len(fields(_Fields))
    ease, floor, cap = bounds.T
    hem = boundary_edges(faces)
    fixed = np.zeros(len(points), dtype=bool)
    fixed[hem] = True
    base = points
    for _ in range(cloth.smoothing_passes):  # Taubin: smooth without shrinking
        base = relax(relax(base, faces, fixed, 0.5), faces, fixed, -0.53)
    normals = vertex_normals(base, faces)
    cloth_points = base + normals * np.minimum(ease, cap)[:, None]
    for _ in range(cloth.relax_passes):
        cloth_points = relax(cloth_points, faces, fixed, 0.5)
        depth = np.einsum("ij,ij->i", cloth_points - base, normals)
        cloth_points += normals * (np.clip(depth, floor, cap) - depth)[:, None]
    outward = vertex_normals(points, faces)
    depth = np.einsum("ij,ij->i", cloth_points - points, outward)
    cloth_points += outward * np.clip(floor - depth, 0.0, None)[:, None]
    rim = np.unique(hem)
    under = np.full(len(points), -1)
    under[rim] = len(points) + np.arange(len(rim))
    fold = [[b, a, under[a]] for a, b in hem] + [
        [b, under[a], under[b]] for a, b in hem
    ]
    weights = body[:, 3:-width]
    return Layer(
        np.vstack([cloth_points, points[rim]]),
        np.vstack([weights, weights[rim]]),
        np.vstack([faces, np.asarray(fold, dtype=np.int64)]),
    )


def dress_skin(
    skin: BodySkin, fitted: FittedSkin, weights: Array, settings: DressSettings
) -> dict[str, Layer]:
    """The ``skin``, ``shirt`` and ``trousers`` layers of the fitted body.

    ``weights`` are the skin weights to carry (the stored ones or the relaxed
    shoulder weights), ``(vertices, groups)``. Postcondition: each garment lies
    at least its floor ease outside the skin, and its open edges lie on the skin.
    """
    if weights.shape != skin.weights.shape:
        raise ValueError("weights must be (vertices, groups) of the skin")
    cut = _fields(skin, fitted, settings)
    body = np.column_stack([fitted.vertices, weights, cut.stack()])
    width = len(fields(_Fields))
    bare = settings.skin_margin_m - np.maximum(cut.shirt, cut.trousers)
    bare = np.minimum(bare, cut.wrist)  # a clean ring instead of the stored ragged rim
    part, faces = clip_mesh(body, skin.faces, bare)
    layers = {"skin": Layer(part[:, :3], part[:, 3:-width], faces)}
    for name, level in (("shirt", cut.shirt), ("trousers", cut.trousers)):
        part, faces = clip_mesh(body, skin.faces, level)
        here = _Fields(*part[:, -width:].T)
        cloth: ClothSettings = getattr(settings, name)
        ease = np.full(len(part), cloth.ease_m)
        floor = np.full(len(part), cloth.min_ease_m)
        cap = np.full(len(part), np.inf)
        if name == "shirt":  # thin under the waistband, close at the neck
            above, blend = settings.tuck_band_m
            band = np.clip(
                (settings.waist_height_m + above - here.height) / blend, 0, 1
            )
            loose, tight = settings.tuck_ease_m
            tucked = (band * here.trunk) > 0.0
            cap[tucked] = (loose - (loose - tight) * band)[tucked]
            hug = settings.neck_hug_m + settings.neck_hug_gain * here.collar
            cap = np.minimum(cap, np.maximum(hug, settings.neck_hug_m))
            floor = np.minimum(floor, cap)
        else:  # the waistband stands off the tucked shirt
            below, rise = settings.waistband_span_m
            lift = (here.height - (settings.waist_height_m - below)) / rise
            floor += settings.waistband_m * np.clip(lift, 0.0, 1.0) * here.trunk
            ease = np.maximum(ease, floor)
        bounds = np.column_stack([ease, floor, cap])
        layers[name] = _garment(part, here, faces, cloth, bounds)
    return layers
