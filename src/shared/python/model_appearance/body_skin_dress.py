"""Dress the fitted CC0 body skin: skin, shirt and trousers layers (#15299).

The clothes are the body surface itself, cut on smooth level sets (collar,
sleeve end, waist, trouser end) and moved outward along the surface normal by
a named ease, so they inherit the skin weights and follow every joint without
a second rig. Each hem is folded back onto the skin, and the bare skin is kept
``skin_margin_m`` under the cloth so no gap opens when the body bends. The
collar is measured from the line where the skin crosses the neck joint's
height, not from the mesh edges about it, and collar and sleeve hems are held
close to the skin (``neck_hug_m``, ``sleeve_hug_m``). The forearm ends on a plane ``wrist_cut_m`` before the wrist joint: a clean ring
for the separate hand model's cuff to lie over. The neck's open end is drawn in
under the head model and closed with a plug that moves with that end
(``neck_cap``), so the join follows the skin and cannot open when the head turns.

Three steps keep the anatomy from showing through the cloth: the chest of the
body itself is evened out (``chest``), the shirt is pulled taut over the chest
and the sleeve caps (relaxed down to its floor ease), and each trouser leg is
held off the leg by a tube of at least ``leg_tube_m`` about the shank.

The shirt is draped first and the trousers' top is set on the normals through
its cloth, ``waist_gap_m`` outside it, so the tucked shirt and the waistband
move as one sheet when the trunk turns and the hips bend.

``dress_skin`` returns the layers in the fitted rest pose; values and sources
are in ``assets/body_skin_cc0/dress_settings.json``. Appearance only; opt-in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.shared.python.model_appearance.body_skin_mesh import (
    Faces,
    boundary_edges,
    clip_mesh,
    relax,
    vertex_areas,
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
class ChestSettings:
    """The chest front: where it is, how far it is evened out and pulled taut."""

    from_m: float
    above_shoulders_m: float
    half_width_m: float
    ahead_m: float
    edge_m: float
    smooth_sigma_m: float
    tent_passes: int

    def __post_init__(self) -> None:
        for name in ("from_m", "half_width_m", "ahead_m", "edge_m"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"chest {name} must be positive")
        if min(self.above_shoulders_m, self.smooth_sigma_m, self.tent_passes) < 0:
            raise ValueError("chest reach, sigma and passes must not be negative")


@dataclass(frozen=True)
class NeckCapSettings:
    """The plug on the neck's end under the head; ``apex_height_m`` 0: none."""

    apex_forward_m: float
    apex_height_m: float
    tuck_m: float
    tuck_span_m: float
    lip_m: float
    lip_passes: int

    def __post_init__(self) -> None:
        if self.apex_height_m < 0.0 or not self.tuck_span_m > 0.0:
            raise ValueError("neck_cap needs apex_height_m >= 0 and tuck_span_m > 0")
        if min(self.tuck_m, self.lip_m, self.lip_passes) < 0:
            raise ValueError("neck_cap needs tuck_m, lip_m and lip_passes >= 0")


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
    sleeve_hug_m: float
    sleeve_hug_gain: float
    tuck_ease_m: tuple[float, float]
    tuck_band_m: tuple[float, float]
    waistband_m: float
    waistband_span_m: tuple[float, float]
    waist_gap_m: float
    wrist_cut_m: float
    chest: ChestSettings
    sleeve_cap_m: float
    sleeve_cap_passes: int
    leg_tube_m: tuple[float, float]
    leg_tube_rise: float
    neck_cap: NeckCapSettings

    def __post_init__(self) -> None:
        if len(self.leg_tube_m) != 2 or min(self.leg_tube_m) < 0.0:
            raise ValueError("leg_tube_m must hold two radii, none negative")
        if not (self.leg_tube_rise > 0.0 and self.sleeve_cap_m > 0.0):
            raise ValueError("leg_tube_rise and sleeve_cap_m must be positive")
        if self.sleeve_cap_passes < 0:
            raise ValueError("sleeve_cap_passes must not be negative")
        positive: tuple[str, ...]
        positive = ("skin_margin_m", "waist_height_m", "tuck_m", "collar_gap_m")
        positive += ("wrist_cut_m", "sleeve_hug_m")
        positive += ("collar_front_reach_m", "neck_hug_m", "waistband_m", "waist_gap_m")
        for name in positive:
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("sleeve_fraction", "trouser_end_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if min(self.collar_front_drop_m, self.neck_hug_gain, self.sleeve_hug_gain) < 0:
            raise ValueError("collar_front_drop_m and the hug gains must be >= 0")
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
    for name in ("tuck_ease_m", "tuck_band_m", "waistband_span_m", "leg_tube_m"):
        values[name] = tuple(values[name])
    values["chest"] = ChestSettings(**values["chest"])
    values["neck_cap"] = NeckCapSettings(**values["neck_cap"])
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
    ahead: Array  # forward of the trunk axis, metres
    chest: Array  # 1 on the chest front, 0 off it
    cap: Array  # 1 within sleeve_cap_m of a shoulder joint

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


def _neck_line(skin: BodySkin, fitted: FittedSkin) -> tuple[Array, Array, Array]:
    """Where the skin crosses the neck joint's height: segment ends and heights.

    One segment per face the height cuts, placed on the fitted body, and each
    stored vertex's height over the joint. Raises ``ValueError`` without a neck.
    """
    over = skin.vertices[:, 2] - skin.landmarks["neck"][2]
    high = over > 0.0
    cut = skin.faces[high[skin.faces].any(axis=1) & ~high[skin.faces].all(axis=1)]
    if not len(cut):
        raise ValueError("the skin has no neck above the neck joint")
    first, second = cut, np.roll(cut, -1, axis=1)
    share = over[first] / np.where(
        high[first] != high[second], over[first] - over[second], 1.0
    )
    points = fitted.vertices[first]
    points = points + share[:, :, None] * (fitted.vertices[second] - points)
    ends = points[high[first] != high[second]].reshape(-1, 2, 3)  # two per face
    return ends[:, 0], ends[:, 1], over


def _collar(skin: BodySkin, fitted: FittedSkin, settings: DressSettings) -> Array:
    """Distance below the collar line: a loop round the neck base, lower in front."""
    start, end, over = _neck_line(skin, fitted)
    span = end - start
    offset = fitted.vertices[:, None, :] - start[None, :, :]
    share = np.einsum("vsk,sk->vs", offset, span) / np.maximum((span**2).sum(1), 1e-18)
    nearest = offset - np.clip(share, 0.0, 1.0)[:, :, None] * span[None, :, :]
    distance = np.linalg.norm(nearest, axis=2).min(axis=1)
    distance[over > 0.0] *= -1.0
    centre = 0.5 * float((start[:, 0] + end[:, 0]).mean())
    ahead = (fitted.vertices[:, 0] - centre) / settings.collar_front_reach_m
    ahead = np.clip(ahead, 0.0, 1.0)
    drop = settings.collar_front_drop_m * ahead**2 * (3.0 - 2.0 * ahead)  # no corner
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
    local, chest = fitted.vertices - hip_mid, settings.chest
    unit = axis / float(np.linalg.norm(axis))
    ahead = local[:, 0] - height * unit[0]
    top = float(np.mean([(joints[f"shoulder_{s}"] - hip_mid) @ unit for s, _ in SIDES]))
    front = np.ones(len(height))
    for inside in (
        height - chest.from_m,
        top + chest.above_shoulders_m - height,
        chest.half_width_m - np.abs(local[:, 1]),
        ahead - chest.ahead_m,
    ):
        front *= np.clip(inside / chest.edge_m, 0.0, 1.0)
    spans = [
        np.linalg.norm(fitted.vertices - joints[f"shoulder_{side}"], axis=1)
        for side, _ in SIDES
    ]
    cap = (np.min(spans, axis=0) < settings.sleeve_cap_m).astype(float)
    return _Fields(
        height,
        trunk.astype(float),
        shirt,
        trousers,
        collar,
        wrist,
        ahead,
        front * trunk,
        cap,
    )


def _even_chest(
    points: Array, faces: Faces, cut: _Fields, chest: ChestSettings
) -> Array:
    """The body with its chest front moved to the local mean depth.

    Each chest vertex moves fore and aft towards the area-weighted mean depth
    of the trunk front within ``smooth_sigma_m`` (a Gaussian in the frontal
    plane), by its chest share: the pectoral relief goes, the depth stays.
    """
    move, source = cut.chest > 0.0, (cut.trunk > 0.5) & (cut.ahead > 0.0)
    if chest.smooth_sigma_m == 0.0 or not move.any():
        return points
    plane = np.column_stack([points[:, 1], cut.height])
    gap = plane[move][:, None, :] - plane[source][None, :, :]
    kernel = np.exp(-(gap**2).sum(axis=-1) / (2.0 * chest.smooth_sigma_m**2))
    kernel *= vertex_areas(points, faces)[source][None, :]
    mean = kernel @ cut.ahead[source] / kernel.sum(axis=1)
    out = points.copy()
    out[move, 0] += cut.chest[move] * (mean - cut.ahead[move])
    return out


def shaped_body(skin: BodySkin, fitted: FittedSkin, settings: DressSettings) -> Array:
    """The fitted body as it is dressed: its chest front evened out."""
    cut = _fields(skin, fitted, settings)
    return _even_chest(fitted.vertices, skin.faces, cut, settings.chest)


def _taut(
    cloth: Array,
    raw: Array,
    faces: Faces,
    free: NDArray[np.bool_],
    ease: tuple[Array, int],
) -> Array:
    """Relax the ``free`` cloth for ``ease[1]`` passes, never under the floor ``ease[0]``."""
    floor, passes = ease
    outward = vertex_normals(raw, faces)
    for _ in range(passes if free.any() else 0):
        cloth = relax(cloth, faces, ~free, 0.5)
        depth = np.einsum("ij,ij->i", cloth - raw, outward)
        cloth = cloth + outward * np.clip(floor - depth, 0.0, None)[:, None]
    return cloth


def _leg_tubes(
    cloth: Array, here: _Fields, fitted: FittedSkin, settings: DressSettings
) -> Array:
    """Hold each trouser leg at least the tube radius off its shank axis.

    The radius is ``leg_tube_m[0]`` at the knee and ``leg_tube_m[1]`` at the
    ankle and falls to nothing ``leg_tube_rise`` shank lengths above the knee.
    """
    out = cloth.copy()
    for side, sign in SIDES:
        knee, ankle = fitted.joints[f"knee_{side}"], fitted.joints[f"ankle_{side}"]
        bone = ankle - knee
        share = (out - knee) @ bone / float(bone @ bone)
        radial = out - knee - np.outer(share, bone)
        radius = np.maximum(np.linalg.norm(radial, axis=1), 1.0e-9)
        want = np.interp(
            share, [-settings.leg_tube_rise, 0.0, 1.0], [0.0, *settings.leg_tube_m]
        )
        grow = (here.trunk < 0.5) & (sign * out[:, 1] > 0.0) & (want > radius)
        out[grow] += radial[grow] * (want / radius - 1.0)[grow, None]
    return out


def _worn(shaped: Array, part: Array, draped: Array) -> Array:
    """Per body vertex: the shirt's cloth over it (else the skin), and whether worn."""
    gap, index = cKDTree(part[:, :3]).query(shaped)
    worn = np.asarray(gap) == 0.0  # the shirt's uncut vertices are the body's rows
    return np.column_stack([np.where(worn[:, None], draped[index], shaped), worn])


def _over_shirt(
    cloth: Array, under: Array, outward: Array, here: _Fields, settings: DressSettings
) -> Array:
    """The trousers' top held ``waist_gap_m`` outside the tucked shirt.

    From ``waistband_span_m[0]`` below the waistband top each vertex is
    put on the normal through the shirt's cloth over the same body point
    (``under``: that point and whether a shirt is worn there, interpolated on
    the cuts), at least the gap outside it, and below the shirt's tail outside
    the tail's ease too. Both garments carry that point's skin weights, so they
    move together when posed: the shirt stays tucked in (bounds in the tests).
    """
    band = here.height > settings.waist_height_m - settings.waistband_span_m[0]
    base, worn = under[:, :3], under[:, 3]
    need = settings.waist_gap_m + (1.0 - worn) * settings.tuck_ease_m[1]
    depth = np.einsum("ij,ij->i", cloth - base, outward)
    out = cloth.copy()
    out[band] = (base + outward * np.maximum(depth, need)[:, None])[band]
    return out


def _drape(points: Array, faces: Faces, cloth: ClothSettings, bounds: Array) -> Array:
    """The cut body ``points`` offset outward by ``bounds`` (ease, floor, cap)."""
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
    return cloth_points + outward * np.clip(floor - depth, 0.0, None)[:, None]


def _fold(body: Array, cloth_points: Array, faces: Faces) -> Layer:
    """The garment layer: the cloth with its hems folded back onto the skin."""
    points, width = body[:, :3], len(fields(_Fields))
    hem = boundary_edges(faces)
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


def close_neck(layer: Layer, neck: Array, cap: NeckCapSettings) -> Layer:
    """``layer`` with its open neck end drawn in and closed by a plug of its own.

    The end is the loop of open edges that reaches highest (z up); it must
    reach above the neck joint ``neck``. The plug's point lies
    ``apex_forward_m`` ahead of and ``apex_height_m`` above the joint, inside
    the head model. The end ring is moved ``tuck_m`` straight towards it, the
    skin about the ring less (smoothstep, none from ``tuck_span_m`` off the
    ring). The plug is a second ring, the end ring evened out along the loop
    (``lip_passes``) and set ``lip_m`` further towards the point, and the
    point; all of it carries the end ring's mean weights, so it follows the
    neck's end as one piece and cannot fold in itself when the head turns.
    Postcondition: that loop is closed with three faces per open edge; weights
    and faces of the layer's own vertices are kept. Raises ``ValueError`` for
    a layer without such an end.
    """
    if cap.apex_height_m == 0.0:
        return layer
    points, count = layer.vertices, len(layer.vertices)
    edges = boundary_edges(layer.faces)
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(count, count))
    loop = connected_components(graph, directed=False)[1][edges[:, 0]]
    top = int(np.argmax(points[edges[:, 0], 2])) if len(edges) else 0
    if not len(edges) or points[edges[top, 0], 2] <= neck[2]:
        raise ValueError("the layer has no open end above the neck joint")
    edges = edges[loop == loop[top]]
    ring, lip = np.unique(edges, return_inverse=True)
    first, second = edges.T
    before, after = lip.reshape(-1, 2).T
    apex = np.asarray(neck, float) + (cap.apex_forward_m, 0.0, cap.apex_height_m)
    reach = cKDTree(points[ring]).query(points)[0]
    share = np.clip(1.0 - reach / cap.tuck_span_m, 0.0, 1.0)
    towards = apex - points
    towards /= np.linalg.norm(towards, axis=1, keepdims=True)
    step = cap.tuck_m * share * share * (3.0 - 2.0 * share)
    drawn = points + step[:, None] * towards
    rim: Array = drawn[ring]
    for _ in range(cap.lip_passes):  # evened out along the loop
        ahead, behind = np.empty_like(rim), np.empty_like(rim)
        ahead[before], behind[after] = rim[after], rim[before]
        rim = 0.5 * rim + 0.25 * (ahead + behind)
    inward = apex - rim
    rim = rim + cap.lip_m * inward / np.linalg.norm(inward, axis=1, keepdims=True)
    whole = layer.weights[ring].mean(axis=0, keepdims=True)
    tip = np.full(len(edges), count + len(ring))
    before, after = before + count, after + count
    closing = [(second, first, before), (second, before, after), (after, before, tip)]
    return Layer(
        np.vstack([drawn, rim, apex]),
        np.vstack([layer.weights, np.repeat(whole, len(ring) + 1, axis=0)]),
        np.vstack([layer.faces, *(np.column_stack(face) for face in closing)]),
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
    shaped = _even_chest(fitted.vertices, skin.faces, cut, settings.chest)
    body = np.column_stack([shaped, weights, cut.stack()])
    width = len(fields(_Fields))
    bare = settings.skin_margin_m - np.maximum(cut.shirt, cut.trousers)
    bare = np.minimum(bare, cut.wrist)  # a clean ring instead of the stored ragged rim
    part, faces = clip_mesh(body, skin.faces, bare)
    bare_skin = Layer(part[:, :3], part[:, 3:-width], faces)
    layers = {"skin": close_neck(bare_skin, fitted.joints["neck"], settings.neck_cap)}
    under = np.column_stack([shaped, np.zeros(len(shaped))])
    for name, level in (("shirt", cut.shirt), ("trousers", cut.trousers)):
        part, faces = clip_mesh(np.column_stack([body, under]), skin.faces, level)
        part, lining = part[:, :-4], part[:, -4:]
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
            sleeve = settings.sleeve_hug_m + settings.sleeve_hug_gain * here.shirt
            arm = here.trunk < 0.5
            cap[arm] = np.minimum(cap, np.maximum(sleeve, settings.sleeve_hug_m))[arm]
            floor = np.minimum(floor, cap)
        else:  # the waistband stands off the tucked shirt
            below, rise = settings.waistband_span_m
            lift = (here.height - (settings.waist_height_m - below)) / rise
            floor += settings.waistband_m * np.clip(lift, 0.0, 1.0) * here.trunk
            ease = np.maximum(ease, floor)
        bounds = np.column_stack([ease, floor, cap])
        draped = _drape(part[:, :3], faces, cloth, bounds)
        if name == "shirt":
            inner = np.ones(len(part), dtype=bool)
            inner[boundary_edges(faces)] = False
            for free, passes in (
                (here.chest > 0.0, settings.chest.tent_passes),
                (here.cap > 0.5, settings.sleeve_cap_passes),
            ):
                draped = _taut(
                    draped, part[:, :3], faces, free & inner, (floor, passes)
                )
            under = _worn(shaped, part, draped)
        else:
            draped = _leg_tubes(draped, here, fitted, settings)
            outward = vertex_normals(part[:, :3], faces)
            draped = _over_shirt(draped, lining, outward, here, settings)
        layers[name] = _fold(part, draped, faces)
    return layers
