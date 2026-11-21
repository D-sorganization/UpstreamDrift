"""Clothing layers of the CC0 body skin on the default subject (#15299)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance import body_skin_dress as dress
from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance import body_skin_mesh as mesh
from src.shared.python.model_appearance import body_skin_pose as pose
from tests.unit.model_appearance.test_body_skin_fit import _subject
from tests.unit.model_appearance.test_body_skin_pose import GIRDLE, RAW, _girdle

pytestmark = pytest.mark.unit

SETTINGS = dress.load_dress_settings()
SKIN = fit.load_body_skin()
FITTED = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings())
LAYERS = dress.dress_skin(SKIN, FITTED, SKIN.weights, SETTINGS)
LIMITS = json.loads((fit.ASSET_DIR / "dress_settings.json").read_text())["limits"]
#: The same clothes without the chest, sleeve-cap and trouser-leg steps.
PLAIN = dataclasses.replace(
    SETTINGS,
    chest=dataclasses.replace(SETTINGS.chest, smooth_sigma_m=0.0, tent_passes=0),
    sleeve_cap_passes=0,
    leg_tube_m=(0.0, 0.0),
)
PLAIN_LAYERS = dress.dress_skin(SKIN, FITTED, SKIN.weights, PLAIN)
TUBELESS = dress.dress_skin(
    SKIN, FITTED, SKIN.weights, dataclasses.replace(SETTINGS, leg_tube_m=(0.0, 0.0))
)["trousers"]
#: Slack of the sampled-surface measure below (curvature between samples).
SAMPLE_SLACK_M = 0.0015


def _body_samples() -> tuple[cKDTree, np.ndarray, np.ndarray]:
    """Points on the dressed body (vertices, edge midpoints, centroids) and normals."""
    points, faces = dress.shaped_body(SKIN, FITTED, SETTINGS), SKIN.faces
    normals = dress.vertex_normals(points, faces)
    picks = [faces[:, [k]] for k in range(3)] + [faces[:, [0, 1]], faces[:, [1, 2]]]
    picks += [faces[:, [2, 0]], faces]
    where = np.vstack([points[p].mean(axis=1) for p in picks])
    facing = np.vstack([normals[p].mean(axis=1) for p in picks])
    facing /= np.linalg.norm(facing, axis=1, keepdims=True)
    return cKDTree(where), where, facing


TREE, WHERE, FACING = _body_samples()


def _height(points: np.ndarray) -> np.ndarray:
    """Signed height of ``points`` over the body surface at the nearest sample."""
    nearest = TREE.query(points)[1]
    return np.einsum("ij,ij->i", points - WHERE[nearest], FACING[nearest])


def _rim(layer: dress.Layer) -> np.ndarray:
    return np.unique(dress.boundary_edges(layer.faces))


#: In a fold (armpit, groin) the nearest skin sample lies across the fold.
FOLD_FLOOR_M = 0.002
#: Where the two trouser legs meet each other instead of the skin.
SEAM_HALF_WIDTH_M = 0.02


@pytest.mark.parametrize("name", ["shirt", "trousers"])
def test_cloth_lies_outside_the_skin_by_its_minimum_ease(name: str) -> None:
    layer, cloth = LAYERS[name], getattr(SETTINGS, name)
    cover = np.setdiff1d(np.arange(len(layer.vertices)), _rim(layer))
    points = layer.vertices[cover]
    if name == "shirt":  # the part not tucked under the trousers
        seen = points[:, 2] > SETTINGS.waist_height_m + SETTINGS.tuck_band_m[0]
        floor = min(cloth.min_ease_m, SETTINGS.neck_hug_m)
    else:
        seen = (np.abs(points[:, 1]) > SEAM_HALF_WIDTH_M) | (points[:, 2] > 0.0)
        floor = cloth.min_ease_m
    assert seen.sum() > 1000
    height = _height(points[seen])
    assert height.min() >= FOLD_FLOOR_M
    assert np.percentile(height, 2.0) >= floor - SAMPLE_SLACK_M
    assert np.median(height) >= cloth.min_ease_m
    assert height.max() < 0.05


@pytest.mark.parametrize("name", ["shirt", "trousers"])
def test_every_hem_is_folded_onto_the_skin(name: str) -> None:
    layer = LAYERS[name]
    rim = _rim(layer)
    loops = {"shirt": 4, "trousers": 3}[name]  # collar, sleeves, tail / waist, legs
    assert len(rim) > 30 * loops
    assert np.abs(_height(layer.vertices[rim])).max() <= SAMPLE_SLACK_M
    edges = dress.boundary_edges(layer.faces)
    assert np.array_equal(np.sort(edges[:, 0]), np.sort(edges[:, 1]))  # closed loops


def test_layers_carry_normalised_weights_and_the_skin_is_bare_where_expected() -> None:
    for layer in LAYERS.values():
        assert layer.weights.shape == (len(layer.vertices), len(SKIN.groups))
        assert np.allclose(layer.weights.sum(axis=1), 1.0, atol=1e-9)
        assert layer.faces.max() == len(layer.vertices) - 1
    bare = LAYERS["skin"].vertices
    joints = FITTED.joints
    assert bare[:, 2].max() > joints["neck"][2]  # the neck
    assert bare[:, 1].max() > joints["elbow_l"][1]  # the forearms
    waist = joints["hip_l"][2] + SETTINGS.waist_height_m
    chest = (np.abs(bare[:, 1]) < 0.1) & (np.abs(bare[:, 2] - waist) < 0.1)
    assert not chest.any()
    assert LAYERS["shirt"].vertices[:, 2].min() < waist < bare[:, 2].max()
    assert LAYERS["trousers"].vertices[:, 2].max() > waist


@pytest.mark.parametrize("side", ["l", "r"])
def test_the_forearm_ends_in_a_clean_closed_ring_before_the_wrist(side: str) -> None:
    elbow, wrist = FITTED.joints[f"elbow_{side}"], FITTED.joints[f"wrist_{side}"]
    reach = float(np.linalg.norm(wrist - elbow))
    axis = (wrist - elbow) / reach

    def ring(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = dress.boundary_edges(faces)
        along = (points - elbow) @ axis
        off = np.linalg.norm(points - elbow - np.outer(along, axis), axis=1)
        mine = (along[edges] > reach / 2).all(axis=1) & (off[edges] < 0.08).all(axis=1)
        return along[np.unique(edges[mine])] - reach, edges[mine]

    stored, _ = ring(FITTED.vertices, SKIN.faces)
    assert np.ptp(stored) > 0.015  # the ragged rim this replaces
    skin = LAYERS["skin"]
    cut, edges = ring(skin.vertices, skin.faces)
    assert len(cut) >= 16
    assert np.abs(cut + SETTINGS.wrist_cut_m).max() < 1e-9  # one plane
    start, count = np.unique(edges[:, 0], return_counts=True)
    assert (count == 1).all() and np.array_equal(start, np.unique(edges[:, 1]))
    walk, steps = int(edges[0, 0]), 0
    follow = dict(edges.tolist())
    while steps == 0 or walk != int(edges[0, 0]):
        walk, steps = follow[walk], steps + 1
    assert steps == len(edges)  # a single closed loop


def _hems(layer: dress.Layer) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each hem, highest first: its rim on the skin and its cloth edge, in loop order."""
    edges = dress.boundary_edges(layer.faces)
    fold = layer.faces[-2 * len(edges) : -len(edges)]  # (next, cloth, rim) per edge
    cloth = dict(zip(fold[:, 2].tolist(), fold[:, 1].tolist(), strict=True))
    follow, hems = dict(edges.tolist()), []
    while follow:
        walk, loop = next(iter(follow)), []
        while walk in follow:
            loop.append(walk)
            walk = follow.pop(walk)
        edge = [cloth[rim] for rim in loop]
        hems.append((layer.vertices[loop], layer.vertices[edge]))
    return sorted(hems, key=lambda hem: -float(hem[0][:, 2].mean()))


def _roughness_mm(loop: np.ndarray) -> float:
    """Largest distance of a closed loop's points from its low-harmonic fit."""
    step = np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1)
    turn = 2.0 * np.pi * np.cumsum(step) / step.sum()
    waves = range(1, LIMITS["hem_harmonics"] + 1)
    basis = [np.ones(len(loop))] + [
        f(k * turn) for k in waves for f in (np.cos, np.sin)
    ]
    design = np.column_stack(basis)
    smooth = design @ np.linalg.lstsq(design, loop, rcond=None)[0]
    return 1.0e3 * float(np.linalg.norm(loop - smooth, axis=1).max())


@pytest.mark.parametrize("profile", [None, "showpiece"])
def test_the_collar_is_a_smooth_line(profile: str | None) -> None:
    fitted = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings(profile=profile))
    shirt = dress.dress_skin(SKIN, fitted, SKIN.weights, SETTINGS)["shirt"]
    rim, edge = _hems(shirt)[0]
    assert rim[:, 2].min() > fitted.joints["shoulder_l"][2]  # the collar, not a sleeve
    assert _roughness_mm(rim) <= LIMITS["collar_roughness_mm"]
    assert _roughness_mm(edge) <= LIMITS["collar_roughness_mm"]


def test_each_sleeve_hem_sits_on_the_arm() -> None:
    hems = _hems(LAYERS["shirt"])
    assert len(hems) == 4  # collar, two sleeves, tail
    for rim, edge in hems[1:3]:
        assert abs(rim[:, 1].mean()) > FITTED.joints["shoulder_l"][1]  # on an arm
        stand = 1.0e3 * np.linalg.norm(edge - rim, axis=1)
        assert stand.max() <= LIMITS["sleeve_hem_stand_mm"]
        assert stand.min() >= 1.0e3 * SETTINGS.sleeve_hug_m - 0.5


def _hem_gap_mm(settings: dress.DressSettings) -> tuple[float, int]:
    """Largest distance from a rim vertex of the collar and the sleeve hems to
    the nearest vertex of the bare skin, and how many rim vertices were measured.

    Posed with the relaxed shoulder weights, the trail girdle protracted
    110 mm and the head turned ``neck_test_turn_rad``.
    """
    posing = pose.load_pose_settings()
    weights = pose.smooth_shoulder_weights(SKIN, FITTED, posing)
    layers = dress.dress_skin(SKIN, FITTED, weights, settings)
    girdle = _girdle(GIRDLE["trail_protracted_110_mm"])
    turn = Rotation.from_euler("z", RAW["neck_test_turn_rad"]).as_matrix()
    motions = {"thorax": pose.IDENTITY, "girdle_l": pose.IDENTITY}
    motions |= {"upper_arm_l": pose.IDENTITY, "girdle_r": girdle, "upper_arm_r": girdle}
    motions["head"] = pose.Rigid.about(turn, FITTED.joints["neck"])
    moves = pose.shoulder_transforms(motions, FITTED, posing)
    posed = {
        name: pose.pose_skin(
            layers[name].vertices, layers[name].weights, SKIN.groups, moves
        )
        for name in ("skin", "shirt")
    }
    rim = _rim(layers["shirt"])
    rim = rim[
        layers["shirt"].vertices[rim, 2] > SETTINGS.waist_height_m
    ]  # not the tail
    gap = cKDTree(posed["skin"]).query(posed["shirt"][rim])[0]
    return 1.0e3 * float(gap.max()), len(rim)


def test_each_hem_rim_stays_on_the_skins_own_vertices_at_a_turned_pose() -> None:
    """Blend skinning is not linear along an edge: a rim that only lies in the
    skin's faces at rest lifts off them when the girdle and the head turn, and
    a camera looks under the hem into the shirt. With the hem line made edges
    of the skin itself, rim and skin move as the same vertices."""
    gap, count = _hem_gap_mm(SETTINGS)
    assert count > 90  # collar and both sleeves
    assert gap <= LIMITS["hem_seam_mm"]
    apart, _ = _hem_gap_mm(dataclasses.replace(SETTINGS, hem_seam=False))
    assert apart > 1.0e3 * LIMITS["hem_seam_mm"]  # the check tells the two apart


def _front(layer: dress.Layer) -> np.ndarray:
    """Shirt surface samples (vertices and face centres) ahead of the hip joints."""
    points = np.vstack([layer.vertices, layer.vertices[layer.faces].mean(axis=1)])
    return points - 0.5 * (FITTED.joints["hip_l"] + FITTED.joints["hip_r"])


def _pectoral_relief_mm(layer: dress.Layer) -> float:
    """How far the shirt over a pectoral stands ahead of the shirt on the sternum."""
    front, relief = _front(layer), []
    for height in LIMITS["relief_heights_m"]:
        band = np.abs(front[:, 2] - height) < LIMITS["band_half_m"]
        side = np.abs(front[:, 1])
        sternum = front[band & (side < LIMITS["sternum_half_m"]), 0].max()
        low, high = LIMITS["pectoral_span_m"]
        relief.append(front[band & (side > low) & (side < high), 0].max() - sternum)
    return 1.0e3 * max(relief)


def _deepest_crease_mm(layer: dress.Layer, region: np.ndarray) -> float:
    """Largest depth of a vertex of ``region`` under the mean of its neighbours."""
    points, edges = layer.vertices, pose.mesh_edges(layer.faces)
    total, count = np.zeros_like(points), np.zeros(len(points))
    for near, far in ((0, 1), (1, 0)):
        np.add.at(total, edges[:, near], points[edges[:, far]])
        np.add.at(count, edges[:, near], 1.0)
    off = points - total / count[:, None]
    normals = dress.vertex_normals(points, layer.faces)
    return -1.0e3 * float(np.einsum("ij,ij->i", off, normals)[region].min())


def _regions(layer: dress.Layer) -> dict[str, np.ndarray]:
    points, joints = layer.vertices, FITTED.joints
    reach = [np.linalg.norm(points - joints[f"shoulder_{s}"], axis=1) for s in "lr"]
    low, high = LIMITS["relief_heights_m"][0], LIMITS["relief_heights_m"][-1]
    chest = (points[:, 0] - joints["hip_l"][0] > SETTINGS.chest.ahead_m) & (
        np.abs(points[:, 1]) < LIMITS["pectoral_span_m"][1]
    )
    chest &= (points[:, 2] > low) & (points[:, 2] < high)
    return {"chest": chest, "cap": np.min(reach, axis=0) < LIMITS["cap_radius_m"]}


def test_no_pectoral_relief_shows_through_the_shirt() -> None:
    assert _pectoral_relief_mm(LAYERS["shirt"]) <= LIMITS["pectoral_relief_mm"]
    assert _pectoral_relief_mm(PLAIN_LAYERS["shirt"]) >= LIMITS["plain_relief_mm"]


@pytest.mark.parametrize("region", ["chest", "cap"])
def test_the_shirt_is_taut_over_the_chest_and_the_sleeve_caps(region: str) -> None:
    taut, plain = LAYERS["shirt"], PLAIN_LAYERS["shirt"]
    assert _regions(taut)[region].sum() > 100
    assert _deepest_crease_mm(taut, _regions(taut)[region]) <= LIMITS["crease_mm"]
    creased = _deepest_crease_mm(plain, _regions(plain)[region])
    assert creased >= LIMITS["plain_crease_mm"]


@pytest.mark.parametrize("profile", [None, "showpiece"])
def test_the_standing_chest_of_the_shirt_stays_within_its_depth(
    profile: str | None,
) -> None:
    fitted = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings(profile=profile))
    for settings in (SETTINGS, PLAIN):
        shirt = dress.dress_skin(SKIN, fitted, SKIN.weights, settings)["shirt"]
        front = _front(shirt)
        limits = LIMITS["chest_depth_mm"][profile or "default"]
        for height, limit in zip(LIMITS["chest_depth_heights_m"], limits, strict=True):
            band = np.abs(front[:, 2] - height) < LIMITS["band_half_m"]
            middle = front[band & (np.abs(front[:, 1]) < 0.05), 0]
            assert len(middle) > 10
            depth = 1.0e3 * float(np.ptp(middle))
            # Evening the chest never deepens it; the bound is the dressed one's.
            assert depth <= limit + (0.0 if settings is SETTINGS else 12.0)


@pytest.mark.parametrize("side", ["l", "r"])
def test_each_trouser_leg_stands_off_the_shank(side: str) -> None:
    knee, ankle = FITTED.joints[f"knee_{side}"], FITTED.joints[f"ankle_{side}"]
    bone = ankle - knee

    def radii(layer: dress.Layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cover = np.setdiff1d(np.arange(len(layer.vertices)), _rim(layer))
        points = layer.vertices[cover]
        points = points[np.sign(points[:, 1]) == dict(fit.SIDES)[side]]
        share = (points - knee) @ bone / (bone @ bone)
        radius = np.linalg.norm(points - knee - np.outer(share, bone), axis=1)
        return share, radius, points

    share, radius, points = radii(LAYERS["trousers"])
    below = (share > 0.0) & (share < SETTINGS.trouser_end_fraction)
    assert below.sum() > 100
    want = np.interp(share, [0.0, 1.0], SETTINGS.leg_tube_m)
    assert (radius - want)[below].min() >= -1.0e-9
    _, plain_radius, plain_points = radii(TUBELESS)
    # Without the tube the cloth follows the calf and the ankle.
    assert plain_radius[below & (share > 0.2)].min() < SETTINGS.leg_tube_m[1] - 0.02
    above = share < -SETTINGS.leg_tube_rise
    assert above.sum() > 100
    assert np.array_equal(points[above], plain_points[above])


@pytest.mark.parametrize("profile", [None, "showpiece"])
def test_the_trousers_stand_off_the_tucked_shirt(profile: str | None) -> None:
    fitted = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings(profile=profile))
    layers = dress.dress_skin(SKIN, fitted, SKIN.weights, SETTINGS)
    trousers, shirt = layers["trousers"], layers["shirt"]
    cover = np.setdiff1d(np.arange(len(trousers.vertices)), _rim(trousers))
    points = trousers.vertices[cover]
    tail = fitted.joints["hip_l"][2] + SETTINGS.waist_height_m - SETTINGS.tuck_m
    band = points[points[:, 2] > tail + 0.01]  # over the shirt, clear of its hem
    assert len(band) > 60
    margin = 1.0e3 * mesh.surface_height(band, shirt.vertices, shirt.faces)
    assert margin.min() >= LIMITS["tucked_margin_mm"]
    assert margin.max() <= 1.0e3 * (SETTINGS.waist_gap_m + SETTINGS.trousers.ease_m)
    with pytest.raises(ValueError, match="points"):
        mesh.surface_height(band[:, :2], shirt.vertices, shirt.faces)


@pytest.mark.parametrize("side", ["l", "r"])
def test_each_trouser_leg_hangs_close_to_the_shank(side: str) -> None:
    knee, ankle = FITTED.joints[f"knee_{side}"], FITTED.joints[f"ankle_{side}"]
    bone = ankle - knee

    def radii(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = points[np.sign(points[:, 1]) == dict(fit.SIDES)[side]]
        share = (points - knee) @ bone / (bone @ bone)
        return share, np.linalg.norm(points - knee - np.outer(share, bone), axis=1)

    layer = LAYERS["trousers"]
    cover = np.setdiff1d(np.arange(len(layer.vertices)), _rim(layer))
    cloth, skin = radii(layer.vertices[cover]), radii(FITTED.vertices)
    spans = ((0.3, 0.7), (0.7, SETTINGS.trouser_end_fraction))
    for (low, high), limit in zip(spans, LIMITS["shank_slack_mm"], strict=True):
        mean = [r[(s > low) & (s < high)].mean() for s, r in (cloth, skin)]
        assert 0.0 < 1.0e3 * (mean[0] - mean[1]) <= limit


@pytest.mark.parametrize("profile", [None, "showpiece"])
def test_each_trouser_hem_hangs_below_the_bare_ankle(profile: str | None) -> None:
    """No bare skin shows under a trouser leg: its hem is the leg's level end."""
    fitted = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings(profile=profile))
    layers = dress.dress_skin(SKIN, fitted, SKIN.weights, SETTINGS)
    hems, bare = _hems(layers["trousers"]), layers["skin"].vertices
    assert len(hems) == 3  # waist, two legs
    for rim, edge in hems[1:]:
        side = "l" if edge[:, 1].mean() > 0.0 else "r"
        knee, ankle = fitted.joints[f"knee_{side}"], fitted.joints[f"ankle_{side}"]
        bone = ankle - knee
        length_mm = 1.0e3 * float(np.linalg.norm(bone))
        share = [(points - knee) @ bone / (bone @ bone) for points in (bare, rim, edge)]
        off = [
            np.linalg.norm(points - knee - np.outer(along, bone), axis=1)
            for points, along in zip((bare, rim, edge), share, strict=True)
        ]
        near = (share[0] > 0.5) & (off[0] < LIMITS["ankle_reach_m"])
        lowest = share[0][near].max(initial=-np.inf)
        assert length_mm * (lowest - share[1].min()) <= LIMITS["bare_ankle_mm"]
        # one level line across the shank, below the ankle joint, on the leg's tube
        assert np.allclose(share[1], SETTINGS.trouser_end_fraction)
        level = length_mm * np.abs(share[2] - SETTINGS.trouser_end_fraction)
        assert level.max() <= LIMITS["hem_level_mm"]
        assert off[2].min() >= SETTINGS.leg_tube_m[1] - 1.0e-9


@pytest.mark.parametrize(
    "change",
    [
        {"waist_gap_m": 0.0},
        {"skin_margin_m": 0.0},
        {"sleeve_fraction": 1.5},
        {"wrist_cut_m": 0.0},
        {"leg_tube_m": (0.07, -0.01)},
        {"leg_tube_rise": 0.0},
        {"sleeve_cap_passes": -1},
        {"sleeve_hug_m": 0.0},
        {"sleeve_hug_gain": -0.1},
        {"ankle_end_fraction": 1.0},
        {"hem_seam": 1},
    ],
)
def test_invalid_settings_and_weights_raise(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SETTINGS, **change)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dress.ClothSettings(0.005, 0.01, 0, 0)
    with pytest.raises(ValueError, match="edge_m"):
        dataclasses.replace(SETTINGS.chest, edge_m=0.0)
    with pytest.raises(ValueError, match="passes"):
        dataclasses.replace(SETTINGS.chest, tent_passes=-1)
    with pytest.raises(ValueError):
        dress.dress_skin(SKIN, FITTED, SKIN.weights[:, :3], SETTINGS)


def _neck_end(layer: dress.Layer) -> np.ndarray:
    """Open edges of a skin layer that lie wholly above the neck joint."""
    edges = dress.boundary_edges(layer.faces)
    above = layer.vertices[edges, 2] > FITTED.joints["neck"][2]
    return np.asarray(edges[above.all(axis=1)])


def test_the_skin_neck_is_drawn_in_and_closed_by_a_plug_of_its_own() -> None:
    """The neck's open end is closed by a plug that moves with that end."""
    cap = SETTINGS.neck_cap
    off = dataclasses.replace(cap, apex_height_m=0.0)
    plain = dataclasses.replace(SETTINGS, neck_cap=off)
    open_skin = dress.dress_skin(SKIN, FITTED, SKIN.weights, plain)["skin"]
    closed, count = LAYERS["skin"], len(open_skin.vertices)
    assert dress.close_neck(open_skin, FITTED.joints["neck"], off) is open_skin
    ring = np.unique(_neck_end(open_skin))
    assert len(ring) >= 40 and len(_neck_end(closed)) == 0
    assert len(closed.vertices) == count + len(ring) + 1
    apex = FITTED.joints["neck"] + (cap.apex_forward_m, 0.0, cap.apex_height_m)
    assert np.allclose(closed.vertices[-1], apex)
    # the plug carries the end ring's mean weights: one piece, moving with the end
    assert np.array_equal(closed.weights[:count], open_skin.weights)
    assert np.allclose(closed.weights[count:], open_skin.weights[ring].mean(axis=0))
    assert np.allclose(closed.weights.sum(axis=1), 1.0)
    # the rim is drawn tuck_m towards that point, the skin tuck_span_m off it not
    step = closed.vertices[:count] - open_skin.vertices
    moved = np.linalg.norm(step, axis=1)
    reach = cKDTree(open_skin.vertices[ring]).query(open_skin.vertices)[0]
    assert np.allclose(moved[ring], cap.tuck_m) and moved.max() <= cap.tuck_m + 1e-12
    assert not moved[reach >= cap.tuck_span_m].any()
    assert (((apex - open_skin.vertices[ring]) * step[ring]).sum(axis=1) > 0.0).all()
    assert not moved[np.setdiff1d(_rim(open_skin), ring)].any()  # collar, wrists
    # the plug's ring lies nearer the point than the rim, at least lip_m on average
    near = np.linalg.norm(closed.vertices[count:-1] - apex, axis=1)
    rim = np.linalg.norm(closed.vertices[ring] - apex, axis=1)
    assert near.max() < rim.max() and (rim - near).mean() >= cap.lip_m
    # three faces per open edge, none turned against the neck it closes
    plug = (closed.faces >= count).any(axis=1)
    assert plug.sum() == 3 * len(_neck_end(open_skin))
    assert np.array_equal(closed.faces[~plug], open_skin.faces)
    assert not mesh.folded_faces(closed.vertices, closed.faces)[plug].any()
    with pytest.raises(ValueError, match="no open end"):
        dress.close_neck(open_skin, FITTED.joints["neck"] + (0.0, 0.0, 1.0), cap)
    for change in ({"apex_height_m": -0.01}, {"tuck_span_m": 0.0}, {"lip_passes": -1}):
        with pytest.raises(ValueError, match="neck_cap"):
            dataclasses.replace(cap, **change)
