"""Clothing layers of the CC0 body skin on the default subject (#15299)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.shared.python.model_appearance import body_skin_dress as dress
from src.shared.python.model_appearance import body_skin_fit as fit
from tests.unit.model_appearance.test_body_skin_fit import _subject

pytestmark = pytest.mark.unit

SETTINGS = dress.load_dress_settings()
SKIN = fit.load_body_skin()
FITTED = fit.fit_body_skin(SKIN, _subject(), fit.load_fit_settings())
LAYERS = dress.dress_skin(SKIN, FITTED, SKIN.weights, SETTINGS)
#: Slack of the sampled-surface measure below (curvature between samples).
SAMPLE_SLACK_M = 0.0015


def _body_samples() -> tuple[cKDTree, np.ndarray, np.ndarray]:
    """Points on the body surface (vertices, edge midpoints, centroids) and normals."""
    points, faces = FITTED.vertices, SKIN.faces
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


@pytest.mark.parametrize(
    "change",
    [{"skin_margin_m": 0.0}, {"sleeve_fraction": 1.5}, {"wrist_cut_m": 0.0}],
)
def test_invalid_settings_and_weights_raise(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SETTINGS, **change)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dress.ClothSettings(0.005, 0.01, 0, 0)
    with pytest.raises(ValueError):
        dress.dress_skin(SKIN, FITTED, SKIN.weights[:, :3], SETTINGS)
