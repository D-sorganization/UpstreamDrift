"""Rest-shape fullness of the skinned hand (epic #15150)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.hand_asset import (
    load_hand_cc0_asset,
)
from src.shared.python.model_appearance.hand_backends.rest_shape import (
    FullnessSettings,
    filled_out,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)
from src.shared.python.model_appearance.hand_backends.skinning import vertex_normals

pytestmark = pytest.mark.unit

_PAD = FullnessSettings(
    along_bone="thumb_meta",
    bones=("wrist", "thumb_meta", "thumb_prox"),
    height_m=0.004,
    centre=0.0,
    half_span=1.1,
    radius_m=0.035,
    facing_full=0.7,
)


def _asset():
    return load_hand_cc0_asset().asset


def _back_of(asset, bone: str) -> np.ndarray:
    i = asset.bone_index(bone)
    along = asset.bone_tails[i] - asset.bone_heads[i]
    back = -np.cross(asset.flex_axes[i], along)
    return back / np.linalg.norm(back)


def test_the_default_hand_has_a_fuller_thumb_base_and_it_can_be_switched_off() -> None:
    pads = default_settings().fullness
    assert [pad.along_bone for pad in pads] == ["thumb_meta"]
    assert pads[0].height_m > 0.0
    raw = load_settings_mapping()
    raw["mesh"] = {**raw["mesh"], "fullness": []}
    assert settings_from_mapping(raw).fullness == ()
    asset = _asset()
    assert filled_out(asset, ()) is asset
    flat = dataclasses.replace(_PAD, height_m=0.0)
    np.testing.assert_array_equal(
        filled_out(asset, (flat,)).rest_vertices, asset.rest_vertices
    )


def test_fullness_adds_tissue_on_the_back_of_the_bone_only() -> None:
    asset = _asset()
    full = filled_out(asset, (_PAD,))
    shift = full.rest_vertices - asset.rest_vertices
    moved = np.linalg.norm(shift, axis=1)
    ids = np.flatnonzero(moved > 1e-12)
    assert len(ids) > 0
    # No more than the set height, and the peak comes close to it.
    assert 0.5 * _PAD.height_m < moved.max() <= _PAD.height_m + 1e-12
    # Only the named bones' skin moves, outward, on the back of the thumb.
    assert set(ids) <= set(asset.dominant_vertex_ids(_PAD.bones))
    normals = vertex_normals(asset.rest_vertices, asset.faces)
    assert (np.einsum("ij,ij->i", shift[ids], normals[ids]) > 0.0).all()
    assert (normals[ids] @ _back_of(asset, "thumb_meta") > 0.0).all()
    # Within the span along the bone and the radius about it.
    i = asset.bone_index("thumb_meta")
    head, tail = asset.bone_heads[i], asset.bone_tails[i]
    length = float(np.linalg.norm(tail - head))
    along = (tail - head) / length
    rel = asset.rest_vertices[ids] - head
    station = rel @ along / length
    assert (np.abs(station - _PAD.centre) < _PAD.half_span).all()
    off_axis = np.linalg.norm(rel - np.outer(rel @ along, along), axis=1)
    assert (off_axis < _PAD.radius_m).all()
    # The rest of the asset is untouched and the input is not modified.
    assert full.faces is asset.faces
    assert full.joint_weights is asset.joint_weights


def test_twice_the_height_moves_the_same_skin_twice_as_far() -> None:
    asset = _asset()
    once = filled_out(asset, (_PAD,)).rest_vertices - asset.rest_vertices
    double = dataclasses.replace(_PAD, height_m=2.0 * _PAD.height_m)
    twice = filled_out(asset, (double,)).rest_vertices - asset.rest_vertices
    np.testing.assert_allclose(twice, 2.0 * once, atol=1e-15)


@pytest.mark.parametrize(
    "bad",
    [
        {"height_m": -0.001},
        {"height_m": float("nan")},
        {"half_span": 0.0},
        {"radius_m": 0.0},
        {"facing_full": 0.0},
        {"facing_full": 1.5},
        {"bones": ()},
    ],
)
def test_bad_fullness_settings_raise(bad: dict) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_PAD, **bad)


def test_fullness_names_must_be_bones_that_flex() -> None:
    asset = _asset()
    with pytest.raises(ValueError, match="unknown bone"):
        filled_out(asset, (dataclasses.replace(_PAD, along_bone="no_such_bone"),))
    with pytest.raises(ValueError, match="names no bone"):
        filled_out(asset, (dataclasses.replace(_PAD, bones=("no_such_bone",)),))
    with pytest.raises(ValueError, match="does not flex"):
        filled_out(asset, (dataclasses.replace(_PAD, along_bone="forearm"),))
