"""The fast posing kernels give exactly what the plain ones give (#15150).

Posing a pair of hands evaluates the skin tens of thousands of times. The
kernels that make that affordable keep work between evaluations (bone
rotations, forward kinematics, skin rows, a grid that tells far points from
near ones, the mesh's edge pairs, a whole posed pair). Each is held here to
the plain computation, bit for bit, and what is kept is shown not to leak.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.shared.python.model_appearance.hand_backends import correctives as co
from src.shared.python.model_appearance.hand_backends import skinned_human as sh
from src.shared.python.model_appearance.hand_backends import skinned_pair
from src.shared.python.model_appearance.hand_backends.palm_fold import _refit
from src.shared.python.model_appearance.hand_backends.posing import skin_distance
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from src.shared.python.model_appearance.hand_backends.skinning import (
    rotation_about_axis,
    vertex_normals,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import (
    _RADIUS,
    _pair,
    _spec,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("skinned_hand_backends")]

_SETTINGS = default_settings()
_ASSET = sh._sized_assets(0.19, 0.085, _SETTINGS)[0]


def _random_pose(rng: np.random.Generator) -> dict[str, float]:
    bones = [f"{f}_{s}" for f in sh.FINGERS for s in sh._SEGMENTS]
    bones += ["thumb_prox", "thumb_dist", sh.THUMB_PRESS, "meta_ring"]
    return {b: float(rng.uniform(-0.2, 1.4)) for b in bones if rng.random() < 0.7}


def test_kept_kinematics_equal_a_hand_that_keeps_nothing() -> None:
    """One hand is posed again and again while its fixed rotations, offsets,
    sideways swings and thumb press axis change; every pose must equal that of
    a new hand in the same state, which has nothing kept from earlier poses."""
    rng = np.random.default_rng(15150)
    hand = sh._GripHand(_ASSET, _SETTINGS)
    ids = hand.moved_ids("index_prox")
    pose = _random_pose(rng)
    for step in range(60):
        change = step % 6
        if change == 0:
            pose = _random_pose(rng)
        elif change == 1:  # one joint only, as a curl search does
            pose = {**pose, "ring_mid": float(rng.uniform(0.0, 1.5))}
        elif change == 2:
            hand.swing_sideways("ring_prox", float(rng.uniform(-0.2, 0.2)))
        elif change == 3:
            turn = rotation_about_axis(rng.normal(size=3), float(rng.normal()))
            hand.fixed["thumb_meta"] = turn
            hand.press_axis = turn[0]
        elif change == 4:
            shift = np.eye(4)
            shift[:3, 3] = rng.normal(size=3) * 0.01
            hand.offsets["thumb_meta"] = shift
        else:
            hand.placement = shift @ hand.placement
        plain = _refit(hand, _ASSET)
        np.testing.assert_array_equal(hand.transforms(pose), plain.transforms(pose))
        np.testing.assert_array_equal(hand.points(pose), plain.points(pose))
        np.testing.assert_array_equal(hand.points(pose, ids), plain.points(pose, ids))
        tip = hand.joint_position("thumb_dist", pose, tail=True)
        np.testing.assert_array_equal(
            tip, plain.joint_position("thumb_dist", pose, tail=True)
        )


def test_the_transforms_handed_out_are_the_callers_own() -> None:
    hand = sh._GripHand(_ASSET, _SETTINGS)
    pose = {"index_prox": 0.6, "index_mid": 0.9}
    first = hand.transforms(pose)
    kept = first.copy()
    first[:] = 0.0
    np.testing.assert_array_equal(hand.transforms(pose), kept)
    np.testing.assert_array_equal(
        hand.points(pose), sh._GripHand(_ASSET, _SETTINGS).points(pose)
    )


def _plain_skin_distance(vertices, normals, reach_m):
    """``posing.skin_distance`` as first written: every point asks the tree."""
    tree = cKDTree(vertices)
    spacing = float(np.median(np.asarray(tree.query(vertices, k=2)[0])[:, 1]))

    def distance(points):
        reach, nearest = tree.query(points, distance_upper_bound=reach_m)
        near = np.isfinite(reach)
        nearest = nearest[near]
        height = np.einsum(
            "ij,ij->i", points[near] - vertices[nearest], normals[nearest]
        )
        size = np.maximum(np.abs(height), reach[near] - spacing)
        out = np.full(len(points), reach_m)
        out[near] = np.where(height < 0.0, -size, size)
        return out

    return distance


@pytest.mark.parametrize("reach_m", [0.004, 0.015, 0.2, math.inf])
def test_skin_distance_with_its_far_point_grid_equals_the_plain_search(
    reach_m: float,
) -> None:
    rng = np.random.default_rng(7)
    skin = _ASSET.rest_vertices
    normals = vertex_normals(skin, _ASSET.faces)
    low, high = skin.min(axis=0), skin.max(axis=0)
    span = high - low
    points = np.concatenate(
        [
            rng.uniform(low - 0.6 * span, high + 0.6 * span, size=(4000, 3)),
            skin[::3] + rng.normal(scale=0.01, size=skin[::3].shape),
            skin[::7] + normals[::7] * min(reach_m, 1.0),  # at the reach itself
            skin[:5],
        ]
    )
    fast = skin_distance(skin, normals, reach_m)(points)
    np.testing.assert_array_equal(
        fast, _plain_skin_distance(skin, normals, reach_m)(points)
    )
    assert skin_distance(skin, normals, reach_m)(points[:0]).shape == (0,)


def test_vertex_neighbours_are_the_sorted_edge_neighbours() -> None:
    faces = _ASSET.faces
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    both = np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0)
    found = co.vertex_neighbours(faces, len(_ASSET.rest_vertices))
    assert len(found) == len(_ASSET.rest_vertices)
    for vertex, neighbours in enumerate(found):
        np.testing.assert_array_equal(neighbours, both[both[:, 0] == vertex, 1])


def test_sharpening_kept_for_a_rest_mesh_equals_sharpened_faces() -> None:
    rng = np.random.default_rng(11)
    rest, faces = _ASSET.rest_vertices, _ASSET.faces
    sharpened = co.sharpening(rest, faces, 20.0)
    for scale in (0.0, 0.0005, 0.003):
        posed = rest + rng.normal(scale=scale, size=rest.shape)
        now, pair = co.edge_folds(posed, faces)
        before, _ = co.edge_folds(rest, faces)
        plain = np.unique(pair[now - before > 20.0]).astype(np.int64)
        np.testing.assert_array_equal(sharpened(posed), plain)
        np.testing.assert_array_equal(
            co.sharpened_faces(posed, rest, faces, 20.0), plain
        )
    with pytest.raises(ValueError, match="max_added_fold_deg"):
        co.sharpening(rest, faces, 0.0)


def test_a_posed_pair_is_kept_and_handed_out_as_copies() -> None:
    """The same grip, radius and settings are posed once in a process. What
    is handed out is a copy: changing it changes neither the next one handed
    out nor one handed out before; a rolled grip reuses the level solve."""
    first = _pair()
    before = skinned_pair._solved_pair.cache_info()
    again = skinned_pair.build_grip_hands(_spec(), grip_radius_m=_RADIUS)
    rolled = skinned_pair.build_grip_hands(_spec(20.0), grip_radius_m=_RADIUS)
    after = skinned_pair._solved_pair.cache_info()
    assert (after.hits, after.misses) == (before.hits + 2, before.misses)
    assert after.maxsize == skinned_pair._PAIR_MEMO_SIZE
    for side in ("lead", "trail"):
        one, two = first[side], again[side]
        assert one.joint_angles == two.joint_angles
        assert one.contact_gaps_m == two.contact_gaps_m
        assert one.contact_gaps_m is not two.contact_gaps_m
        for name in ("vertex_normals", "bone_transforms", "frame_from_rest"):
            np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
            assert not np.shares_memory(getattr(one, name), getattr(two, name))
        np.testing.assert_array_equal(one.mesh.vertices, two.mesh.vertices)
        assert not np.shares_memory(one.mesh.vertices, two.mesh.vertices)
        assert not np.shares_memory(one.mesh.faces, two.mesh.faces)
        assert not np.shares_memory(rolled[side].mesh.faces, one.mesh.faces)
        kept = one.mesh.vertices.copy()
        two.mesh.vertices[:] = 0.0
        two.contact_gaps_m.clear()
        two.bone_transforms[:] = 0.0
        third = skinned_pair.build_grip_hands(_spec(), grip_radius_m=_RADIUS)[side]
        np.testing.assert_array_equal(third.mesh.vertices, kept)
        np.testing.assert_array_equal(one.mesh.vertices, kept)
        assert third.contact_gaps_m == one.contact_gaps_m
        np.testing.assert_array_equal(third.bone_transforms, one.bone_transforms)


def test_neighbour_means_taken_together_equal_those_taken_one_by_one() -> None:
    rng = np.random.default_rng(3)
    rest = _ASSET.rest_vertices
    neighbours = co.vertex_neighbours(_ASSET.faces, len(rest))
    assert max(len(n) for n in neighbours) >= 8  # where a run is summed otherwise
    ids = np.unique(rng.integers(0, len(rest), size=900))
    for values in (rest + rng.normal(scale=0.01, size=rest.shape), rest[:, 0]):
        plain = np.array([values[neighbours[i]].mean(axis=0) for i in ids])
        got = co._neighbour_mean(values, neighbours, ids)
        assert got.tobytes() == plain.tobytes()
