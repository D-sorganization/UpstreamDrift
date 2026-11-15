"""Loop subdivision of a skinned mesh (epic #15150): synthetic meshes only."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    bone_transforms,
    pose_vertices,
    rotation_about_axis,
)
from src.shared.python.model_appearance.hand_backends.subdivision import (
    creased_loop_subdivision,
    loop_subdivision,
    original_resolution,
    sharp_edges,
    subdivide_asset,
)

pytestmark = pytest.mark.unit


def _octahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    return float(
        np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    )


def _edge_use(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    return np.unique(edges, axis=0, return_counts=True)[1]


def _asset() -> SkinnedHandAsset:
    """An octahedron on two bones: the lower half rigid, the upper half hinged."""
    vertices, faces = _octahedron()
    upper = np.clip(vertices[:, 2] + 0.5, 0.0, 1.0)
    return SkinnedHandAsset(
        rest_vertices=vertices,
        faces=faces,
        joint_indices=np.tile(np.array([0, 1], dtype=np.int64), (len(vertices), 1)),
        joint_weights=np.column_stack([1.0 - upper, upper]),
        bone_names=("root", "tip"),
        bone_parents=np.array([-1, 0], dtype=np.int64),
        bone_heads=np.array([[0.0, 0, -1], [0, 0, 0]]),
        bone_tails=np.array([[0.0, 0, 0], [0, 0, 1]]),
        flex_axes=np.array([[0.0, 0, 0], [1, 0, 0]]),
        rest_flexion_rad=np.zeros(2),
    )


def test_one_level_splits_every_triangle_in_four_and_keeps_the_mesh_closed() -> None:
    vertices, faces = _octahedron()
    operator, fine = loop_subdivision(faces, len(vertices))
    assert operator.shape == (len(vertices) + 12, len(vertices))
    assert len(fine) == 4 * len(faces)
    assert (_edge_use(fine) == 2).all()
    np.testing.assert_allclose(np.asarray(operator.sum(axis=1)).ravel(), 1.0)
    assert _volume(operator @ vertices, fine) > 0.0


def test_subdivision_rounds_the_mesh_toward_its_smooth_limit() -> None:
    vertices, faces = _octahedron()
    for _ in range(2):
        operator, faces = loop_subdivision(faces, len(vertices))
        vertices = operator @ vertices
    radius = np.linalg.norm(vertices, axis=1)
    # The octahedron's corners stand 73 % farther out than its face centres.
    assert radius.max() / radius.min() < 1.25


def test_subdividing_an_asset_keeps_the_skeleton_and_a_valid_skin() -> None:
    asset = _asset()
    fine = subdivide_asset(asset, 1)
    assert len(fine.faces) == 4 * len(asset.faces)
    assert len(fine.rest_vertices) == len(asset.rest_vertices) + 12
    assert fine.joint_indices.shape == (len(fine.rest_vertices), 2)
    np.testing.assert_allclose(fine.joint_weights.sum(axis=1), 1.0, atol=1e-12)
    assert (fine.joint_weights >= 0.0).all()
    for name in ("bone_names", "bone_parents", "bone_heads", "bone_tails"):
        np.testing.assert_array_equal(getattr(fine, name), getattr(asset, name))
    # The first vertices are the original ones, in order: ids stay valid.
    original = fine.rest_vertices[: len(asset.rest_vertices)]
    assert np.linalg.norm(original - asset.rest_vertices, axis=1).max() < 0.5
    np.testing.assert_array_equal(
        np.sign(original).astype(int), np.sign(asset.rest_vertices).astype(int)
    )


def test_a_subdivided_asset_still_poses_by_linear_blend_skinning() -> None:
    fine = subdivide_asset(_asset(), 2)
    rest = pose_vertices(fine, bone_transforms(fine, {}))
    np.testing.assert_allclose(rest, fine.rest_vertices, atol=1e-12)
    bent = bone_transforms(
        fine, {"tip": rotation_about_axis(fine.flex_axes[1], np.radians(40.0))}
    )
    posed = pose_vertices(fine, bent)
    # Independent blend from the dense weights: same skin, bone by bone.
    homogeneous = np.column_stack([fine.rest_vertices, np.ones(len(posed))])
    expected = np.einsum(
        "vb,bij,vj->vi", fine.dense_weights(), bent[:, :3, :], homogeneous
    )
    np.testing.assert_allclose(posed, expected, atol=1e-12)
    assert np.abs(posed - fine.rest_vertices).max() > 0.1


def test_zero_levels_return_the_asset_itself_and_bad_levels_raise() -> None:
    asset = _asset()
    assert subdivide_asset(asset, 0) is asset
    for bad in (-1, 1.5, True):
        with pytest.raises((ValueError, TypeError)):
            subdivide_asset(asset, bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        loop_subdivision(asset.faces[:-1], len(asset.rest_vertices))


def test_original_resolution_is_the_fine_skin_at_the_source_vertices() -> None:
    source = _asset()
    fine = subdivide_asset(source, 2)
    coarse = original_resolution(fine, source)
    count = len(source.rest_vertices)
    np.testing.assert_array_equal(coarse.faces, source.faces)
    np.testing.assert_array_equal(coarse.rest_vertices, fine.rest_vertices[:count])
    np.testing.assert_array_equal(coarse.joint_weights, fine.joint_weights[:count])
    assert original_resolution(source, source) is source
    with pytest.raises(ValueError):
        original_resolution(source, fine)


_EQUATOR = np.array([[0, 2], [2, 1], [1, 3], [3, 0]], dtype=np.int64)


def test_a_creased_edge_stays_sharp_and_the_rest_stays_smooth() -> None:
    vertices, faces = _octahedron()
    smooth, _ = loop_subdivision(faces, len(vertices))
    operator, fine, fine_creases = creased_loop_subdivision(
        faces, len(vertices), _EQUATOR
    )
    np.testing.assert_allclose(np.asarray(operator.sum(axis=1)).ravel(), 1.0)
    out, plain = operator @ vertices, smooth @ vertices
    # A crease vertex moves only along its crease: 3/4 itself, 1/8 each side.
    np.testing.assert_allclose(
        out[0], 0.75 * vertices[0] + 0.125 * (vertices[2] + vertices[3])
    )
    assert np.linalg.norm(out[0]) > np.linalg.norm(plain[0])
    # The new vertex on a crease edge is the plain midpoint of that edge.
    mids = {tuple(np.round(row, 12)) for row in out[len(vertices) :]}
    assert tuple(np.round(0.5 * (vertices[0] + vertices[2]), 12)) in mids
    # Vertices off the crease follow the smooth rule unchanged.
    np.testing.assert_allclose(out[4:6], plain[4:6])
    # Each crease edge is split in two, ready for the next level.
    assert fine_creases.shape == (2 * len(_EQUATOR), 2)
    used = {
        tuple(sorted(edge))
        for face in fine
        for edge in zip(face, np.roll(face, -1), strict=True)
    }
    assert all(tuple(sorted(edge)) in used for edge in fine_creases.tolist())
    assert len(fine) == 4 * len(faces)


def test_no_creases_is_plain_loop_subdivision_and_a_stray_edge_raises() -> None:
    vertices, faces = _octahedron()
    smooth, smooth_faces = loop_subdivision(faces, len(vertices))
    operator, fine, creases = creased_loop_subdivision(
        faces, len(vertices), np.empty((0, 2), dtype=np.int64)
    )
    np.testing.assert_array_equal(fine, smooth_faces)
    np.testing.assert_allclose(operator.toarray(), smooth.toarray())
    assert creases.shape == (0, 2)
    with pytest.raises(ValueError):  # 0 and 1 are opposite corners: no such edge
        creased_loop_subdivision(faces, len(vertices), np.array([[0, 1]]))


def test_sharp_edges_are_found_by_dihedral_angle_within_a_vertex_mask() -> None:
    vertices, faces = _octahedron()
    everything = np.ones(len(vertices), dtype=bool)
    # Every octahedron edge folds by about 70.5 degrees.
    assert len(sharp_edges(vertices, faces, 60.0, everything)) == 12
    assert len(sharp_edges(vertices, faces, 80.0, everything)) == 0
    upper = vertices[:, 2] >= 0.0
    found = sharp_edges(vertices, faces, 60.0, upper)
    assert len(found) == 8 and upper[found].all()
    with pytest.raises(ValueError):
        sharp_edges(vertices, faces, 60.0, everything[:-1])


def test_an_asset_subdivided_with_creases_keeps_them_sharp_at_every_level() -> None:
    asset = _asset()
    plain = subdivide_asset(asset, 2)
    creased = subdivide_asset(asset, 2, _EQUATOR)
    assert creased.faces.shape == plain.faces.shape
    np.testing.assert_allclose(creased.joint_weights.sum(axis=1), 1.0, atol=1e-12)
    equator = np.abs(asset.rest_vertices[:, 2]) < 1e-12
    count = len(asset.rest_vertices)
    reach = np.linalg.norm(creased.rest_vertices[:count][equator], axis=1)
    smooth = np.linalg.norm(plain.rest_vertices[:count][equator], axis=1)
    assert (reach > smooth + 0.05).all()
