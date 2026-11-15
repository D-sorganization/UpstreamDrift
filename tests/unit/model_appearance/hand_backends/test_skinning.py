"""MakeHuman CC0 rig asset and hierarchical linear blend skinning (#15148)."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.hand_asset import (
    ASSET_DIR,
    load_hand_cc0_asset,
)
from src.shared.python.model_appearance.hand_backends.skinning import (
    bone_transforms,
    degenerate_face_count,
    flipped_face_count,
    pose_vertices,
    rotation_about_axis,
    scale_asset,
    vertex_normals,
)

pytestmark = pytest.mark.unit

_FINGERS = ("index", "middle", "ring", "little")


def _asset():
    return load_hand_cc0_asset().asset


def test_asset_is_the_makehuman_rig_with_cc0_provenance() -> None:
    record = json.loads((ASSET_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert record["licence"] == "CC0-1.0"
    assert "MANO" in record["research_only_alternatives"]
    urls = " ".join(record["source_urls"])
    for name in ("base.obj", "default.mhskel", "default_weights.mhw"):
        assert name in urls
    notice = (ASSET_DIR / "NOTICE").read_text(encoding="utf-8")
    for name in ("default.mhskel", "default_weights.mhw", "CC0", "MANO", "SMPL-X"):
        assert name in notice
    assert record["source_commit"] in notice
    # The one change to MakeHuman's weights is declared in both records.
    assert record["thumb_web_smoothing"]["iterations"] >= 0
    assert "thumb_web_smoothing" in notice


def test_asset_has_a_dense_mesh_and_a_full_hand_skeleton() -> None:
    asset = _asset()
    assert len(asset.rest_vertices) >= 1500
    assert len(asset.faces) >= 1500
    for finger in _FINGERS:
        for segment in ("prox", "mid", "dist"):
            assert f"{finger}_{segment}" in asset.bone_names
        assert f"meta_{finger}" in asset.bone_names
    for bone in ("thumb_meta", "thumb_prox", "thumb_dist", "wrist", "forearm"):
        assert bone in asset.bone_names
    np.testing.assert_allclose(asset.joint_weights.sum(axis=1), 1.0, atol=1e-5)
    assert asset.joint_indices.max() < len(asset.bone_names)


def test_parents_come_before_children_and_chains_are_connected() -> None:
    asset = _asset()
    for i, parent in enumerate(asset.bone_parents):
        assert parent < i
    for finger in _FINGERS:
        mid = asset.bone_index(f"{finger}_mid")
        prox = asset.bone_index(f"{finger}_prox")
        assert asset.bone_parents[mid] == prox
        assert asset.bone_parents[prox] == asset.bone_index(f"meta_{finger}")
        np.testing.assert_allclose(
            asset.bone_heads[mid], asset.bone_tails[prox], atol=1e-6
        )


def test_rest_mesh_is_closed_outward_wound_and_a_right_hand() -> None:
    asset = _asset()
    tri = asset.rest_vertices[asset.faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
    assert volume / 6.0 > 0.0
    f = asset.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all(), "mesh has open or non-manifold edges"
    # Canonical frame: x radial, y distal, z palmar; the thumb is radial.
    thumb = asset.bone_tails[asset.bone_index("thumb_dist")]
    little = asset.bone_tails[asset.bone_index("little_dist")]
    assert thumb[0] > 0.0 > little[0]
    assert asset.bone_tails[asset.bone_index("middle_dist")][1] > 0.15


def test_identity_pose_reproduces_the_rest_mesh() -> None:
    asset = _asset()
    posed = pose_vertices(asset, bone_transforms(asset, {}))
    np.testing.assert_allclose(posed, asset.rest_vertices, atol=1e-9)


def test_rotation_is_hierarchical() -> None:
    asset = _asset()
    prox = asset.bone_index("index_prox")
    rot = rotation_about_axis(asset.flex_axes[prox], math.radians(40.0))
    transforms = bone_transforms(asset, {"index_prox": rot})
    tip_rest = asset.bone_tails[asset.bone_index("index_dist")]
    head = asset.bone_heads[prox]
    expected = head + rot @ (tip_rest - head)
    dist = asset.bone_index("index_dist")
    moved = transforms[dist][:3, :3] @ tip_rest + transforms[dist][:3, 3]
    np.testing.assert_allclose(moved, expected, atol=1e-9)
    # Flexion moves the fingertip out of the palm (+z) and leaves the middle alone.
    assert moved[2] > tip_rest[2]
    np.testing.assert_allclose(
        transforms[asset.bone_index("middle_dist")], np.eye(4), atol=1e-12
    )


def test_pose_vertices_needs_one_transform_per_bone() -> None:
    asset = _asset()
    with pytest.raises(ValueError):
        pose_vertices(asset, np.eye(4)[None])
    with pytest.raises(ValueError):
        bone_transforms(asset, {"no_such_bone": np.eye(3)})


def test_moderately_curled_fingers_keep_every_triangle_valid() -> None:
    asset = _asset()
    rotations = {}
    for finger in _FINGERS:
        for segment, deg in (("prox", 30.0), ("mid", 45.0), ("dist", 30.0)):
            bone = f"{finger}_{segment}"
            axis = asset.flex_axes[asset.bone_index(bone)]
            rotations[bone] = rotation_about_axis(axis, math.radians(deg))
    transforms = bone_transforms(asset, rotations)
    posed = pose_vertices(asset, transforms)
    assert degenerate_face_count(posed, asset.faces, min_area_m2=1e-12) == 0
    assert flipped_face_count(asset, transforms, posed) == 0


def test_vertex_normals_are_unit_and_point_outward() -> None:
    asset = _asset()
    normals = vertex_normals(asset.rest_vertices, asset.faces)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)

    def volume(vertices: np.ndarray) -> float:
        tri = vertices[asset.faces]
        return float(
            np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
        )

    inflated = asset.rest_vertices + 1e-4 * normals
    assert volume(inflated) > volume(asset.rest_vertices) > 0.0


def test_scale_asset_scales_mesh_and_skeleton_together() -> None:
    asset = _asset()
    scaled = scale_asset(asset, (1.1, 1.05, 1.0))
    np.testing.assert_allclose(
        scaled.rest_vertices[:, 0], asset.rest_vertices[:, 0] * 1.1
    )
    np.testing.assert_allclose(scaled.bone_heads[:, 1], asset.bone_heads[:, 1] * 1.05)
    moving = np.linalg.norm(asset.flex_axes, axis=1) > 0.0
    np.testing.assert_allclose(
        np.linalg.norm(scaled.flex_axes[moving], axis=1), 1.0, atol=1e-9
    )
    with pytest.raises(ValueError):
        scale_asset(asset, (0.0, 1.0, 1.0))


def test_asset_matches_makehuman_source_when_a_checkout_is_available() -> None:
    """Skipped unless ``assets/hand_cc0/download/makehuman`` holds the pinned tree."""
    root = ASSET_DIR / "download" / "makehuman"
    if not (root / "makehuman" / "data" / "rigs" / "default.mhskel").is_file():
        pytest.skip("no MakeHuman checkout under assets/hand_cc0/download/makehuman")
    done = subprocess.run(
        [sys.executable, str(ASSET_DIR / "build_hand_asset.py"), str(root), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
