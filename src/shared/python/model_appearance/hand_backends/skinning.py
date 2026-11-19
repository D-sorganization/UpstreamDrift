"""Hierarchical linear blend skinning for hand assets (epic #15150).

A hand is posed by **joint rotation only**. Each bone carries a rotation about
its own head (the joint it hangs from), expressed in the rest frame; forward
kinematics chains them down the hierarchy and linear blend skinning applies
the weighted bone transforms to the rest vertices. Nothing in this module
moves a vertex in any other way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SkinnedHandAsset:
    """Rest mesh, skeleton and skin weights of one (right) hand, in metres.

    Rest frame: origin at the wrist joint, ``x`` radial, ``y`` distal, ``z``
    palmar. ``bone_parents`` lists parents before children (``-1`` = root).
    ``flex_axes`` are unit hinge axes (zero for bones that do not flex);
    a positive rotation about one curls the digit toward the palm.
    ``rest_flexion_rad`` is how far each joint is already flexed at rest.
    """

    rest_vertices: Array  # (V, 3)
    faces: IntArray  # (F, 3) outward wound
    joint_indices: IntArray  # (V, K)
    joint_weights: Array  # (V, K), rows sum to 1
    bone_names: tuple[str, ...]
    bone_parents: IntArray  # (B,)
    bone_heads: Array  # (B, 3)
    bone_tails: Array  # (B, 3)
    flex_axes: Array  # (B, 3)
    rest_flexion_rad: Array  # (B,)

    def bone_index(self, name: str) -> int:
        try:
            return self.bone_names.index(name)
        except ValueError:
            raise ValueError(f"unknown bone {name!r}") from None

    def descendants(self, name: str) -> tuple[int, ...]:
        """Indices of ``name`` and every bone below it."""
        root = self.bone_index(name)
        keep = [root]
        for i in range(root + 1, len(self.bone_names)):
            if int(self.bone_parents[i]) in keep:
                keep.append(i)
        return tuple(keep)

    def dense_weights(self) -> Array:
        """``(V, B)`` weight matrix."""
        dense = np.zeros((len(self.rest_vertices), len(self.bone_names)))
        rows = np.arange(len(self.rest_vertices))[:, None]
        np.add.at(dense, (rows, self.joint_indices), self.joint_weights)
        return dense

    def dominant_vertex_ids(self, bones: Sequence[str]) -> IntArray:
        """Vertices whose strongest influence is one of ``bones``."""
        ids = [self.bone_index(b) for b in bones]
        return np.flatnonzero(np.isin(self.dense_weights().argmax(axis=1), ids))

    def influenced_vertex_ids(self, bones: Sequence[str]) -> IntArray:
        """Vertices with any weight on one of ``bones``."""
        ids = [self.bone_index(b) for b in bones]
        return np.flatnonzero(self.dense_weights()[:, ids].sum(axis=1) > 0.0)


def rotation_about_axis(axis: Array, angle_rad: float) -> Array:
    """Rodrigues rotation matrix. Precondition: finite, non-zero axis."""
    a = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(a))
    if a.shape != (3,) or not np.isfinite(a).all() or norm < 1e-12:
        raise ValueError("rotation axis must be a finite non-zero 3-vector")
    if not np.isfinite(angle_rad):
        raise ValueError("rotation angle must be finite")
    x, y, z = a / norm
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


def bone_transforms(asset: SkinnedHandAsset, rotations: Mapping[str, Array]) -> Array:
    """Forward kinematics: ``(B, 4, 4)`` rest-to-posed transform of every bone.

    ``rotations[name]`` is a proper 3x3 rotation of that bone about its head,
    in rest-frame axes; missing bones do not rotate. Raises ``ValueError`` for
    an unknown bone or a matrix that is not a rotation.
    """
    unknown = set(rotations) - set(asset.bone_names)
    if unknown:
        raise ValueError(f"unknown bones in pose: {sorted(unknown)}")
    out = np.tile(np.eye(4), (len(asset.bone_names), 1, 1))
    for i, name in enumerate(asset.bone_names):
        local = np.eye(4)
        if name in rotations:
            rot = np.asarray(rotations[name], dtype=float)
            if rot.shape != (3, 3) or not np.allclose(
                rot @ rot.T, np.eye(3), atol=1e-9
            ):
                raise ValueError(f"pose of {name!r} is not a rotation matrix")
            head = asset.bone_heads[i]
            local[:3, :3] = rot
            local[:3, 3] = head - rot @ head
        parent = int(asset.bone_parents[i])
        out[i] = local if parent < 0 else out[parent] @ local
    return out


def pose_vertices(asset: SkinnedHandAsset, transforms: Array) -> Array:
    """Linear blend skinning of the rest vertices by ``transforms``."""
    t = np.asarray(transforms, dtype=float)
    if t.shape != (len(asset.bone_names), 4, 4) or not np.isfinite(t).all():
        raise ValueError("need one finite 4x4 transform per bone")
    per_vertex = np.einsum("vk,vkij->vij", asset.joint_weights, t[asset.joint_indices])
    rest = asset.rest_vertices
    return np.einsum("vij,vj->vi", per_vertex[:, :3, :3], rest) + per_vertex[:, :3, 3]


def scale_asset(
    asset: SkinnedHandAsset, scale_xyz: Sequence[float]
) -> SkinnedHandAsset:
    """Scale mesh and skeleton together along the rest axes (anthropometry)."""
    s = np.asarray(scale_xyz, dtype=float)
    if s.shape != (3,) or not np.isfinite(s).all() or (s <= 0.0).any():
        raise ValueError("scale must be three positive finite factors")
    axes = asset.flex_axes / s  # hinge axes transform as plane normals
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = np.divide(axes, norms, out=np.zeros_like(axes), where=norms > 1e-12)
    return replace(
        asset,
        rest_vertices=asset.rest_vertices * s,
        bone_heads=asset.bone_heads * s,
        bone_tails=asset.bone_tails * s,
        flex_axes=axes,
    )


def face_normals(vertices: Array, faces: IntArray) -> Array:
    """Area-weighted (unnormalised) face normals."""
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.asarray(normals, dtype=np.float64)


def vertex_normals(vertices: Array, faces: IntArray) -> Array:
    """Unit area-weighted vertex normals, for smooth shading."""
    normals = np.zeros_like(vertices)
    fn = face_normals(vertices, faces)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-30)


def degenerate_face_count(
    vertices: Array, faces: IntArray, *, min_area_m2: float
) -> int:
    """Triangles whose area is below ``min_area_m2``."""
    areas = 0.5 * np.linalg.norm(face_normals(vertices, faces), axis=1)
    return int((areas < min_area_m2).sum())


def flipped_face_ids(
    asset: SkinnedHandAsset, transforms: Array, posed: Array
) -> NDArray[np.int64]:
    """Triangles that skinning turned inside out.

    A posed triangle is flipped when its normal points against the rest
    normal carried along by the blended rotation of its own corners. Under
    linear blend skinning this happens only where the skin of two segments
    folds into the inner crease of a strongly flexed joint.
    """
    per_vertex = np.einsum(
        "vk,vkij->vij",
        asset.joint_weights,
        transforms[asset.joint_indices][..., :3, :3],
    )
    per_face = per_vertex[asset.faces].mean(axis=1)
    carried = np.einsum(
        "fij,fj->fi", per_face, face_normals(asset.rest_vertices, asset.faces)
    )
    posed_normals = face_normals(posed, asset.faces)
    return np.flatnonzero(np.einsum("ij,ij->i", carried, posed_normals) <= 0.0)


def flipped_face_count(asset: SkinnedHandAsset, transforms: Array, posed: Array) -> int:
    """Number of :func:`flipped_face_ids`."""
    return int(len(flipped_face_ids(asset, transforms, posed)))


def face_areas(vertices: Array, faces: NDArray[np.int64]) -> Array:
    """Area of every triangle."""
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
