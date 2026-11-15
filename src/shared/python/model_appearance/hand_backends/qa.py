"""Quality metrics for hand visual meshes (tolerances come from settings)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    degenerate_face_count,
    face_areas,
    flipped_face_ids,
    pose_vertices,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

# Point-triangle pairs evaluated at once by the winding-number check (memory).
_WINDING_CHUNK_ELEMENTS = 400_000
_RAY_CHUNK_POINTS = 128

QUALITY_PATH = (
    Path(__file__).resolve().parents[5]
    / "scripts"
    / "config"
    / "hand_visual_quality.json"
)


@lru_cache(maxsize=1)
def load_hand_visual_quality() -> dict[str, object]:
    """Test tolerances and anthropometric references. ``ValueError`` if missing."""
    if not QUALITY_PATH.is_file():
        raise ValueError(f"missing hand visual quality settings: {QUALITY_PATH}")
    return json.loads(QUALITY_PATH.read_text(encoding="utf-8"))


def max_grip_cylinder_penetration_m(
    vertices: NDArray[np.float64], radius_m: float
) -> float:
    """Depth of the deepest vertex inside the grip cylinder about ``z`` (0 if none)."""
    rho = np.hypot(vertices[:, 0], vertices[:, 1])
    return float(max(0.0, radius_m - rho.min()))


def grip_contact_area_m2(result: HandVisualResult, within_m: float) -> float:
    """Skin area lying on the grip: triangles with every corner no farther
    than ``within_m`` from the grip cylinder about ``z``."""
    if result.grip_radius_m is None:
        raise ValueError("result carries no grip radius")
    v, f = result.mesh.vertices, result.mesh.faces
    near = (np.hypot(v[:, 0], v[:, 1]) - result.grip_radius_m <= within_m)[f].all(
        axis=1
    )
    tri = v[f[near]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def min_bone_grip_clearance_m(result: HandVisualResult, samples: int = 5) -> float:
    """Least distance from a finger or thumb bone axis to the grip surface.

    Negative means a bone lies inside the grip. Metacarpals and the forearm
    are left out: they do not wrap the grip.
    """
    asset, transforms, frame = _audit_parts(result)
    if result.grip_radius_m is None:
        raise ValueError("result carries no grip radius")
    least = np.inf
    for i, name in enumerate(asset.bone_names):
        if not name.endswith(("_prox", "_mid", "_dist")):
            continue
        t = np.linspace(0.0, 1.0, samples)[:, None]
        rest = (1.0 - t) * asset.bone_heads[i] + t * asset.bone_tails[i]
        posed = rest @ transforms[i][:3, :3].T + transforms[i][:3, 3]
        club = posed @ frame[:3, :3].T + frame[:3, 3]
        least = min(least, float(np.hypot(club[:, 0], club[:, 1]).min()))
    return float(least - result.grip_radius_m)


def skinning_defects(
    result: HandVisualResult, *, min_area_m2: float
) -> dict[str, float]:
    """Degenerate and flipped triangles of a skinned result.

    ``flipped`` counts triangles folded into the inner crease of a flexed
    joint (see :func:`skinning.flipped_face_ids`); ``flipped_area_fraction``
    is their share of the skin area.
    """
    asset, transforms, frame = _audit_parts(result)
    # The mesh itself, taken back to the rest frame: correctives count too.
    posed = (result.mesh.vertices - frame[:3, 3]) @ np.linalg.inv(frame[:3, :3]).T
    flipped = flipped_face_ids(asset, transforms, posed)
    areas = face_areas(posed, asset.faces)
    return {
        "degenerate": degenerate_face_count(
            posed, asset.faces, min_area_m2=min_area_m2
        ),
        "flipped": int(len(flipped)),
        "flipped_area_fraction": float(areas[flipped].sum() / areas.sum()),
    }


def pinched_edge_count(
    result: HandVisualResult, bones: tuple[str, ...], max_added_fold_deg: float
) -> int:
    """Edges on skin led by ``bones`` that fold more than ``max_added_fold_deg``
    beyond their fold at rest: creases skinning pinched into the skin."""
    from src.shared.python.model_appearance.hand_backends.correctives import (
        edge_folds,
    )

    asset, _, _ = _audit_parts(result)
    now, pair = edge_folds(result.mesh.vertices, asset.faces)
    rest, _ = edge_folds(asset.rest_vertices, asset.faces)
    mask = np.zeros(len(asset.rest_vertices), dtype=bool)
    mask[asset.dominant_vertex_ids(bones)] = True
    inside = mask[asset.faces].all(axis=1)
    both = inside[pair[:, 0]] & inside[pair[:, 1]]
    return int(((now - rest > max_added_fold_deg) & both).sum())


def skinning_residual_m(result: HandVisualResult) -> float:
    """Largest distance between the mesh and pure LBS + one rigid frame.

    Zero (to rounding) proves the hand was posed by joint rotation alone:
    no vertex was pushed, clamped or inflated after skinning. With the
    corrective shapes on it is the farthest any vertex was moved by them.
    """
    asset, transforms, frame = _audit_parts(result)
    expected = pose_vertices(asset, transforms) @ frame[:3, :3].T + frame[:3, 3]
    return float(np.abs(expected - result.mesh.vertices).max())


def _audit_parts(
    result: HandVisualResult,
) -> tuple[SkinnedHandAsset, NDArray[np.float64], NDArray[np.float64]]:
    if (
        result.rest_asset is None
        or result.bone_transforms is None
        or result.frame_from_rest is None
    ):
        raise ValueError("result carries no skinning audit data")
    return result.rest_asset, result.bone_transforms, result.frame_from_rest


def anthropometric_ratios(
    asset: SkinnedHandAsset, breadth_band_m: float
) -> dict[str, float]:
    """Ratios of a rest hand, to compare with the references in settings.

    Lengths are joint-chain lengths (knuckle to tip; thumb from its base).
    """
    from src.shared.python.model_appearance.hand_backends.skinned_human import (
        asset_hand_breadth_m,
        asset_hand_length_m,
    )

    def chain(names: tuple[str, ...]) -> float:
        ids = [asset.bone_index(n) for n in names]
        return float(
            np.linalg.norm(asset.bone_tails[ids] - asset.bone_heads[ids], axis=1).sum()
        )

    finger = {
        f: chain(tuple(f"{f}_{s}" for s in ("prox", "mid", "dist"))) for f in FINGERS
    }
    thumb = chain(("thumb_meta", "thumb_prox", "thumb_dist"))
    length = asset_hand_length_m(asset)
    return {
        "breadth_over_length": asset_hand_breadth_m(asset, breadth_band_m) / length,
        "middle_finger_over_length": finger["middle"] / length,
        "index_over_middle": finger["index"] / finger["middle"],
        "ring_over_middle": finger["ring"] / finger["middle"],
        "little_over_middle": finger["little"] / finger["middle"],
        "thumb_over_middle": thumb / finger["middle"],
    }


def _winding_numbers(
    points: NDArray[np.float64], vertices: NDArray[np.float64], faces: NDArray
) -> NDArray[np.float64]:
    """Winding number of a closed outward-wound mesh about each point."""
    total = np.zeros(len(points))
    per_chunk = max(1, _WINDING_CHUNK_ELEMENTS // max(1, len(faces)))
    for chunk in np.array_split(
        np.arange(len(points)), max(1, len(points) // per_chunk)
    ):
        rel = vertices[faces][None, :, :, :] - points[chunk][:, None, None, :]
        a, b, c = rel[:, :, 0], rel[:, :, 1], rel[:, :, 2]
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        triple = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        denom = (
            la * lb * lc
            + np.einsum("ijk,ijk->ij", a, b) * lc
            + np.einsum("ijk,ijk->ij", b, c) * la
            + np.einsum("ijk,ijk->ij", c, a) * lb
        )
        total[chunk] = np.arctan2(triple, denom).sum(axis=1) / (2.0 * np.pi)
    return total


def _edge_side(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    point: NDArray[np.float64],
    plane: tuple[int, int],
) -> NDArray[np.float64]:
    """Signed area of ``start -> end -> point`` projected on the ``plane`` axes."""
    u, v = plane
    return (end[..., u] - start[..., u]) * (point[..., v] - start[..., v]) - (
        end[..., v] - start[..., v]
    ) * (point[..., u] - start[..., u])


def _ray_says_inside(
    points: NDArray[np.float64], tri: NDArray[np.float64], axis: int
) -> NDArray[np.bool_]:
    """Whether a ray from each point along ``+axis`` crosses the closed mesh
    an odd number of times. Fast, but a ray through an edge can miscount, so
    it only preselects points for the exact winding number."""
    u, v = (k for k in range(3) if k != axis)
    order = np.argsort(points[:, u], kind="stable")
    low, high = tri[:, :, u].min(axis=1), tri[:, :, u].max(axis=1)
    inside = np.zeros(len(points), dtype=bool)
    for chunk in np.array_split(order, max(1, len(order) // _RAY_CHUNK_POINTS)):
        if len(chunk) == 0:
            continue
        p = points[chunk]
        near = tri[(high >= p[:, u].min()) & (low <= p[:, u].max())]
        a, b, c = near[None, :, 0], near[None, :, 1], near[None, :, 2]
        q = p[:, None, :]
        ab, bc, ca = (_edge_side(s, e, q, (u, v)) for s, e in ((a, b), (b, c), (c, a)))
        area = ab + bc + ca
        over = ((ab >= 0) & (bc >= 0) & (ca >= 0)) | ((ab <= 0) & (bc <= 0) & (ca <= 0))
        over &= area != 0.0
        safe = np.where(area == 0.0, 1.0, area)
        height = (bc * a[..., axis] + ca * b[..., axis] + ab * c[..., axis]) / safe
        inside[chunk] = (over & (height > q[..., axis])).sum(axis=1) % 2 == 1
    return inside


def _depth_inside_m(
    points: NDArray[np.float64], vertices: NDArray[np.float64], faces: NDArray
) -> float:
    """How far the deepest of ``points`` lies inside the closed mesh (0 if none).

    Depth is the distance to the nearest mesh vertex, edge midpoint or face
    centre, so it overstates true depth by at most about half an edge length.
    """
    # A point outside the mesh's bounding box is outside the mesh; of the
    # rest only those a ray test calls inside get the exact winding number.
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    points = points[((points >= low) & (points <= high)).all(axis=1)]
    tri = vertices[faces]
    maybe = _ray_says_inside(points, tri, 2) | _ray_says_inside(points, tri, 1)
    points = points[maybe]
    inside = points[_winding_numbers(points, vertices, faces) > 0.5]
    if len(inside) == 0:
        return 0.0
    tri = vertices[faces]
    samples = np.vstack(
        [vertices, tri.mean(axis=1)]
        + [0.5 * (tri[:, i] + tri[:, (i + 1) % 3]) for i in range(3)]
    )
    gaps = np.linalg.norm(inside[:, None, :] - samples[None, :, :], axis=2)
    return float(gaps.min(axis=1).max())


def hand_pair_penetration_m(first: HandVisualResult, second: HandVisualResult) -> float:
    """Deepest that either hand's skin vertices sit inside the other hand."""
    a, b = first.mesh, second.mesh
    return max(
        _depth_inside_m(a.vertices, b.vertices, b.faces),
        _depth_inside_m(b.vertices, a.vertices, a.faces),
    )


def forearm_spread_deg(first: HandVisualResult, second: HandVisualResult) -> float:
    """Angle between the two forearms seen along the shaft (from the butt)."""

    def proximal(result: HandVisualResult) -> NDArray[np.float64]:
        asset, transforms, frame = _audit_parts(result)
        i = asset.bone_index("forearm")
        bone = asset.bone_heads[i] - asset.bone_tails[i]
        direction = frame[:3, :3] @ transforms[i][:3, :3] @ bone
        flat = direction[:2]
        return flat / np.linalg.norm(flat)

    cosine = float(np.clip(proximal(first) @ proximal(second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
