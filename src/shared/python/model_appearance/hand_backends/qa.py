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


def skinning_defects(
    result: HandVisualResult, *, min_area_m2: float
) -> dict[str, float]:
    """Degenerate and flipped triangles of a skinned result.

    ``flipped`` counts triangles folded into the inner crease of a flexed
    joint (see :func:`skinning.flipped_face_ids`); ``flipped_area_fraction``
    is their share of the skin area.
    """
    asset, transforms = _audit_parts(result)
    posed = pose_vertices(asset, transforms)
    flipped = flipped_face_ids(asset, transforms, posed)
    areas = face_areas(posed, asset.faces)
    return {
        "degenerate": degenerate_face_count(
            posed, asset.faces, min_area_m2=min_area_m2
        ),
        "flipped": int(len(flipped)),
        "flipped_area_fraction": float(areas[flipped].sum() / areas.sum()),
    }


def skinning_residual_m(result: HandVisualResult) -> float:
    """Largest distance between the mesh and pure LBS + one rigid frame.

    Zero (to rounding) proves the hand was posed by joint rotation alone:
    no vertex was pushed, clamped or inflated after skinning.
    """
    asset, transforms = _audit_parts(result)
    frame = result.frame_from_rest
    expected = pose_vertices(asset, transforms) @ frame[:3, :3].T + frame[:3, 3]
    return float(np.abs(expected - result.mesh.vertices).max())


def _audit_parts(
    result: HandVisualResult,
) -> tuple[SkinnedHandAsset, NDArray[np.float64]]:
    if (
        result.rest_asset is None
        or result.bone_transforms is None
        or result.frame_from_rest is None
    ):
        raise ValueError("result carries no skinning audit data")
    return result.rest_asset, result.bone_transforms


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
    for chunk in np.array_split(np.arange(len(points)), max(1, len(points) // 256)):
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


def _depth_inside_m(
    points: NDArray[np.float64], vertices: NDArray[np.float64], faces: NDArray
) -> float:
    """How far the deepest of ``points`` lies inside the closed mesh (0 if none).

    Depth is the distance to the nearest mesh vertex, edge midpoint or face
    centre, so it overstates true depth by at most about half an edge length.
    """
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
        asset, transforms = _audit_parts(result)
        i = asset.bone_index("forearm")
        bone = asset.bone_heads[i] - asset.bone_tails[i]
        direction = result.frame_from_rest[:3, :3] @ transforms[i][:3, :3] @ bone
        flat = direction[:2]
        return flat / np.linalg.norm(flat)

    cosine = float(np.clip(proximal(first) @ proximal(second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
