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
