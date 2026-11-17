"""Rest-shape changes to the skinned hand asset (epic #15150).

The CC0 hand is thin over the base of the thumb: with the joint straightened
the skin there reads as a hollow where a real hand carries tissue. A fullness
pad adds that tissue to the rest mesh, before subdivision and skinning, so it
is part of the hand in every pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from src.shared.python.model_appearance.hand_backends.skinning import (
    SkinnedHandAsset,
    vertex_normals,
)


@dataclass(frozen=True)
class FullnessSettings:
    """Tissue added on the back of one bone (the side it flexes away from).

    Skin whose strongest bone is one of ``bones`` moves out along its normal
    by up to ``height_m`` (``0`` switches the pad off). The pad is centred at
    station ``centre`` along ``along_bone`` (0 = head, 1 = tail, in bone
    lengths) and fades to nothing ``half_span`` bone lengths either side and
    ``radius_m`` from the bone's line. A vertex moves in full once the cosine
    between its normal and the back of the bone reaches ``facing_full``.
    """

    along_bone: str
    bones: tuple[str, ...]
    height_m: float
    centre: float
    half_span: float
    radius_m: float
    facing_full: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.height_m) and self.height_m >= 0.0):
            raise ValueError("fullness height_m must be finite and not negative")
        if not math.isfinite(self.centre):
            raise ValueError("fullness centre must be finite")
        if not (self.half_span > 0.0 and self.radius_m > 0.0):
            raise ValueError("fullness half_span and radius_m must be positive")
        if not 0.0 < self.facing_full <= 1.0:
            raise ValueError("fullness facing_full must be in (0, 1]")
        if not self.bones:
            raise ValueError("fullness bones must name at least one bone")


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return np.asarray(t * t * (3.0 - 2.0 * t))


def _pad_shift(asset: SkinnedHandAsset, pad: FullnessSettings) -> np.ndarray:
    """Per-vertex outward shift of one pad, in the rest frame."""
    bone = asset.bone_index(pad.along_bone)
    unknown = set(pad.bones) - set(asset.bone_names)
    if unknown:
        raise ValueError(f"fullness bones names no bone: {sorted(unknown)}")
    head = asset.bone_heads[bone]
    span = asset.bone_tails[bone] - head
    length = float(np.linalg.norm(span))
    along = span / length
    back = -np.cross(asset.flex_axes[bone], along)
    if not float(np.linalg.norm(back)) > 0.0:
        raise ValueError(f"fullness bone {pad.along_bone!r} does not flex")
    back /= np.linalg.norm(back)
    ids = asset.dominant_vertex_ids(pad.bones)
    normals = vertex_normals(asset.rest_vertices, asset.faces)[ids]
    rel = asset.rest_vertices[ids] - head
    station = rel @ along / length
    off_axis = np.linalg.norm(rel - np.outer(rel @ along, along), axis=1)
    weight = (
        _smoothstep(1.0 - np.abs(station - pad.centre) / pad.half_span)
        * _smoothstep(2.0 * (1.0 - off_axis / pad.radius_m))
        * _smoothstep(normals @ back / pad.facing_full)
    )
    shift = np.zeros_like(asset.rest_vertices)
    shift[ids] = pad.height_m * weight[:, None] * normals
    return shift


def filled_out(
    asset: SkinnedHandAsset, pads: tuple[FullnessSettings, ...]
) -> SkinnedHandAsset:
    """``asset`` with every fullness pad added to its rest mesh.

    Postconditions: no vertex moves farther than the sum of the pad heights;
    faces, skeleton and weights are shared with ``asset``; with no pad of
    positive height ``asset`` itself is returned. Raises ``ValueError`` for a
    bone that does not exist or an ``along_bone`` that has no hinge.
    """
    shift = np.zeros_like(asset.rest_vertices)
    for pad in pads:
        shift += _pad_shift(asset, pad)
    if not shift.any():
        return asset
    return replace(asset, rest_vertices=asset.rest_vertices + shift)


__all__ = ["FullnessSettings", "filled_out"]
