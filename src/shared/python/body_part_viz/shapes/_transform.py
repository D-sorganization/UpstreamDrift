"""Shared per-frame transform helper for shape implementations.

Applies anisotropic scale (in the shape's local frame) followed by
rotation and translation, vectorised over frames. Frames marked invalid
in ``fitted.valid_mask`` produce NaN-filled vertices rather than raising.
"""

from __future__ import annotations

import numpy as np

from .._types import FittedShape

__all__ = ["apply_fitted_to_rest_vertices"]


def apply_fitted_to_rest_vertices(
    rest_vertices: np.ndarray, fitted: FittedShape
) -> np.ndarray:
    """Return ``(T, V, 3)`` world-frame vertices for each frame.

    For each frame ``t``:
        ``world[t, v] = rotation[t] @ (scale[t] * rest[v]) + centroid[t]``.

    On invalid frames (``valid_mask[t] == False``) every vertex is NaN.
    """
    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        raise ValueError(f"rest_vertices must be (V, 3); got {rest_vertices.shape}")

    n_frames = fitted.centroid.shape[0]
    n_verts = rest_vertices.shape[0]
    out = np.full((n_frames, n_verts, 3), np.nan, dtype=np.float64)
    if n_frames == 0 or n_verts == 0:
        return out

    valid = fitted.valid_mask
    if not bool(valid.any()):
        return out

    rest = rest_vertices.astype(np.float64, copy=False)
    scale = fitted.scale[valid]
    rot = fitted.rotation_matrix[valid]
    cen = fitted.centroid[valid]

    scaled = rest[None, :, :] * scale[:, None, :]
    rotated = np.einsum("tij,tvj->tvi", rot, scaled)
    world = rotated + cen[:, None, :]
    out[valid] = world
    return out
