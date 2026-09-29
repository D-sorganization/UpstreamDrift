"""Shared helpers for shape unit tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.body_part_viz import (
    BindingKind,
    FittedShape,
    MarkerBinding,
)


def make_identity_fitted(
    shape_id: str = "x",
    n_frames: int = 1,
    *,
    valid: bool = True,
    rotation: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> FittedShape:
    binding = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("a", "b"),
    )
    if rotation is None:
        rotation = np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy()
    if centroid is None:
        centroid = np.zeros((n_frames, 3))
    if scale is None:
        scale = np.ones((n_frames, 3))
    mask = np.full((n_frames,), valid, dtype=bool)
    return FittedShape(
        shape_id=shape_id,
        binding=binding,
        centroid=centroid,
        rotation_matrix=rotation,
        scale=scale,
        valid_mask=mask,
    )
