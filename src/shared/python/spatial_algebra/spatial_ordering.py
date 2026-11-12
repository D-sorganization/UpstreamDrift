"""Task-space 6-D row ordering tags and conversion helpers."""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = [
    "SpatialOrdering",
    "permute_task_jacobian",
    "spatial_row_permutation",
]


class SpatialOrdering(str, Enum):
    """Row order for 6-D task Jacobians and wrenches.

    ``LINEAR_FIRST`` — ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]``; used by
    :mod:`operational_space` and transport helpers.

    ``ANGULAR_FIRST`` — Featherstone / recursive rigid-body conventions:
    ``[ω_x, ω_y, ω_z, v_x, v_y, v_z]``.
    """

    LINEAR_FIRST = "linear_first"
    ANGULAR_FIRST = "angular_first"


def spatial_row_permutation(
    source: SpatialOrdering,
    target: SpatialOrdering,
) -> np.ndarray:
    """Return indices ``p`` with ``rows_target = rows_source[p]``."""
    if source == target:
        return np.arange(6, dtype=int)
    if (
        source == SpatialOrdering.LINEAR_FIRST
        and target == SpatialOrdering.ANGULAR_FIRST
    ):
        return np.array([3, 4, 5, 0, 1, 2], dtype=int)
    if (
        source == SpatialOrdering.ANGULAR_FIRST
        and target == SpatialOrdering.LINEAR_FIRST
    ):
        return np.array([3, 4, 5, 0, 1, 2], dtype=int)
    raise ValueError(f"unsupported ordering conversion {source!r} -> {target!r}")


def permute_task_jacobian(
    jacobian: np.ndarray,
    source: SpatialOrdering,
    target: SpatialOrdering = SpatialOrdering.LINEAR_FIRST,
) -> np.ndarray:
    """Permute task rows of a (k, nv) Jacobian; k must be 3 or 6."""
    mat = np.asarray(jacobian, dtype=float)
    k = mat.shape[0]
    if k == 3:
        if source != target:
            raise ValueError("3-row Jacobians have no angular/translational split")
        return mat
    if k != 6:
        raise ValueError(f"Jacobian must have 3 or 6 rows, got {k}")
    perm = spatial_row_permutation(source, target)
    return mat[perm, :]
