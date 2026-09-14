"""Unit-separated diagnostics for the ST-01 feasibility probe (#10124)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError("Evidence must be a nonempty finite two-dimensional matrix")
    return array


def closure_metrics(residuals: ArrayLike) -> dict[str, float]:
    """Summarize Nx6 displacement/rotation-vector errors without mixing units.

    Postcondition: translation norms are in metres, rotation norms in radians;
    input arrays are unchanged. Missing/nonfinite evidence raises ValueError.
    """
    values = _matrix(residuals)
    if values.shape[1] != 6:
        raise ValueError("Closure evidence must have six columns: xyz m, rotvec rad")
    return {
        "max_grip_translation_m": float(np.linalg.norm(values[:, :3], axis=1).max()),
        "max_grip_rotation_rad": float(np.linalg.norm(values[:, 3:], axis=1).max()),
    }


def replay_difference(
    reference: ArrayLike,
    candidate: ArrayLike,
    translation_indices: tuple[int, ...],
) -> dict[str, float]:
    """Compare aligned scalar-coordinate traces with explicit translation slots.

    The caller must verify equal time grids and coordinate order. This function
    accepts scalar native coordinates/rates, not quaternions. Postcondition:
    separate maximum component errors retain input units (m/rad or m/s/rad/s).
    """
    first, second = _matrix(reference), _matrix(candidate)
    if first.shape != second.shape:
        raise ValueError("Replays must have equal shapes")
    size = first.shape[1]
    if (
        not translation_indices
        or len(set(translation_indices)) != len(translation_indices)
        or any(type(i) is not int or i < 0 or i >= size for i in translation_indices)
        or len(translation_indices) == size
    ):
        raise ValueError("Translation indices must be unique valid scalar slots")
    rotation_indices = [i for i in range(size) if i not in translation_indices]
    difference = np.abs(first - second)
    return {
        "max_translation_difference": float(difference[:, translation_indices].max()),
        "max_rotation_difference": float(difference[:, rotation_indices].max()),
    }
