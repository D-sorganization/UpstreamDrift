"""Engine frame conventions mapped to the canonical parity frame.

The biomech parity standard fixes the canonical frame at ``+Z`` up, ``+X``
forward and ``+Y`` left.  MuJoCo, Drake and Pinocchio already use it;
OpenSim is Y-up and needs the rotation below.

Postcondition: every returned vector is a finite length-3 ``numpy`` array.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CANONICAL_Z_UP = "z_up"
OPENSIM_Y_UP = "y_up"

# Rows copied from biomech_parity_standard.json ``frame.to_canonical``.
_TO_CANONICAL: dict[str, np.ndarray] = {
    CANONICAL_Z_UP: np.eye(3),
    OPENSIM_Y_UP: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}


def to_canonical(frame: str, vec: Sequence[float]) -> np.ndarray:
    """Rotate a world-frame vector of engine frame *frame* into the canonical frame.

    Raises:
        ValueError: If *frame* is unknown or *vec* is not a finite 3-vector.
    """
    if frame not in _TO_CANONICAL:
        raise ValueError(f"unknown engine frame {frame!r}; use {sorted(_TO_CANONICAL)}")
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"expected a finite 3-vector, got {vec!r}")
    return _TO_CANONICAL[frame] @ arr


def from_canonical(frame: str, vec: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`to_canonical` (rotation matrices are orthonormal)."""
    if frame not in _TO_CANONICAL:
        raise ValueError(f"unknown engine frame {frame!r}; use {sorted(_TO_CANONICAL)}")
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"expected a finite 3-vector, got {vec!r}")
    return _TO_CANONICAL[frame].T @ arr
