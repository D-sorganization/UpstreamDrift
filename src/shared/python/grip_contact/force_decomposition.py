"""Net and internal parts of the two hand forces on the club (issue #11739).

Definition (binding, used by every grip report).  With ``F_L`` and ``F_R`` the
forces exerted BY THE HANDS ON THE CLUB:

* net force      ``F_net = F_L + F_R`` -- the only part that accelerates the
  club centre of mass or supports its weight;
* per-hand net   ``F_net / 2`` -- the common part of both hands;
* internal force ``F_int = (F_L - F_R) / 2`` -- the antagonistic part.  The
  hands carry ``F_L = F_net/2 + F_int`` and ``F_R = F_net/2 - F_int``.

The internal force is split along the inter-hand line ``u`` (from the left
to the right grip point):

* axial part  ``F_int . u`` -- positive when the hands squeeze toward each
  other (compression of the grip pair), negative for traction;
* transverse part ``F_int - (F_int . u) u`` -- an equal and opposite force
  pair across the grip, which transmits a couple ``|F_int_t| * d`` where
  ``d`` is the hand separation.  A force pair is how two hands carry a
  pitch or yaw moment, so a large transverse internal force with a small net
  force means the club needs a large couple, not that the hands squeeze.

Input arrays may be single ``(3,)`` vectors or ``(n, 3)`` time series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceDecomposition:
    """Result of :func:`decompose_hand_forces` (all arrays in the input frame)."""

    net_n: np.ndarray
    internal_n: np.ndarray
    internal_axial_n: np.ndarray | float
    internal_transverse_n: np.ndarray
    couple_moment_nm: np.ndarray | float

    def peak_internal_n(self) -> float:
        """Largest internal-force magnitude over the series."""
        return float(np.max(np.linalg.norm(np.atleast_2d(self.internal_n), axis=1)))

    def peak_net_n(self) -> float:
        """Largest net-force magnitude over the series."""
        return float(np.max(np.linalg.norm(np.atleast_2d(self.net_n), axis=1)))


def _as_vectors(name: str, value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim not in (1, 2) or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (n, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def decompose_hand_forces(
    force_left_n: object,
    force_right_n: object,
    point_left_m: object,
    point_right_m: object,
) -> ForceDecomposition:
    """Split the two hand forces into net and internal parts.

    Preconditions: the four inputs share one shape, ``(3,)`` or ``(n, 3)``,
    are finite, and the two hand points are distinct.  Postconditions:
    ``net/2 + internal == F_L`` and ``net/2 - internal == F_R`` exactly.

    Raises:
        ValueError: on malformed, non-finite or mismatched inputs, or when the
            hand points coincide.
    """
    f_l = _as_vectors("force_left_n", force_left_n)
    f_r = _as_vectors("force_right_n", force_right_n)
    p_l = _as_vectors("point_left_m", point_left_m)
    p_r = _as_vectors("point_right_m", point_right_m)
    if not f_l.shape == f_r.shape == p_l.shape == p_r.shape:
        raise ValueError("forces and points must share one shape")
    line = p_r - p_l
    sep = np.linalg.norm(line, axis=-1)
    if np.any(sep <= 1e-9):
        raise ValueError("hand points must be distinct")
    unit = line / sep[..., None]
    net = f_l + f_r
    internal = 0.5 * (f_l - f_r)
    axial = np.sum(internal * unit, axis=-1)
    transverse = internal - axial[..., None] * unit
    couple = np.linalg.norm(transverse, axis=-1) * sep
    return ForceDecomposition(net, internal, axial, transverse, couple)
