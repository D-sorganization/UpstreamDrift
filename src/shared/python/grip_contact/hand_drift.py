"""Drift of the prescribed hand-to-hand relative pose (issue #11986, OSV-7).

A contact grip prescribes both hands.  If the pose of the trail hand grip frame
in the lead hand grip frame changes over the swing, a rigid club held by both
must absorb that change as internal (squeeze or shear) force.  This module
measures that change from the prescribed hand frames, as translation along and
across the grip axis (in the lead hand frame) and as a rotation angle.

Frames are ``(R, p)`` series with ``R`` of shape ``(n, 3, 3)`` (world <- frame)
and ``p`` of shape ``(n, 3)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["HandDrift", "hand_relative_drift"]


@dataclass(frozen=True)
class HandDrift:
    """Relative-pose change of the trail frame in the lead frame, vs sample 0."""

    along_mm: np.ndarray  # (n,) change along the lead frame x (grip) axis
    across_mm: np.ndarray  # (n,) magnitude of the change in the lead y-z plane
    rotation_deg: np.ndarray  # (n,) angle of the relative rotation change

    @property
    def peak(self) -> dict[str, float]:
        """Peak absolute drift: ``along_mm``, ``across_mm``, ``rotation_deg``."""
        return {
            "along_mm": float(np.abs(self.along_mm).max()),
            "across_mm": float(self.across_mm.max()),
            "rotation_deg": float(self.rotation_deg.max()),
        }


def _check(frames: tuple[np.ndarray, np.ndarray], label: str) -> tuple:
    rot, pos = (np.asarray(a, dtype=float) for a in frames)
    if rot.ndim != 3 or rot.shape[1:] != (3, 3) or pos.shape != (rot.shape[0], 3):
        raise ValueError(f"{label}: need R (n,3,3) and p (n,3)")
    if rot.shape[0] < 1 or not (np.isfinite(rot).all() and np.isfinite(pos).all()):
        raise ValueError(f"{label}: frames must be finite and non-empty")
    return rot, pos


def hand_relative_drift(
    hand_pose: Mapping[str, tuple[np.ndarray, np.ndarray]],
    lead: str = "L",
    trail: str = "R",
) -> HandDrift:
    """Drift of the ``trail`` frame in the ``lead`` frame relative to sample 0.

    Postcondition: a rigidly connected pair (constant relative pose) gives
    zero drift to round-off.

    Raises:
        ValueError: on a malformed or mismatched frame series.
    """
    r_l, p_l = _check(hand_pose[lead], lead)
    r_t, p_t = _check(hand_pose[trail], trail)
    if r_l.shape[0] != r_t.shape[0]:
        raise ValueError("lead and trail series need the same number of samples")
    rel_p = np.einsum("nji,nj->ni", r_l, p_t - p_l)
    rel_r = np.einsum("nji,njk->nik", r_l, r_t)
    dp = rel_p - rel_p[0]
    dr = np.einsum("nij,kj->nik", rel_r, rel_r[0])  # R_n R_0^T
    # atan2 of the skew part keeps round-off-level angles accurate (arccos does not)
    skew = np.stack(
        [
            dr[:, 2, 1] - dr[:, 1, 2],
            dr[:, 0, 2] - dr[:, 2, 0],
            dr[:, 1, 0] - dr[:, 0, 1],
        ],
        axis=1,
    )
    cos = (np.trace(dr, axis1=1, axis2=2) - 1.0) / 2.0
    return HandDrift(
        along_mm=1e3 * dp[:, 0],
        across_mm=1e3 * np.linalg.norm(dp[:, 1:], axis=1),
        rotation_deg=np.degrees(np.arctan2(np.linalg.norm(skew, axis=1) / 2.0, cos)),
    )
