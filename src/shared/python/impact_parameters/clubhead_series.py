"""World-frame clubhead time series input (GCV-15)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _array(value: object, name: str, n: int | None, optional: bool = False):
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} is required")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class ClubheadSeries:
    """Face-centre kinematics in the world frame.

    ``face_normal`` / ``toe_axis`` / ``grip_axis`` may be ``None`` when club
    axial rotation is unobservable (e.g. mocap-matched swings, MS-108); then
    ``face_unobservable_reason`` must say why and every face-derived quantity
    is reported unavailable, never invented.
    """

    times_s: np.ndarray
    face_center_m: np.ndarray
    velocity_mps: np.ndarray
    face_normal: np.ndarray | None = None
    toe_axis: np.ndarray | None = None
    grip_axis: np.ndarray | None = None
    face_unobservable_reason: str | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.times_s, dtype=float)
        if t.ndim != 1 or t.shape[0] < 2:
            raise ValueError("times_s must be 1-D with at least 2 samples")
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("times_s must be finite and strictly increasing")
        n = t.shape[0]
        object.__setattr__(self, "times_s", t)
        for name in ("face_center_m", "velocity_mps"):
            object.__setattr__(self, name, _array(getattr(self, name), name, n))
        for name in ("face_normal", "toe_axis", "grip_axis"):
            object.__setattr__(
                self, name, _array(getattr(self, name), name, n, optional=True)
            )
        if self.face_normal is None and not self.face_unobservable_reason:
            raise ValueError(
                "face_normal is None: face_unobservable_reason must be given"
            )
        if self.face_normal is not None:
            norms = np.linalg.norm(self.face_normal, axis=1)
            if np.any(norms < 1e-9):
                raise ValueError("face_normal rows must be nonzero")

    def __len__(self) -> int:
        return int(self.times_s.shape[0])
