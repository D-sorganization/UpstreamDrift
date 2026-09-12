"""Prepare a bounded native polynomial restart without shifting its envelope."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
    recover_native_bernstein,
)


class NativeRestart(NamedTuple):
    """Recovered controls and the exact candidate they reconstruct.

    Callers must independently replay this candidate to build new node charts.
    Numerical reconstruction can change its hash; this is not fit acceptance.
    """

    controls: NDArray[np.float64]
    candidate: NativeReplayCandidate
    source_candidate_sha256: str
    max_bound_snap: float
    snapped_control_count: int


def prepare_native_restart(
    base: NativeReplayCandidate,
    seed: NativeReplayCandidate,
    *,
    basis_duration_s: float,
    lower_controls: NDArray[np.float64],
    upper_controls: NDArray[np.float64],
    roundoff_tolerance: float,
) -> NativeRestart:
    """Recover controls relative to base and snap only declared bound roundoff.

    Bounds have the native (coordinate, Bernstein control) layout and remain
    relative to the original base, never to seed. Non-control identities must
    match. A violation larger than the explicit absolute tolerance is rejected.
    The returned candidate preserves geometry, original q0/qd0 and time basis.
    """
    controls = recover_native_bernstein(base, seed, basis_duration_s=basis_duration_s)
    lo, hi = (
        np.asarray(value, dtype=float) for value in (lower_controls, upper_controls)
    )
    if (
        lo.shape != controls.shape
        or hi.shape != controls.shape
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(lo >= hi)
        or not np.isfinite(roundoff_tolerance)
        or roundoff_tolerance < 0
    ):
        raise ValueError("Invalid restart bounds or roundoff tolerance")
    bounded = np.clip(controls, lo, hi)
    snap = np.abs(bounded - controls)
    if np.max(snap) > roundoff_tolerance:
        raise ValueError("Restart controls violate original bounds")
    candidate = increment_native_bernstein(
        base, bounded, basis_duration_s=basis_duration_s
    )
    bounded.setflags(write=False)
    return NativeRestart(
        bounded,
        candidate,
        seed.sha256,
        float(np.max(snap)),
        int(np.count_nonzero(snap)),
    )
