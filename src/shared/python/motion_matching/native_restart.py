"""Prepare native restarts and explicit numerical control-envelope continuation."""

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


def widen_control_envelope(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    selected: NDArray[np.bool_],
    *,
    factor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Widen selected numerical intervals about their unchanged centers.

    Selection and factor are explicit experimental policy, never inferred from
    fit failure. Unselected intervals remain bit-identical. These bounds apply
    to correction controls; they do not declare total physical actuator limits
    or bound polynomial values outside a Bernstein basis interval. Callers must
    preserve prior envelopes and record this change in a new run manifest.
    """
    if isinstance(factor, (bool, np.bool_)) or not np.isfinite(factor) or factor <= 1:
        raise ValueError("Expansion factor must be finite and greater than one")
    lo, hi = (np.array(value, dtype=float, copy=True) for value in (lower, upper))
    mask = np.asarray(selected)
    if (
        not lo.size
        or hi.shape != lo.shape
        or mask.shape != lo.shape
        or mask.dtype != np.bool_
        or not mask.any()
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(lo >= hi)
    ):
        raise ValueError("Invalid bounds or explicit control selection")
    with np.errstate(over="ignore", invalid="ignore"):
        increment = (factor - 1) * (hi[mask] / 2 - lo[mask] / 2)
        lo[mask] -= increment
        hi[mask] += increment
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Expanded control bounds must remain finite")
    lo.setflags(write=False)
    hi.setflags(write=False)
    return lo, hi
