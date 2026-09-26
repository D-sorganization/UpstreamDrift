"""Kinematic trajectory smoothing, derivative compatibility, and spike auditing (PF-02, #10432).

Provides:
1. Joint trajectory smoothing for generalized coordinates (q) with analytical/numerical
   derivative compatibility (q_dot ≈ v, v_dot ≈ a).
2. Boundary spike and discontinuity auditing (BoundarySpikeAudit) checking for jerk
   or acceleration jumps at window boundaries.
3. Cutoff frequency sensitivity analysis to balance marker fit fidelity against
   unphysical acceleration spikes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class BoundarySpikeAudit:
    """Audit report for boundary acceleration and jerk discontinuities."""

    has_spikes: bool
    worst_boundary_jerk: float
    max_allowed_jerk: float
    initial_jerk: float
    terminal_jerk: float
    spike_locations: tuple[str, ...]
    details: dict[str, Any]


@dataclass(frozen=True)
class CutoffSensitivityReport:
    """Sensitivity analysis across candidate filter cutoff frequencies."""

    cutoffs_hz: tuple[float, ...]
    position_rmse: tuple[float, ...]
    max_accelerations: tuple[float, ...]
    boundary_jerks: tuple[float, ...]
    recommended_cutoff_hz: float


@dataclass(frozen=True)
class SmoothedTrajectory:
    """Smoothed joint trajectory with analytical/numerical derivative compatibility.

    Attributes
    ----------
    q:
        Joint coordinate array matching input shape.
    v:
        Derivative-compatible joint velocities.
    a:
        Derivative-compatible joint accelerations.
    smoothing_applied:
        True if zero-phase Butterworth filter was successfully applied;
        False if SciPy filtering failed and unsmoothed coordinates were retained.
    """

    q: Array
    v: Array
    a: Array
    smoothing_applied: bool = True

    def __iter__(self) -> Any:
        return iter((self.q, self.v, self.a))

    def __getitem__(self, idx: int) -> Array:
        return (self.q, self.v, self.a)[idx]

    def __len__(self) -> int:
        return 3


def smooth_kinematic_trajectory(
    time_s: Array,
    q: Array,
    *,
    cutoff_hz: float = 15.0,
    order: int = 4,
    padlen: int | None = None,
) -> SmoothedTrajectory:
    """Smooth joint trajectory q and compute derivative-compatible v and a.

    Parameters
    ----------
    time_s:
        Strictly increasing 1D timestamp array of shape (N,).
    q:
        Joint coordinate array of shape (N, D) or (N,).
    cutoff_hz:
        Low-pass filter cutoff frequency in Hz.
    order:
        Butterworth filter order (default 4).
    padlen:
        Number of reflection padding nodes for filtfilt. Default min(N - 1, 24).

    Returns
    -------
    SmoothedTrajectory:
        SmoothedTrajectory(q, v, a, smoothing_applied) with matching shape to input q.
    """
    t = np.asarray(time_s, dtype=np.float64)
    require(t.ndim == 1, "time_s must be a 1D array", t.shape)
    n_nodes = len(t)
    require(n_nodes >= 4, "time_s must have at least 4 nodes", n_nodes)
    require(
        bool(np.all(np.diff(t) > 0.0)),
        "time_s must be strictly monotonically increasing",
    )

    q_arr = np.asarray(q, dtype=np.float64)
    require(
        q_arr.shape[0] == n_nodes,
        "q length must match time_s length",
        (q_arr.shape[0], n_nodes),
    )
    require(bool(np.all(np.isfinite(q_arr))), "q must contain only finite numbers")
    require(cutoff_hz > 0.0, "cutoff_hz must be strictly positive", cutoff_hz)

    was_1d = q_arr.ndim == 1
    q_2d = q_arr[:, None] if was_1d else q_arr

    # Estimate sampling frequency
    duration = float(t[-1] - t[0])
    require(duration > 0.0, "trajectory duration must be positive", duration)
    fs = float(n_nodes - 1) / duration
    f_nyq = 0.5 * fs

    eff_cutoff = min(cutoff_hz, 0.95 * f_nyq)
    require(eff_cutoff > 0.0, "effective cutoff must be positive", eff_cutoff)

    # Filter with zero-phase Butterworth filter via SOS representation
    smoothing_applied = True
    try:
        from scipy import signal

        sos = signal.butter(order, eff_cutoff, fs=fs, btype="low", output="sos")
        effective_padlen = (
            min(n_nodes - 1, 24) if padlen is None else min(n_nodes - 1, padlen)
        )
        q_smooth_2d = signal.sosfiltfilt(sos, q_2d, axis=0, padlen=effective_padlen)
    except (ValueError, TypeError, ImportError):
        # Fallback if scipy signal raises or is unavailable
        q_smooth_2d = q_2d.copy()
        smoothing_applied = False

    # Compute derivative-compatible velocities and accelerations
    v_smooth_2d = np.gradient(q_smooth_2d, t, axis=0)
    a_smooth_2d = np.gradient(v_smooth_2d, t, axis=0)

    if was_1d:
        q_out = q_smooth_2d[:, 0]
        v_out = v_smooth_2d[:, 0]
        a_out = a_smooth_2d[:, 0]
    else:
        q_out = q_smooth_2d
        v_out = v_smooth_2d
        a_out = a_smooth_2d

    ensure(q_out.shape == q_arr.shape, "q_out shape must match q shape")
    ensure(v_out.shape == q_arr.shape, "v_out shape must match q shape")
    ensure(a_out.shape == q_arr.shape, "a_out shape must match q shape")
    return SmoothedTrajectory(
        q=q_out,
        v=v_out,
        a=a_out,
        smoothing_applied=smoothing_applied,
    )


def audit_boundary_spikes(
    time_s: Array,
    q: Array,
    v: Array,
    a: Array,
    *,
    max_allowed_jerk: float = 500.0,
) -> BoundarySpikeAudit:
    """Audit boundary jerk and acceleration steps at the initial and terminal edges.

    Parameters
    ----------
    time_s:
        Timestamp vector of shape (N,).
    q, v, a:
        State trajectories of shape (N, D) or (N,).
    max_allowed_jerk:
        Maximum permissible boundary jerk in rad/s^3 or m/s^3.

    Returns
    -------
    BoundarySpikeAudit
    """
    t = np.asarray(time_s, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    require(t.ndim == 1, "time_s must be 1D")
    require(len(t) >= 2, "time_s must have at least 2 nodes")
    require(a_arr.shape[0] == len(t), "a length must match time_s")

    a_2d = a_arr[:, None] if a_arr.ndim == 1 else a_arr

    dt_init = float(t[1] - t[0])
    dt_term = float(t[-1] - t[-2])
    require(dt_init > 0.0 and dt_term > 0.0, "dt must be strictly positive")

    # Boundary jerk estimates
    j_init_vec = np.abs((a_2d[1] - a_2d[0]) / dt_init)
    j_term_vec = np.abs((a_2d[-1] - a_2d[-2]) / dt_term)

    init_jerk = float(np.max(j_init_vec)) if j_init_vec.size else 0.0
    term_jerk = float(np.max(j_term_vec)) if j_term_vec.size else 0.0
    worst_jerk = max(init_jerk, term_jerk)

    spikes: list[str] = []
    if init_jerk > max_allowed_jerk:
        spikes.append("initial")
    if term_jerk > max_allowed_jerk:
        spikes.append("terminal")

    has_spikes = len(spikes) > 0

    return BoundarySpikeAudit(
        has_spikes=has_spikes,
        worst_boundary_jerk=worst_jerk,
        max_allowed_jerk=float(max_allowed_jerk),
        initial_jerk=init_jerk,
        terminal_jerk=term_jerk,
        spike_locations=tuple(spikes),
        details={
            "dt_init": dt_init,
            "dt_term": dt_term,
            "n_nodes": len(t),
            "max_a": float(np.max(np.abs(a_arr))),
        },
    )


def audit_cutoff_sensitivity(
    time_s: Array,
    q: Array,
    *,
    cutoffs_hz: Sequence[float] = (10.0, 15.0, 20.0, 30.0),
    max_allowed_jerk: float = 500.0,
) -> CutoffSensitivityReport:
    """Evaluate filtering sensitivity across candidate cutoff frequencies."""
    q_arr = np.asarray(q, dtype=np.float64)
    rmses: list[float] = []
    max_accels: list[float] = []
    boundary_jerks: list[float] = []

    best_cutoff = float(cutoffs_hz[0])
    best_score = float("inf")

    for fc in cutoffs_hz:
        res = smooth_kinematic_trajectory(time_s, q_arr, cutoff_hz=fc)
        if not res.smoothing_applied:
            rmses.append(float("nan"))
            max_accels.append(float("nan"))
            boundary_jerks.append(float("nan"))
            continue

        q_s, v_s, a_s = res.q, res.v, res.a
        rmse = float(np.sqrt(np.mean((q_s - q_arr) ** 2)))
        max_a = float(np.max(np.abs(a_s)))
        audit = audit_boundary_spikes(
            time_s, q_s, v_s, a_s, max_allowed_jerk=max_allowed_jerk
        )
        rmses.append(rmse)
        max_accels.append(max_a)
        boundary_jerks.append(audit.worst_boundary_jerk)

        # Heuristic score: lower RMSE is better, penalized if jerk exceeds limit
        penalty = 10.0 if audit.has_spikes else 1.0
        score = rmse * penalty
        if score < best_score:
            best_score = score
            best_cutoff = float(fc)

    return CutoffSensitivityReport(
        cutoffs_hz=tuple(float(fc) for fc in cutoffs_hz),
        position_rmse=tuple(rmses),
        max_accelerations=tuple(max_accels),
        boundary_jerks=tuple(boundary_jerks),
        recommended_cutoff_hz=best_cutoff,
    )
