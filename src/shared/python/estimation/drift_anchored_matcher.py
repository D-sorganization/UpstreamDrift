"""Recursive ZTCF-anchored kinematic matcher with overlaid local torque bands.

Implements the owner's ZTCF prediction strategy for epic #11421:

* **Predict.** Each window starts from "ZTCF + the same torque as before".
* **Bound.** Admissible torques form a tight, rate-limited band around the
  previous torque; together with the drift they define the viable range of
  next states. Observations outside it are ruled out (outliers/occlusion).
* **Recurse.** Windows march through the data. Each inherits the previous
  fitted state and torque, so torque continuity is enforced and windows with
  no usable data fall back to the drift + constant-torque prediction.
* **Overlay.** The torque profile is the tapered, inverse-variance overlay of
  the small local solutions; where they disagree, the data are suspect.
* **Replay.** One uninterrupted forward-dynamics simulation from the first
  fitted state under the assembled torque validates the match.

Carried arrival information is a fixed-std approximation of the marginalised
prior (DIME-07 owns a proper arrival cost). Contact and manifold coordinates
are out of scope; see :mod:`drift_prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import check_finite, ensure, require
from src.shared.python.estimation.drift_prediction import (
    ControlBand,
    drift_dominance_index,
    linearize_drift,
)
from src.shared.python.estimation.local_torque_window import (
    LocalWindowSolution,
    WindowObservation,
    WindowOptions,
    rollout,
    solve_local_window,
)

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.protocol import DynamicsProvider

__all__ = [
    "DriftAnchoredMatchResult",
    "MatchOptions",
    "SampleLabel",
    "WindowRecord",
    "match_kinematics",
]


class SampleLabel(StrEnum):
    """Per-sample verdict of the matcher."""

    ACCEPTED = "accepted"  # observed and explained by bounded dynamics
    OUTLIER = "outlier"  # short run ruled out by the viable band / residual
    UNEXPLAINED = "unexplained"  # persistent run no admissible torque reaches
    GAP_FILLED = "gap_filled"  # missing; estimated from dynamics only


@dataclass(frozen=True)
class MatchOptions:
    """Window geometry, carried priors and labelling thresholds.

    Attributes:
        window: Options for each local solve (first window uses its arrival
            stds; later windows use the carried stds below).
        window_steps: Steps per window (samples = steps + 1).
        stride: Steps between window starts (``<= window_steps``); overlap is
            ``window_steps - stride``.
        carry_sigma_q: Arrival std for a carried start configuration [rad].
        carry_sigma_v: Arrival std for a carried start velocity [rad/s].
        initial_torque: Torque assumed before the first sample (``None`` =
            zeros, a declared initialisation, not an inactivity claim).
        max_outlier_run: Longest rejected run still called an outlier; longer
            runs are ``UNEXPLAINED``.
        reject_vote: Fraction of covering windows that must reject a sample.
        min_observed: Observed samples a window needs; sparser windows are
            extended (bridging gaps with data on both sides). ``None`` =
            half the window samples.
        max_window_steps: Extension cap (``None`` = 4 x ``window_steps``).
        reacquire_sigma_q: Arrival std [rad] used after a failed window, when
            the drift + constant-torque fallback may have walked off.
    """

    window: WindowOptions
    window_steps: int = 12
    stride: int = 4
    carry_sigma_q: float | None = None
    carry_sigma_v: float | None = None
    initial_torque: np.ndarray | None = None
    max_outlier_run: int = 2
    reject_vote: float = 0.5
    min_observed: int | None = None
    max_window_steps: int | None = None
    reacquire_sigma_q: float = 0.05

    def __post_init__(self) -> None:
        require(self.window_steps >= 2, "window_steps must be >= 2")
        require(
            1 <= self.stride <= self.window_steps,
            "stride must lie in [1, window_steps]",
            (self.stride, self.window_steps),
        )
        for name in ("carry_sigma_q", "carry_sigma_v"):
            value = getattr(self, name)
            require(value is None or value > 0.0, f"{name} must be > 0")
        require(self.max_outlier_run >= 1, "max_outlier_run must be >= 1")
        require(0.0 < self.reject_vote <= 1.0, "reject_vote must lie in (0, 1]")
        require(
            self.min_observed is None or self.min_observed >= 2,
            "min_observed must be >= 2",
        )
        require(
            self.max_window_steps is None or self.max_window_steps >= self.window_steps,
            "max_window_steps must be >= window_steps",
        )
        require(self.reacquire_sigma_q > 0.0, "reacquire_sigma_q must be > 0")

    @property
    def required_observed(self) -> int:
        """Observed samples a window must contain before it is solved."""
        return self.min_observed or (self.window_steps + 2) // 2

    @property
    def extension_cap(self) -> int:
        """Largest number of steps an extended window may span."""
        return self.max_window_steps or 4 * self.window_steps

    def window_options_for(
        self, steps: int, *, first: bool, reacquire: bool
    ) -> WindowOptions:
        """Options for a window of ``steps`` (knots scale with its length)."""
        base = self.window if first else self.carried_window()
        if reacquire:
            base = replace(
                self.window, sigma_q0=max(self.reacquire_sigma_q, base.sigma_q0)
            )
        knots = base.n_knots * int(np.ceil(steps / self.window_steps))
        return replace(base, n_knots=knots)

    def carried_window(self) -> WindowOptions:
        """Window options for windows whose start state is carried."""
        return replace(
            self.window,
            sigma_q0=self.carry_sigma_q or self.window.sigma_q0,
            sigma_v0=self.carry_sigma_v or self.window.sigma_v0,
        )


@dataclass(frozen=True)
class WindowRecord:
    """Summary of one local solve (for reports and failure receipts)."""

    start: int
    steps: int
    success: bool
    status: str
    chi2_per_dof: float
    raw_rms_residual: float
    explained_fraction: float
    saturated: np.ndarray
    rate_limited: np.ndarray
    torque_condition_number: float


@dataclass(frozen=True)
class DriftAnchoredMatchResult:
    """Assembled match. Per-sample arrays have ``T`` rows; ``tau[k]`` acts
    during step ``k`` (the final row repeats the last step).

    ``tau_std`` is the taper-weighted mean of the window posterior stds
    (overlapping windows share data, so it is deliberately not combined as
    if independent); samples no successful window covers get the band
    half-width. ``tau_disagreement`` is the weighted spread of the
    overlapping local solutions.
    """

    t: np.ndarray
    labels: np.ndarray
    q_estimate: np.ndarray
    v_estimate: np.ndarray
    tau: np.ndarray
    tau_std: np.ndarray
    tau_disagreement: np.ndarray
    q_replay: np.ndarray
    v_replay: np.ndarray
    normalized_residual: np.ndarray
    replay_residual: np.ndarray
    ztcf_divergence: np.ndarray
    explained_fraction: np.ndarray
    rejection_rate: np.ndarray
    drift_dominance: np.ndarray
    windows: tuple[WindowRecord, ...]


def _window_starts(num_steps: int, steps: int, stride: int) -> list[int]:
    """Starts covering ``[0, num_steps]``; the last window ends at the end."""
    last = max(num_steps - steps, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _taper(samples: int) -> np.ndarray:
    """Hann taper that keeps a small weight at the window edges."""
    j = np.arange(samples)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (j + 1) / (samples + 1))


class _Accumulator:
    """Weighted overlay of per-window quantities onto the sample axis."""

    def __init__(self, num: int, n: int, m: int) -> None:
        self.w_tau = np.zeros((num, m))
        self.tau = np.zeros((num, m))
        self.tau_sq = np.zeros((num, m))
        self.std = np.zeros((num, m))
        self.w = np.zeros(num)
        self.q = np.zeros((num, n))
        self.v = np.zeros((num, n))
        self.explained = np.zeros(num)
        self.w_obs = np.zeros(num)
        self.div = np.zeros(num)
        self.rejects = np.zeros(num)
        self.votes = np.zeros(num)

    def add(self, start: int, sol: LocalWindowSolution) -> None:
        span = slice(start, start + sol.q.shape[0])
        taper = _taper(sol.q.shape[0])
        inv_var = taper[:, None] / np.maximum(sol.tau_std, 1e-12) ** 2
        self.w_tau[span] += inv_var
        self.tau[span] += inv_var * sol.tau
        self.tau_sq[span] += inv_var * sol.tau**2
        self.std[span] += taper[:, None] * sol.tau_std
        self.w[span] += taper
        self.q[span] += taper[:, None] * sol.q
        self.v[span] += taper[:, None] * sol.v
        self.explained[span] += taper * sol.explained_fraction
        seen = sol.observed
        w_seen = np.where(seen, taper, 0.0)
        self.w_obs[span] += w_seen
        self.div[span] += np.where(
            seen, taper * np.nan_to_num(sol.ztcf_divergence), 0.0
        )
        self.votes[span] += seen.astype(float)
        self.rejects[span] += (seen & (sol.weights == 0.0)).astype(float)


def _initial_state(
    q_obs: np.ndarray, mask: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    require(
        bool(mask[0] and mask[1]),
        "the first two samples must be observed to form the initial state",
    )
    return q_obs[0].copy(), (q_obs[1] - q_obs[0]) / dt


def _label_samples(
    mask: np.ndarray, rejection: np.ndarray, options: MatchOptions
) -> np.ndarray:
    labels = np.full(mask.size, SampleLabel.ACCEPTED, dtype=object)
    labels[~mask] = SampleLabel.GAP_FILLED
    flagged = mask & (rejection >= options.reject_vote)
    observed_idx = np.flatnonzero(mask)
    run: list[int] = []
    for idx in [*observed_idx, -1]:
        if idx >= 0 and flagged[idx]:
            run.append(idx)
            continue
        if run:
            verdict = (
                SampleLabel.OUTLIER
                if len(run) <= options.max_outlier_run
                else SampleLabel.UNEXPLAINED
            )
            labels[run] = verdict
            run = []
    return labels


def match_kinematics(
    provider: DynamicsProvider,
    t: np.ndarray,
    q_observed: np.ndarray,
    mask: np.ndarray,
    band: ControlBand,
    options: MatchOptions,
    *,
    selection: np.ndarray | None = None,
) -> DriftAnchoredMatchResult:
    """Match a forward-dynamics model to observed kinematics.

    Args:
        provider: Contact-free dynamics provider in local coordinates.
        t: Uniformly spaced sample times ``(T,)`` [s].
        q_observed: Observed configurations ``(T, n)`` (NaN allowed where
            ``mask`` is False).
        mask: ``(T,)`` True where an observation exists. Samples 0 and 1 must
            be observed (initial state).
        band: Actuator box and torque-rate limit.
        options: Window and labelling settings.
        selection: Actuation map ``(n, m)``; identity when ``None``.

    Returns:
        :class:`DriftAnchoredMatchResult` with assembled torque, estimates,
        replay, per-sample labels and diagnostics.
    """
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    q_obs = np.asarray(q_observed, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    num = t_arr.size
    require(q_obs.ndim == 2 and q_obs.shape[0] == num, "q_observed must be (T, n)")
    require(mask_arr.shape == (num,), "mask must be (T,)")
    require(num > options.window_steps, "need more samples than one window")
    steps_dt = np.diff(t_arr)
    dt = float(steps_dt[0])
    require(dt > 0.0 and np.allclose(steps_dt, dt, rtol=1e-6), "t must be uniform")
    require(check_finite(q_obs[mask_arr]), "observed rows must be finite")
    n = q_obs.shape[1]
    m = n if selection is None else int(np.asarray(selection).shape[1])
    require(band.size == m, "band size must match actuation")

    q_state, v_state = _initial_state(q_obs, mask_arr, dt)
    tau_prev = (
        np.zeros(m)
        if options.initial_torque is None
        else np.asarray(options.initial_torque, dtype=float).reshape(m)
    )
    acc = _Accumulator(num, n, m)
    fallback_q = np.full((num, n), np.nan)
    fallback_v = np.full((num, n), np.nan)
    fallback_tau = np.full((num, m), np.nan)
    records: list[WindowRecord] = []
    replay_start: tuple[np.ndarray, np.ndarray] | None = None
    starts = _window_starts(num - 1, options.window_steps, options.stride)

    reacquire = False
    for i, start in enumerate(starts):
        steps = _window_length(mask_arr, start, options)
        span = slice(start, start + steps + 1)
        win_opts = options.window_options_for(steps, first=i == 0, reacquire=reacquire)
        sol = solve_local_window(
            provider,
            q_state,
            v_state,
            WindowObservation(q=q_obs[span], mask=mask_arr[span]),
            dt,
            band,
            tau_prev,
            win_opts,
            selection=selection,
        )
        records.append(_record(start, steps, sol))
        advance = (starts[i + 1] - start) if i + 1 < len(starts) else steps
        reacquire = not sol.success
        if sol.success:
            acc.add(start, sol)
            q_traj, v_traj, tau_traj = sol.q, sol.v, sol.tau
        else:  # see through: drift + the same torque as before
            hold = np.tile(tau_prev, (steps, 1))
            q_traj, v_traj = rollout(
                provider, q_state, v_state, hold, dt, selection=selection
            )
            tau_traj = np.vstack([hold, tau_prev])
            fallback_q[span], fallback_v[span] = q_traj, v_traj
            fallback_tau[span] = tau_traj
        if replay_start is None:
            replay_start = (q_traj[0].copy(), v_traj[0].copy())
        q_state, v_state = q_traj[advance].copy(), v_traj[advance].copy()
        tau_prev = tau_traj[max(advance - 1, 0)].copy()

    return _assemble(
        provider, t_arr, q_obs, mask_arr, band, options, acc,
        (fallback_q, fallback_v, fallback_tau), replay_start, records, selection,
    )  # fmt: skip


def _window_length(mask: np.ndarray, start: int, options: MatchOptions) -> int:
    """Nominal window steps, extended until it holds enough observations."""
    last = mask.size - 1
    steps = min(options.window_steps, last - start)
    cap = min(options.extension_cap, last - start)
    while (
        int(mask[start : start + steps + 1].sum()) < options.required_observed
        and steps < cap
    ):
        steps += 1
    return steps


def _record(start: int, steps: int, sol: LocalWindowSolution) -> WindowRecord:
    return WindowRecord(
        start=start,
        steps=steps,
        success=sol.success,
        status=sol.status,
        chi2_per_dof=sol.chi2_per_dof,
        raw_rms_residual=sol.raw_rms_residual,
        explained_fraction=sol.explained_fraction,
        saturated=sol.saturated.copy(),
        rate_limited=sol.rate_limited.copy(),
        torque_condition_number=sol.torque_condition_number,
    )


def _assemble(
    provider: DynamicsProvider,
    t: np.ndarray,
    q_obs: np.ndarray,
    mask: np.ndarray,
    band: ControlBand,
    options: MatchOptions,
    acc: _Accumulator,
    fallback: tuple[np.ndarray, np.ndarray, np.ndarray],
    replay_start: tuple[np.ndarray, np.ndarray] | None,
    records: list[WindowRecord],
    selection: np.ndarray | None,
) -> DriftAnchoredMatchResult:
    fb_q, fb_v, fb_tau = fallback
    dt = float(t[1] - t[0])
    covered = acc.w > 0.0
    w_col = np.maximum(acc.w, 1e-300)[:, None]
    q_est = np.where(covered[:, None], acc.q / w_col, fb_q)
    v_est = np.where(covered[:, None], acc.v / w_col, fb_v)
    has_tau = acc.w_tau > 0.0
    w_tau = np.maximum(acc.w_tau, 1e-300)
    tau = np.where(has_tau, acc.tau / w_tau, fb_tau)
    spread = np.where(has_tau, acc.tau_sq / w_tau - (acc.tau / w_tau) ** 2, 0.0)
    disagreement = np.sqrt(np.maximum(spread, 0.0))
    half_band = 0.5 * (band.upper - band.lower)
    tau_std = np.where(covered[:, None], acc.std / w_col, half_band[None, :])
    require(check_finite(tau) and check_finite(q_est), "assembled match must be finite")

    rejection = np.where(acc.votes > 0, acc.rejects / np.maximum(acc.votes, 1.0), 0.0)
    labels = _label_samples(mask, rejection, options)
    sigma = options.window.sigma_obs * np.sqrt(q_obs.shape[1])
    resid = np.full(t.size, np.nan)
    resid[mask] = np.linalg.norm(q_est[mask] - q_obs[mask], axis=1) / sigma

    assert replay_start is not None  # at least one window always runs
    q_rep, v_rep = rollout(provider, *replay_start, tau[:-1], dt, selection=selection)
    replay_resid = np.full(t.size, np.nan)
    replay_resid[mask] = np.linalg.norm(q_rep[mask] - q_obs[mask], axis=1) / sigma

    dominance = np.array(
        [
            drift_dominance_index(
                linearize_drift(provider, q, v, selection=selection),
                band.lower,
                band.upper,
            )
            for q, v in zip(q_est, v_est, strict=True)
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        divergence = np.where(acc.w_obs > 0, acc.div / acc.w_obs, np.nan)
        explained = np.where(covered, acc.explained / np.maximum(acc.w, 1e-300), np.nan)
    ensure(labels.shape == (t.size,), "one label per sample")
    return DriftAnchoredMatchResult(
        t=t,
        labels=labels,
        q_estimate=q_est,
        v_estimate=v_est,
        tau=tau,
        tau_std=tau_std,
        tau_disagreement=disagreement,
        q_replay=q_rep,
        v_replay=v_rep,
        normalized_residual=resid,
        replay_residual=replay_resid,
        ztcf_divergence=divergence,
        explained_fraction=explained,
        rejection_rate=rejection,
        drift_dominance=dominance,
        windows=tuple(records),
    )
