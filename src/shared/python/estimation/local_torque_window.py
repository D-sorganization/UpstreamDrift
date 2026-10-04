"""ZTCF-anchored local torque-band window solver (DIME-04/05/16, epic #11421).

One short window of observed configurations is explained by forward dynamics
from a (corrected) start state under a tight band of admissible torques:

1. **Anchor.** Start from "ZTCF + same torque as before": the previous torque
   held constant (clipped into the band) and integrated with the full drift.
2. **Viable range.** Linearise positions in the torque knots. Every admissible
   torque maps to a per-sample position interval (the plausible band). An
   observation outside ``band +/- k sigma`` cannot be produced by reasonable
   inputs; it is gated out before fitting (ruled out, not averaged in).
3. **Local inverse problem.** Gauss-Newton over ``z = [dq0, dv0, c]`` with the
   torque knots ``c`` box/rate bounded (``scipy.optimize.lsq_linear``), an
   arrival prior on the start-state correction and robust (Huber +
   hard-reject) observation weights.
4. **Diagnostics.** Divergence of the data from the pure ZTCF branch, the share
   of that divergence the bounded control explains, saturated knots, posterior
   torque spread and chi-square, so callers can tell noise and occlusion apart
   from dynamics the model cannot reach.

The band is a first-order (linearised) envelope around the fitted solution,
plus the posterior start-state spread; it is not a certified reachable set.
Contact and manifold coordinates are out of scope (see ``drift_prediction``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

from src.shared.python.core.contracts import check_finite, ensure, require
from src.shared.python.estimation.drift_prediction import ControlBand, integrate_step

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.protocol import DynamicsProvider

#: Relative Gauss-Newton step tolerance, consistent with FD sensitivity error.
_STEP_TOL = 1e-6

__all__ = [
    "LocalWindowSolution",
    "WindowObservation",
    "WindowOptions",
    "rollout",
    "solve_local_window",
]


@dataclass(frozen=True)
class WindowOptions:
    """Noise model and robustness settings for one window solve.

    Attributes:
        sigma_obs: Observation noise std per coordinate [rad].
        sigma_q0: Arrival-prior std on the start configuration [rad].
        sigma_v0: Arrival-prior std on the start velocity [rad/s].
        continuity_sigma: Soft std [N m] tying the first knot to the previous
            torque; ``None`` relies on the hard rate limit only.
        n_knots: Torque knots per window (1 = constant, 2 = linear ramp, ...).
        huber_k: Normalised residual beyond which weights decay as ``k/d``.
        reject_threshold: Normalised residual beyond which a sample is dropped.
        band_sigma_multiple: Noise allowance (in ``sigma_obs``) added to the
            plausible band before a sample counts as unexplainable.
        max_iterations: Gauss-Newton iteration cap.
        fd_step: Relative finite-difference step for the sensitivities.
    """

    sigma_obs: float
    sigma_q0: float
    sigma_v0: float
    continuity_sigma: float | None = None
    n_knots: int = 1
    huber_k: float = 2.0
    reject_threshold: float = 6.0
    band_sigma_multiple: float = 4.0
    max_iterations: int = 8
    fd_step: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("sigma_obs", "sigma_q0", "sigma_v0", "fd_step"):
            value = float(getattr(self, name))
            require(np.isfinite(value) and value > 0.0, f"{name} must be > 0", value)
        if self.continuity_sigma is not None:
            require(self.continuity_sigma > 0.0, "continuity_sigma must be > 0")
        require(self.n_knots >= 1, "n_knots must be at least 1", self.n_knots)
        require(0.0 < self.huber_k <= self.reject_threshold, "need 0<huber_k<=reject")
        require(self.band_sigma_multiple >= 0.0, "band_sigma_multiple must be >= 0")
        require(self.max_iterations >= 1, "max_iterations must be >= 1")


@dataclass(frozen=True)
class WindowObservation:
    """Observed configurations ``q`` (S, n) sampled every ``dt`` [s].

    Rows with ``mask=False`` are missing.
    """

    q: np.ndarray
    mask: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        mask = np.asarray(self.mask, dtype=bool).reshape(-1)
        require(q.ndim == 2 and q.shape[1] >= 1, "q must be 2-D (S, n)", q.shape)
        require(q.shape[0] >= 2, "a window needs at least two samples", q.shape)
        require(mask.shape == (q.shape[0],), "mask must be (S,)", mask.shape)
        require(check_finite(q[mask]), "observed rows must be finite")
        dt = float(self.dt)
        require(np.isfinite(dt) and dt > 0.0, "dt must be positive and finite", dt)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "mask", mask)

    @property
    def num_samples(self) -> int:
        """Samples in the window (steps + 1)."""
        return int(self.q.shape[0])


@dataclass(frozen=True)
class LocalWindowSolution:
    """Fitted window: trajectories, torque band use and data-quality signals.

    All per-sample arrays have ``S = steps + 1`` rows; ``tau[j]`` acts during
    step ``j`` (the last row repeats the final step). Normalised quantities are
    RMS-per-coordinate in units of ``sigma_obs`` and NaN where unobserved.
    ``chi2_per_dof`` uses the robust weights (inliers); ``raw_rms_residual``
    ignores them, so a window whose samples were rejected because no
    admissible torque reaches them still reports the failure. ``saturated``
    marks channels pinned at the global actuator box (inputs unreasonable);
    ``rate_limited`` marks channels pinned by the continuity rate limit only.
    """

    success: bool
    status: str
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    tau_std: np.ndarray
    q_ztcf: np.ndarray
    q_constant_torque: np.ndarray
    band_lower: np.ndarray
    band_upper: np.ndarray
    observed: np.ndarray
    weights: np.ndarray
    inside_band: np.ndarray
    normalized_residual: np.ndarray
    ztcf_divergence: np.ndarray
    constant_torque_divergence: np.ndarray
    explained_fraction: float
    saturated: np.ndarray
    rate_limited: np.ndarray
    chi2_per_dof: float
    raw_rms_residual: float
    torque_condition_number: float
    iterations: int


def rollout(
    provider: DynamicsProvider,
    q0: np.ndarray,
    v0: np.ndarray,
    tau_steps: np.ndarray,
    dt: float,
    *,
    selection: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate RK4 steps under ``tau_steps`` (steps, m); returns (steps+1, n)."""
    taus = np.asarray(tau_steps, dtype=float)
    require(taus.ndim == 2 and taus.shape[0] >= 1, "tau_steps must be (steps, m)")
    qs = [np.asarray(q0, dtype=float).reshape(-1)]
    vs = [np.asarray(v0, dtype=float).reshape(-1)]
    for tau in taus:
        qn, vn = integrate_step(provider, qs[-1], vs[-1], tau, dt, selection=selection)
        qs.append(qn)
        vs.append(vn)
    return np.array(qs), np.array(vs)


def _knot_basis(steps: int, n_knots: int) -> tuple[np.ndarray, np.ndarray]:
    """Hat-function basis (steps, K) and knot step positions (K,)."""
    if n_knots == 1:
        return np.ones((steps, 1)), np.zeros(1)
    positions = np.linspace(0.0, steps - 1.0, n_knots)
    idx = np.arange(steps, dtype=float)
    basis = np.column_stack(
        [np.interp(idx, positions, np.eye(n_knots)[k]) for k in range(n_knots)]
    )
    return basis, positions


@dataclass(frozen=True)
class _Layout:
    """Decision-vector layout ``z = [dq0 (n), dv0 (n), c (K*m)]``."""

    n: int
    m: int
    basis: np.ndarray

    @property
    def size(self) -> int:
        return 2 * self.n + self.basis.shape[1] * self.m

    def torques(self, z: np.ndarray) -> np.ndarray:
        knots = z[2 * self.n :].reshape(self.basis.shape[1], self.m)
        return self.basis @ knots

    def state(self, z: np.ndarray, q0: np.ndarray, v0: np.ndarray) -> tuple:
        return q0 + z[: self.n], v0 + z[self.n : 2 * self.n]


class _WindowModel:
    """Residuals and sensitivities of one window (internal; LoD boundary)."""

    def __init__(
        self,
        provider: DynamicsProvider,
        q0: np.ndarray,
        v0: np.ndarray,
        obs: WindowObservation,
        tau_prev: np.ndarray,
        options: WindowOptions,
        selection: np.ndarray | None,
        m: int,
    ) -> None:
        steps = obs.num_samples - 1
        basis, positions = _knot_basis(steps, options.n_knots)
        self.layout = _Layout(n=q0.size, m=m, basis=basis)
        self.positions = positions
        self._provider, self._q0, self._v0 = provider, q0, v0
        self._obs, self._dt, self._opts = obs, obs.dt, options
        self._tau_prev, self._sel = tau_prev, selection

    @property
    def observation(self) -> WindowObservation:
        """The window's observations."""
        return self._obs

    @property
    def options(self) -> WindowOptions:
        """The window's solve options."""
        return self._opts

    def trajectory(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q0, v0 = self.layout.state(z, self._q0, self._v0)
        return rollout(
            self._provider,
            q0,
            v0,
            self.layout.torques(z),
            self._dt,
            selection=self._sel,
        )

    def sensitivity(self, z: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        """Forward-difference ``d q_j / d z``, shape (S, n, len(z))."""
        jac = np.empty((*q_ref.shape, z.size))
        for i in range(z.size):
            h = self._opts.fd_step * max(1.0, abs(z[i]))
            zp = z.copy()
            zp[i] += h
            jac[:, :, i] = (self.trajectory(zp)[0] - q_ref) / h
        return jac

    def residual_system(
        self, z: np.ndarray, q: np.ndarray, sens: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Whitened, weighted residual ``r`` and Jacobian ``J`` at ``z``."""
        o, n = self._opts, self.layout.n
        rows = np.flatnonzero(self._obs.mask & (weights > 0.0))
        sw = np.sqrt(weights[rows])[:, None]
        r_obs = (sw * (q[rows] - self._obs.q[rows]) / o.sigma_obs).reshape(-1)
        j_obs = (sw[:, :, None] * sens[rows] / o.sigma_obs).reshape(-1, z.size)
        prior_scale = np.r_[np.full(n, o.sigma_q0), np.full(n, o.sigma_v0)]
        r_prior = z[: 2 * n] / prior_scale
        j_prior = np.zeros((2 * n, z.size))
        j_prior[:, : 2 * n] = np.diag(1.0 / prior_scale)
        r_parts, j_parts = [r_obs, r_prior], [j_obs, j_prior]
        if o.continuity_sigma is not None:
            tau0 = self.layout.torques(z)[0]
            j_cont = np.zeros((self.layout.m, z.size))
            row0 = self.layout.basis[0]
            for k, coeff in enumerate(row0):
                start = 2 * n + k * self.layout.m
                j_cont[:, start : start + self.layout.m] = np.eye(self.layout.m) * coeff
            r_parts.append((tau0 - self._tau_prev) / o.continuity_sigma)
            j_parts.append(j_cont / o.continuity_sigma)
        return np.concatenate(r_parts), np.vstack(j_parts)

    def normalized_residual(self, q: np.ndarray) -> np.ndarray:
        """RMS-per-coordinate residual in sigma units; NaN where unobserved."""
        out = np.full(q.shape[0], np.nan)
        mask = self._obs.mask
        diff = q[mask] - self._obs.q[mask]
        out[mask] = np.linalg.norm(diff, axis=1) / (
            self._opts.sigma_obs * np.sqrt(self.layout.n)
        )
        return out

    def divergence(self, q_branch: np.ndarray) -> np.ndarray:
        """Observed-minus-branch distance in sigma units (NaN where missing)."""
        return self.normalized_residual(q_branch)


def _robust_weights(d: np.ndarray, options: WindowOptions) -> np.ndarray:
    """Huber weights with a hard reject; 0 for missing samples (NaN)."""
    w = np.zeros_like(d)
    ok = np.isfinite(d)
    dd = d[ok]
    w_ok = np.where(
        dd <= options.huber_k, 1.0, options.huber_k / np.maximum(dd, 1e-300)
    )
    w_ok[dd > options.reject_threshold] = 0.0
    w[ok] = w_ok
    return w


def _knot_bounds(
    band: ControlBand, tau_prev: np.ndarray, positions: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-knot box bounds; rate limit measured from the previous torque."""
    lows, highs = [], []
    for p in positions:
        lo, hi = band.local(tau_prev, dt * (p + 1.0))
        lows.append(lo)
        highs.append(hi)
    return np.concatenate(lows), np.concatenate(highs)


def _band(
    q: np.ndarray,
    sens: np.ndarray,
    n_state: int,
    knot_half: np.ndarray,
    state_std: np.ndarray,
    allowance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearised plausible position band around ``q`` (S, n)."""
    s_state = np.abs(sens[:, :, :n_state])
    s_tau = np.abs(sens[:, :, n_state:])
    half = s_tau @ knot_half + s_state @ state_std + allowance
    return q - half, q + half


def solve_local_window(
    provider: DynamicsProvider,
    q0: np.ndarray,
    v0: np.ndarray,
    observation: WindowObservation,
    band: ControlBand,
    tau_prev: np.ndarray,
    options: WindowOptions,
    *,
    selection: np.ndarray | None = None,
) -> LocalWindowSolution:
    """Explain one window of kinematics with bounded torques from a ZTCF anchor.

    Args:
        provider: Dynamics provider (contact-free, local coordinates).
        q0: Start-configuration estimate ``(n,)`` aligned with sample 0.
        v0: Start-velocity estimate ``(n,)``.
        observation: Window observations ``(S, n)`` with mask and sample
            period.
        band: Global torque box and optional rate limit.
        tau_prev: Torque acting just before the window (warm start + rate
            anchor) ``(m,)``.
        options: Noise model and robustness settings.
        selection: Actuation map ``S^T`` ``(n, m)``; identity when ``None``.

    Returns:
        :class:`LocalWindowSolution`. Data-dependent failures (too few observed
        samples, non-finite rollout) return ``success=False`` with a status;
        invalid arguments raise ``ValueError``.
    """
    q0_arr = np.asarray(q0, dtype=float).reshape(-1)
    v0_arr = np.asarray(v0, dtype=float).reshape(-1)
    prev = np.asarray(tau_prev, dtype=float).reshape(-1)
    n = q0_arr.size
    require(observation.q.shape[1] == n, "observation width must match q0", n)
    require(v0_arr.shape == (n,), "v0 must match q0", v0_arr.shape)
    dt = observation.dt
    m = n if selection is None else int(np.asarray(selection).shape[1])
    require(band.size == m, f"band size {band.size} must match actuation {m}")
    require(prev.shape == (m,), "tau_prev must match actuation size", prev.shape)

    model = _WindowModel(
        provider, q0_arr, v0_arr, observation, prev, options, selection, m
    )
    layout = model.layout
    lb_c, ub_c = _knot_bounds(band, prev, model.positions, dt)
    lb = np.r_[np.full(2 * n, -np.inf), lb_c]
    ub = np.r_[np.full(2 * n, np.inf), ub_c]
    z = np.r_[np.zeros(2 * n), np.clip(np.tile(prev, len(model.positions)), lb_c, ub_c)]

    q_anchor, _ = model.trajectory(z)
    q_const = q_anchor.copy()
    sens = model.sensitivity(z, q_anchor)
    prior_std = np.r_[np.full(n, options.sigma_q0), np.full(n, options.sigma_v0)]
    knot_mid, knot_half = 0.5 * (lb_c + ub_c), 0.5 * (ub_c - lb_c)
    allowance = options.band_sigma_multiple * options.sigma_obs
    # Gate: centre the band on the mid-band torque, then rule out the rest.
    centre = q_anchor + sens[:, :, 2 * n :] @ (knot_mid - z[2 * n :])
    lo_g, hi_g = _band(
        centre,
        sens,
        2 * n,
        knot_half,
        options.band_sigma_multiple * prior_std,
        allowance,
    )
    obs_q = observation.q
    gate = observation.mask & np.all((obs_q >= lo_g) & (obs_q <= hi_g), axis=1)
    weights = gate.astype(float)

    q = q_anchor
    iterations = 0
    while iterations < options.max_iterations:
        iterations += 1
        r, jac = model.residual_system(z, q, sens, weights)
        dof = r.size - z.size
        if dof <= 0:
            return _failure(model, q, z, "insufficient observed samples")
        step = lsq_linear(jac, -r, bounds=(lb - z, ub - z), method="bvls").x
        z = np.clip(z + step, lb, ub)
        q, _ = model.trajectory(z)
        if not check_finite(q):
            return _failure(model, q_anchor, z, "non-finite rollout")
        sens = model.sensitivity(z, q)
        new_weights = _robust_weights(model.normalized_residual(q), options)
        # Finite-difference sensitivities limit attainable precision.
        converged = bool(np.all(np.abs(step) <= _STEP_TOL * (1.0 + np.abs(z))))
        if converged and np.array_equal(new_weights > 0, weights > 0):
            weights = new_weights
            break
        weights = new_weights

    fit = _FitState(z=z, q=q, sens=sens, weights=weights, iterations=iterations)
    return _assemble(model, fit, q_const, (lb_c, ub_c), band)


@dataclass(frozen=True)
class _FitState:
    """Final Gauss-Newton iterate handed to :func:`_assemble`."""

    z: np.ndarray
    q: np.ndarray
    sens: np.ndarray
    weights: np.ndarray
    iterations: int


def _posterior_cov(jac: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(jac.T @ jac)


def _assemble(
    model: _WindowModel,
    fit: _FitState,
    q_const: np.ndarray,
    knot_bounds: tuple[np.ndarray, np.ndarray],
    band: ControlBand,
) -> LocalWindowSolution:
    lb_c, ub_c = knot_bounds
    z, q, sens, weights = fit.z, fit.q, fit.sens, fit.weights
    iterations = fit.iterations
    obs = model.observation
    layout, opts = model.layout, model.options
    n, m, steps = layout.n, layout.m, obs.num_samples - 1
    r, jac = model.residual_system(z, q, sens, weights)
    dof = max(r.size - z.size, 1)
    cov = _posterior_cov(jac)
    q_fit, v_fit = model.trajectory(z)
    zero_z = z.copy()
    zero_z[2 * n :] = 0.0
    q_ztcf, _ = model.trajectory(zero_z)

    tau_steps = layout.torques(z)
    tau = np.vstack([tau_steps, tau_steps[-1]])
    knot_cov = cov[2 * n :, 2 * n :]
    var_steps = np.array(
        [
            np.diag(
                np.kron(b[None, :], np.eye(m))
                @ knot_cov
                @ np.kron(b[:, None], np.eye(m))
            )
            for b in layout.basis
        ]
    )
    tau_std = np.sqrt(np.maximum(np.vstack([var_steps, var_steps[-1]]), 0.0))

    state_std = np.sqrt(np.maximum(np.diag(cov)[: 2 * n], 0.0))
    knot_half = 0.5 * (ub_c - lb_c)
    knot_mid = 0.5 * (lb_c + ub_c)
    centre = q_fit + sens[:, :, 2 * n :] @ (knot_mid - z[2 * n :])
    band_lo, band_hi = _band(
        centre,
        sens,
        2 * n,
        knot_half,
        opts.band_sigma_multiple * state_std,
        opts.band_sigma_multiple * opts.sigma_obs,
    )
    inside = obs.mask & np.all((obs.q >= band_lo) & (obs.q <= band_hi), axis=1)

    resid = model.normalized_residual(q_fit)
    div = model.divergence(q_ztcf)
    inliers = obs.mask & (weights > 0.0)
    num = float(np.nansum(weights[inliers] * resid[inliers] ** 2))
    den = float(np.nansum(weights[inliers] * div[inliers] ** 2))
    explained = 1.0 if den <= 1e-300 else float(np.clip(1.0 - num / den, 0.0, 1.0))

    knots = z[2 * n :]
    width = ub_c - lb_c
    tol = 1e-9 * np.maximum(1.0, np.abs(knots))
    at_lo, at_hi = knots - lb_c <= tol, ub_c - knots <= tol
    box_lo = np.tile(band.lower, layout.basis.shape[1])
    box_hi = np.tile(band.upper, layout.basis.shape[1])
    on_box = (at_lo & (np.abs(lb_c - box_lo) <= tol)) | (
        at_hi & (np.abs(ub_c - box_hi) <= tol)
    )
    active = (width > 0.0) & (at_lo | at_hi)
    j_tau = jac[:, 2 * n :]
    ensure(tau.shape == (steps + 1, m), "tau must be (S, m)")
    return LocalWindowSolution(
        success=True,
        status="converged" if iterations < opts.max_iterations else "iteration cap",
        q=q_fit,
        v=v_fit,
        tau=tau,
        tau_std=tau_std,
        q_ztcf=q_ztcf,
        q_constant_torque=q_const,
        band_lower=band_lo,
        band_upper=band_hi,
        observed=obs.mask.copy(),
        weights=weights,
        inside_band=inside,
        normalized_residual=resid,
        ztcf_divergence=div,
        constant_torque_divergence=model.divergence(q_const),
        explained_fraction=explained,
        saturated=(active & on_box).reshape(-1, m).any(axis=0),
        rate_limited=(active & ~on_box).reshape(-1, m).any(axis=0),
        chi2_per_dof=float(r @ r) / dof,
        raw_rms_residual=float(np.sqrt(np.nanmean(resid[obs.mask] ** 2))),
        torque_condition_number=float(np.linalg.cond(j_tau)) if j_tau.size else np.inf,
        iterations=iterations,
    )


def _failure(
    model: _WindowModel,
    q: np.ndarray,
    z: np.ndarray,
    status: str,
) -> LocalWindowSolution:
    obs = model.observation
    layout = model.layout
    s, n, m = obs.num_samples, layout.n, layout.m
    nan_sn = np.full((s, n), np.nan)
    tau_steps = layout.torques(z)
    return LocalWindowSolution(
        success=False,
        status=status,
        q=q,
        v=nan_sn.copy(),
        tau=np.vstack([tau_steps, tau_steps[-1]]),
        tau_std=np.full((s, m), np.inf),
        q_ztcf=nan_sn.copy(),
        q_constant_torque=nan_sn.copy(),
        band_lower=nan_sn.copy(),
        band_upper=nan_sn.copy(),
        observed=obs.mask.copy(),
        weights=np.zeros(s),
        inside_band=np.zeros(s, bool),
        normalized_residual=np.full(s, np.nan),
        ztcf_divergence=np.full(s, np.nan),
        constant_torque_divergence=np.full(s, np.nan),
        explained_fraction=0.0,
        saturated=np.zeros(m, bool),
        rate_limited=np.zeros(m, bool),
        chi2_per_dof=np.inf,
        raw_rms_residual=np.inf,
        torque_condition_number=np.inf,
        iterations=0,
    )
