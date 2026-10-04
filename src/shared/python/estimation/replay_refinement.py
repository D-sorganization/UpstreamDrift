"""Whole-trajectory replay refinement for a drift-anchored match (DIME-09 slice).

The local-window overlay is locally consistent, but integrating its torque
noise open loop drifts away from the data. This step makes the final answer
**one uninterrupted forward-dynamics simulation**: single shooting over the
full horizon adjusts the initial state and a piecewise-linear torque profile
(knots every ``knot_stride`` steps) so the replay matches the ``accepted``
observations::

    min  sum_{j accepted} |(q_j(z) - y_j)/sigma|^2
         + |(c - c_overlay)/sigma_c|^2 + |dq0/sigma_q0|^2 + |dv0/sigma_v0|^2
    s.t. c in actuator box

The overlay is the prior (its posterior spread sets ``sigma_c``), outliers and
gaps are excluded via the matcher's labels, and the solve is Levenberg-damped
Gauss-Newton on forward-difference sensitivities. Each knot's sensitivity
rollout restarts from the cached base state where that knot first acts
(causality), roughly halving the cost.

The torque-rate limit is not enforced here (the prior keeps the profile near
the rate-limited overlay); contact and manifold limits are inherited from
:mod:`drift_prediction`. Simulator truth is not capture qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

from src.shared.python.core.contracts import check_finite, ensure, require
from src.shared.python.estimation.drift_anchored_matcher import (
    DriftAnchoredMatchResult,
    SampleLabel,
)
from src.shared.python.estimation.drift_prediction import ControlBand
from src.shared.python.estimation.local_torque_window import rollout

if TYPE_CHECKING:
    from src.shared.python.simulation_backends.protocol import DynamicsProvider

__all__ = ["ReplayOptions", "ReplayRefinement", "refine_match_replay"]


@dataclass(frozen=True)
class ReplayOptions:
    """Settings for the single-shooting replay refinement.

    Attributes:
        sigma_obs: Observation noise std per coordinate [rad].
        knot_stride: Steps between torque knots.
        prior_floor: Minimum prior std on a knot [N m].
        sigma_q0: Prior std on the initial configuration change [rad]
            (``None`` = ``5 * sigma_obs``).
        sigma_v0: Prior std on the initial velocity change [rad/s].
        max_iterations: Gauss-Newton iteration cap.
        fd_step: Relative finite-difference step.
        damping: Initial Levenberg damping (whitened units).
        knot_stride_candidates: When non-empty, ignore ``knot_stride`` and
            pick the coarsest candidate whose accepted-data residual is
            consistent with ``sigma_obs`` in **every** coordinate
            (discrepancy principle, mean squared normalised residual
            ``<= 1 + 3 sqrt(2/N)``). A pooled test would hide one
            under-fitted joint; finer knots than needed would fit noise.
    """

    sigma_obs: float
    knot_stride: int = 4
    prior_floor: float = 1.0
    sigma_q0: float | None = None
    sigma_v0: float = 1.0
    max_iterations: int = 6
    fd_step: float = 1e-6
    damping: float = 1e-3
    knot_stride_candidates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("sigma_obs", "prior_floor", "sigma_v0", "fd_step", "damping"):
            value = float(getattr(self, name))
            require(np.isfinite(value) and value > 0.0, f"{name} must be > 0", value)
        require(self.sigma_q0 is None or self.sigma_q0 > 0.0, "sigma_q0 must be > 0")
        require(self.knot_stride >= 1, "knot_stride must be >= 1")
        require(
            all(int(c) >= 1 for c in self.knot_stride_candidates),
            "knot_stride_candidates must be >= 1",
        )
        require(self.max_iterations >= 1, "max_iterations must be >= 1")

    @property
    def arrival_std_q(self) -> float:
        """Prior std on the initial configuration change [rad]."""
        return self.sigma_q0 or 5.0 * self.sigma_obs


@dataclass(frozen=True)
class ReplayRefinement:
    """One uninterrupted replay; ``tau[k]`` acts during step ``k``.

    ``observation_rms_*`` are RMS-per-coordinate residuals over ``used``
    samples in units of ``sigma_obs``.
    """

    success: bool
    status: str
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    used: np.ndarray
    observation_rms_before: float
    observation_rms_after: float
    iterations: int
    knot_stride: int


class _Shooting:
    """Single-shooting residual model (internal)."""

    def __init__(
        self,
        provider: DynamicsProvider,
        match: DriftAnchoredMatchResult,
        y: np.ndarray,
        used: np.ndarray,
        options: ReplayOptions,
        selection: np.ndarray | None,
        stride: int,
    ) -> None:
        steps = y.shape[0] - 1
        self.stride = stride
        self.positions = np.unique(np.r_[np.arange(0, steps, stride), steps - 1])
        idx = np.arange(steps, dtype=float)
        k = self.positions.size
        self.basis = np.column_stack(
            [np.interp(idx, self.positions, np.eye(k)[i]) for i in range(k)]
        )
        self.n, self.m = y.shape[1], match.tau.shape[1]
        self.q0, self.v0 = match.q_replay[0], match.v_replay[0]
        self.c_prior = match.tau[self.positions].reshape(-1)
        self.c_std = np.maximum(
            match.tau_std[self.positions], options.prior_floor
        ).reshape(-1)
        self._provider, self._y, self._used = provider, y, used
        self._opts, self._sel = options, selection
        self._dt = float(match.t[1] - match.t[0])

    @property
    def used(self) -> np.ndarray:
        """Samples the refinement fits (matcher label ``accepted``)."""
        return self._used

    @property
    def size(self) -> int:
        return 2 * self.n + self.c_prior.size

    def torques(self, z: np.ndarray) -> np.ndarray:
        return self.basis @ z[2 * self.n :].reshape(-1, self.m)

    def simulate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        return rollout(
            self._provider,
            self.q0 + z[:n],
            self.v0 + z[n : 2 * n],
            self.torques(z),
            self._dt,
            selection=self._sel,
        )

    def residual(self, z: np.ndarray, q: np.ndarray) -> np.ndarray:
        o, n = self._opts, self.n
        r_obs = ((q[self._used] - self._y[self._used]) / o.sigma_obs).reshape(-1)
        r_c = (z[2 * n :] - self.c_prior) / self.c_std
        r_x = np.r_[z[:n] / o.arrival_std_q, z[n : 2 * n] / o.sigma_v0]
        return np.concatenate([r_obs, r_c, r_x])

    def jacobian(self, z: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Forward differences; knot columns restart where the knot first acts."""
        o, n, m = self._opts, self.n, self.m
        sens = np.zeros((*q.shape, z.size))
        tau = self.torques(z)
        for i in range(2 * n):
            zp = z.copy()
            h = o.fd_step * max(1.0, abs(z[i]))
            zp[i] += h
            sens[:, :, i] = (self.simulate(zp)[0] - q) / h
        for k, _pos in enumerate(self.positions):
            first = 0 if k == 0 else int(self.positions[k - 1]) + 1
            for ch in range(m):
                col = 2 * n + k * m + ch
                h = o.fd_step * max(1.0, abs(z[col]))
                tau_p = tau[first:].copy()
                tau_p[:, ch] += h * self.basis[first:, k]
                q_p, _ = rollout(
                    self._provider,
                    q[first],
                    v[first],
                    tau_p,
                    self._dt,
                    selection=self._sel,
                )
                sens[first:, :, col] = (q_p - q[first:]) / h
        j_obs = (sens[self._used] / o.sigma_obs).reshape(-1, z.size)
        j_c = np.zeros((self.c_prior.size, z.size))
        j_c[:, 2 * n :] = np.diag(1.0 / self.c_std)
        j_x = np.zeros((2 * n, z.size))
        j_x[:, : 2 * n] = np.diag(
            np.r_[np.full(n, 1.0 / o.arrival_std_q), np.full(n, 1.0 / o.sigma_v0)]
        )
        return np.vstack([j_obs, j_c, j_x])

    def observation_rms(self, q: np.ndarray) -> float:
        diff = (q[self._used] - self._y[self._used]) / self._opts.sigma_obs
        return float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan")


def refine_match_replay(
    provider: DynamicsProvider,
    match: DriftAnchoredMatchResult,
    q_observed: np.ndarray,
    band: ControlBand,
    options: ReplayOptions,
    *,
    selection: np.ndarray | None = None,
) -> ReplayRefinement:
    """Refine the match into one uninterrupted replay of the accepted data.

    Args:
        provider: The dynamics provider used for the match.
        match: Output of :func:`match_kinematics`.
        q_observed: The observations given to the matcher ``(T, n)``.
        band: Actuator box (rate limit not enforced here).
        options: Refinement settings.
        selection: Actuation map, as for the match.

    Returns:
        :class:`ReplayRefinement`; ``success=False`` when fewer than two
        samples are accepted (the overlay replay is returned unchanged).
        With candidates, the finest candidate's result is returned when none
        meets the discrepancy bound (callers should inspect ``knot_stride``
        and ``observation_rms_after``).
    """
    y = np.asarray(q_observed, dtype=float)
    require(y.shape == match.q_estimate.shape, "q_observed must match the match shape")
    require(band.size == match.tau.shape[1], "band size must match the torque width")
    used = np.asarray(match.labels == SampleLabel.ACCEPTED, dtype=bool)
    require(check_finite(y[used]), "accepted observations must be finite")
    if not options.knot_stride_candidates:
        return _refine(
            provider, match, y, used, band, options, selection, options.knot_stride
        )
    threshold = 1.0 + 3.0 * np.sqrt(2.0 / max(int(used.sum()), 1))
    result: ReplayRefinement | None = None
    for stride in sorted(
        {int(c) for c in options.knot_stride_candidates}, reverse=True
    ):
        result = _refine(provider, match, y, used, band, options, selection, stride)
        per_coord = np.mean(
            ((result.q[used] - y[used]) / options.sigma_obs) ** 2, axis=0
        )
        if result.success and bool(np.all(per_coord <= threshold)):
            return result
    assert result is not None  # candidates are non-empty
    return result


def _refine(
    provider: DynamicsProvider,
    match: DriftAnchoredMatchResult,
    y: np.ndarray,
    used: np.ndarray,
    band: ControlBand,
    options: ReplayOptions,
    selection: np.ndarray | None,
    stride: int,
) -> ReplayRefinement:
    """Single-shooting refinement with knots every ``stride`` steps."""
    model = _Shooting(provider, match, y, used, options, selection, stride)
    z = np.r_[np.zeros(2 * model.n), np.clip(model.c_prior, *_box(band, model))]
    lb = np.r_[np.full(2 * model.n, -np.inf), _box(band, model)[0]]
    ub = np.r_[np.full(2 * model.n, np.inf), _box(band, model)[1]]
    q, v = model.simulate(z)
    rms_before = model.observation_rms(q)
    if int(used.sum()) < 2:
        return _result(
            model, z, (q, v), rms_before, False, "too few accepted samples", 0
        )

    r = model.residual(z, q)
    cost, damping, status, iterations = (
        float(r @ r),
        options.damping,
        "iteration cap",
        0,
    )
    while iterations < options.max_iterations:
        iterations += 1
        jac = model.jacobian(z, q, v)
        while True:
            aug = np.vstack([jac, np.sqrt(damping) * np.eye(z.size)])
            rhs = np.r_[-r, np.zeros(z.size)]
            step = lsq_linear(aug, rhs, bounds=(lb - z, ub - z), method="bvls").x
            z_try = np.clip(z + step, lb, ub)
            q_try, v_try = model.simulate(z_try)
            r_try = model.residual(z_try, q_try) if check_finite(q_try) else None
            if r_try is not None and float(r_try @ r_try) < cost:
                break
            damping *= 10.0
            if damping > 1e8:
                return _result(
                    model, z, (q, v), rms_before, True, "no further descent", iterations
                )
        improvement = (cost - float(r_try @ r_try)) / max(cost, 1e-300)
        z, q, v, r, cost = z_try, q_try, v_try, r_try, float(r_try @ r_try)
        damping = max(damping / 10.0, 1e-9)
        if improvement < 1e-6:
            status = "converged"
            break
    ensure(check_finite(q) and check_finite(model.torques(z)), "replay must be finite")
    return _result(model, z, (q, v), rms_before, True, status, iterations)


def _box(band: ControlBand, model: _Shooting) -> tuple[np.ndarray, np.ndarray]:
    k = model.positions.size
    return np.tile(band.lower, k), np.tile(band.upper, k)


def _result(
    model: _Shooting,
    z: np.ndarray,
    trajectory: tuple[np.ndarray, np.ndarray],
    rms_before: float,
    success: bool,
    status: str,
    iterations: int,
) -> ReplayRefinement:
    q, v = trajectory
    tau_steps = model.torques(z)
    return ReplayRefinement(
        success=success,
        status=status,
        q=q,
        v=v,
        tau=np.vstack([tau_steps, tau_steps[-1]]),
        used=model.used.copy(),
        observation_rms_before=rms_before,
        observation_rms_after=model.observation_rms(q),
        iterations=iterations,
        knot_stride=model.stride,
    )
