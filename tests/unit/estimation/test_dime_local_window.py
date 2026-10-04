"""Behavioural tests for the ZTCF-anchored local torque-band window solver.

Each window starts from the "ZTCF + same torque as before" prediction, solves
for the tight, rate-limited torque band that best explains the observed
kinematics, rules out samples outside the viable ZTCF+control range, and
reports how much of the divergence from the drift prediction is explained.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.drift_prediction import ControlBand, integrate_step
from src.shared.python.estimation.local_torque_window import (
    WindowObservation,
    WindowOptions,
    rollout,
    solve_local_window,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

pytestmark = pytest.mark.unit

DT = 0.002
W = 12  # steps per window -> W + 1 samples
Q0 = np.array([-1.2, -1.0])
V0 = np.array([6.0, 4.0])
WIDE = ControlBand(lower=np.full(2, -200.0), upper=np.full(2, 200.0))


@pytest.fixture(scope="module")
def pendulum():  # type: ignore[no-untyped-def]
    return make_backend("ode", GolfModelParams.default())


def _truth(provider, tau_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    q, v = [Q0.copy()], [V0.copy()]
    for tau in tau_rows:
        qn, vn = integrate_step(provider, q[-1], v[-1], tau, DT)
        q.append(qn)
        v.append(vn)
    return np.array(q), np.array(v)


def _obs(q: np.ndarray, mask: np.ndarray | None = None) -> WindowObservation:
    mask = np.ones(q.shape[0], bool) if mask is None else mask
    qq = q.copy()
    qq[~mask] = np.nan
    return WindowObservation(q=qq, mask=mask)


OPTS = WindowOptions(sigma_obs=1e-3, sigma_q0=1e-3, sigma_v0=0.5)


class TestRollout:
    def test_rollout_matches_repeated_integrate_step(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        tau = np.tile([20.0, 3.0], (W, 1))
        q_ref, v_ref = _truth(pendulum, tau)
        q, v = rollout(pendulum, Q0, V0, tau, DT)
        np.testing.assert_allclose(q, q_ref, atol=1e-12)
        np.testing.assert_allclose(v, v_ref, atol=1e-12)


class TestRecovery:
    def test_noise_free_constant_torque_is_recovered(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        tau_true = np.array([40.0, -6.0])
        q, _ = _truth(pendulum, np.tile(tau_true, (W, 1)))
        sol = solve_local_window(pendulum, Q0, V0, _obs(q), DT, WIDE, np.zeros(2), OPTS)
        assert sol.success, sol.status
        np.testing.assert_allclose(sol.tau[0], tau_true, rtol=1e-3, atol=0.05)
        assert np.nanmax(sol.normalized_residual) < 1e-2

    def test_linear_torque_ramp_is_recovered_with_two_knots(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        ramp = np.linspace([10.0, 0.0], [50.0, -10.0], W)
        q, _ = _truth(pendulum, ramp)
        opts = WindowOptions(sigma_obs=1e-3, sigma_q0=1e-3, sigma_v0=0.5, n_knots=2)
        sol = solve_local_window(pendulum, Q0, V0, _obs(q), DT, WIDE, ramp[0], opts)
        assert sol.success, sol.status
        np.testing.assert_allclose(sol.tau[:W], ramp, atol=1.5)

    def test_biased_torque_warm_start_cannot_force_passive_motion(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        """A wrong prior torque (here: zero = pure ZTCF) must not win."""
        tau_true = np.array([60.0, 8.0])
        q, _ = _truth(pendulum, np.tile(tau_true, (W, 1)))
        sol = solve_local_window(
            pendulum, Q0, V0, _obs(q), DT, WIDE, np.array([-60.0, -8.0]), OPTS
        )
        np.testing.assert_allclose(sol.tau[0], tau_true, rtol=0.02, atol=0.5)
        # ...the observed motion really does diverge from the ZTCF branch, and
        # the bounded control explains essentially all of that divergence.
        assert np.max(sol.ztcf_divergence) > 5.0
        assert np.nanmax(sol.normalized_residual) < 1e-2

    def test_missing_samples_are_ignored(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        tau_true = np.array([30.0, 5.0])
        q, _ = _truth(pendulum, np.tile(tau_true, (W, 1)))
        mask = np.ones(W + 1, bool)
        mask[4:9] = False
        sol = solve_local_window(
            pendulum, Q0, V0, _obs(q, mask), DT, WIDE, np.zeros(2), OPTS
        )
        assert sol.success, sol.status
        np.testing.assert_allclose(sol.tau[0], tau_true, rtol=1e-2, atol=0.2)
        assert np.all(np.isnan(sol.normalized_residual[~mask]))
        # Gap samples are filled by the dynamically consistent fit.
        np.testing.assert_allclose(sol.q[4:9], q[4:9], atol=1e-4)


class TestRobustness:
    def test_spike_outlier_is_rejected_and_does_not_bias_torque(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        tau_true = np.array([35.0, -4.0])
        q, _ = _truth(pendulum, np.tile(tau_true, (W, 1)))
        rng = np.random.default_rng(3)
        noisy = q + rng.normal(0.0, 1e-3, q.shape)
        clean = solve_local_window(
            pendulum, Q0, V0, _obs(noisy), DT, WIDE, np.zeros(2), OPTS
        )
        noisy[7] += np.array([0.15, -0.12])  # gross marker swap / occluder
        sol = solve_local_window(
            pendulum, Q0, V0, _obs(noisy), DT, WIDE, np.zeros(2), OPTS
        )
        assert sol.weights[7] == 0.0
        assert not sol.inside_band[7]
        assert np.mean(sol.weights[np.arange(W + 1) != 7] > 0.5) > 0.9
        # Not biased: the spiked solve matches the spike-free solve...
        assert np.all(np.abs(sol.tau[0] - clean.tau[0]) <= 0.5 * clean.tau_std[0])
        # ...and the truth lies inside the reported posterior uncertainty.
        assert np.all(np.abs(sol.tau[0] - tau_true) <= 3.0 * sol.tau_std[0])

    def test_motion_needing_out_of_band_torque_saturates_and_is_flagged(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        """Kinematics needing 150 N m cannot be explained by a 20 N m band.

        The start state is pinned as it is when carried from a previous window;
        a loose velocity prior could otherwise absorb part of the acceleration.
        """
        q, _ = _truth(pendulum, np.tile([150.0, 0.0], (W, 1)))
        tight = ControlBand(lower=np.full(2, -20.0), upper=np.full(2, 20.0))
        pinned = WindowOptions(sigma_obs=1e-3, sigma_q0=1e-3, sigma_v0=0.05)
        sol = solve_local_window(
            pendulum, Q0, V0, _obs(q), DT, tight, np.zeros(2), pinned
        )
        assert sol.saturated[0]
        assert np.all(sol.tau <= 20.0 + 1e-9)
        # Robust weights drop the unexplainable samples, so inlier chi2 alone
        # would look benign; the unweighted residual exposes the failure.
        assert sol.raw_rms_residual > 3.0
        assert not np.all(sol.inside_band[sol.observed])

    def test_rate_limit_keeps_window_torque_near_previous(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        q, _ = _truth(pendulum, np.tile([60.0, 0.0], (W, 1)))
        band = ControlBand(
            lower=np.full(2, -200.0),
            upper=np.full(2, 200.0),
            rate_limit=np.full(2, 500.0),
        )
        sol = solve_local_window(pendulum, Q0, V0, _obs(q), DT, band, np.zeros(2), OPTS)
        assert np.max(np.abs(sol.tau)) <= 500.0 * DT * W + 1e-9
        assert sol.rate_limited[0] and not sol.saturated[0]


class TestBandsAndDiagnostics:
    def test_clean_observations_lie_inside_plausible_band(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        q, _ = _truth(pendulum, np.tile([25.0, 2.0], (W, 1)))
        sol = solve_local_window(pendulum, Q0, V0, _obs(q), DT, WIDE, np.zeros(2), OPTS)
        assert np.all(sol.band_lower <= sol.band_upper)
        assert np.all(sol.inside_band)

    def test_ztcf_branch_is_zero_torque_rollout_from_fitted_start(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        q, _ = _truth(pendulum, np.tile([25.0, 2.0], (W, 1)))
        sol = solve_local_window(pendulum, Q0, V0, _obs(q), DT, WIDE, np.zeros(2), OPTS)
        q_ztcf, _ = rollout(pendulum, sol.q[0], sol.v[0], np.zeros((W, 2)), DT)
        np.testing.assert_allclose(sol.q_ztcf, q_ztcf, atol=1e-12)

    def test_explained_fraction_is_high_for_consistent_data(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        q, _ = _truth(pendulum, np.tile([45.0, -3.0], (W, 1)))
        sol = solve_local_window(pendulum, Q0, V0, _obs(q), DT, WIDE, np.zeros(2), OPTS)
        assert sol.explained_fraction > 0.99

    def test_underdetermined_window_reports_failure_not_exception(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        q, _ = _truth(pendulum, np.tile([25.0, 2.0], (W, 1)))
        mask = np.zeros(W + 1, bool)
        mask[0] = True
        sol = solve_local_window(
            pendulum, Q0, V0, _obs(q, mask), DT, WIDE, np.zeros(2), OPTS
        )
        assert not sol.success
        assert "observ" in sol.status


class TestContracts:
    def test_rejects_mismatched_observation_width(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            WindowObservation(q=np.zeros((5, 3)), mask=np.ones(4, bool))

    def test_rejects_nan_in_observed_rows(self) -> None:
        q = np.zeros((4, 2))
        q[1, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            WindowObservation(q=q, mask=np.ones(4, bool))

    def test_rejects_band_size_mismatch(self, pendulum) -> None:  # type: ignore[no-untyped-def]
        q, _ = _truth(pendulum, np.tile([25.0, 2.0], (W, 1)))
        band = ControlBand(lower=-np.ones(3), upper=np.ones(3))
        with pytest.raises(ValueError, match="band"):
            solve_local_window(pendulum, Q0, V0, _obs(q), DT, band, np.zeros(2), OPTS)

    def test_options_validate(self) -> None:
        with pytest.raises(ValueError):
            WindowOptions(sigma_obs=0.0, sigma_q0=1e-3, sigma_v0=0.5)
        with pytest.raises(ValueError):
            WindowOptions(sigma_obs=1e-3, sigma_q0=1e-3, sigma_v0=0.5, n_knots=0)


class TestIdentifiability:
    def test_reported_torque_uncertainty_shrinks_when_start_state_is_pinned(
        self, pendulum
    ) -> None:  # type: ignore[no-untyped-def]
        """Velocity offsets and constant torque are nearly collinear in short
        windows; the posterior spread must say so instead of hiding it."""
        q, _ = _truth(pendulum, np.tile([35.0, -4.0], (W, 1)))
        loose = solve_local_window(
            pendulum, Q0, V0, _obs(q), DT, WIDE, np.zeros(2), OPTS
        )
        pinned_opts = WindowOptions(sigma_obs=1e-3, sigma_q0=1e-3, sigma_v0=0.01)
        pinned = solve_local_window(
            pendulum, Q0, V0, _obs(q), DT, WIDE, np.zeros(2), pinned_opts
        )
        assert np.all(pinned.tau_std[0] < 0.5 * loose.tau_std[0])
