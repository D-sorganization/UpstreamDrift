"""Tests for DisturbanceSpec and its sampler (NSX-82, #12711).

Spec: docs/research/loop_nullspace/loop_nullspace.tex ``sec:var-cov`` (V6
timestep consistency) and ``sec:var-mechanisms`` (M1 signal-dependent law).

Statistical tolerances are derived from the sample size, not tuned: for ``n``
independent Gaussian draws the sample variance has relative standard error
``sqrt(2 / (n - 1))`` and the sample covariance of components ``i, j`` has
standard error ``sqrt((C_ii C_jj + C_ij^2) / n)``.  Every check uses
``Z = 4.5`` standard errors (two-sided false-alarm rate about 7e-6) with fixed
seeds, so the tests are deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)

pytestmark = pytest.mark.unit

Z = 4.5
COV = np.array([[4.0, 1.5], [1.5, 2.0]])


def _spec(**overrides: object) -> DisturbanceSpec:
    kwargs: dict[str, object] = {
        "name": "shoulder_torque_noise",
        "stage": InjectionStage.TORQUE,
        "channels": ("u1", "u2"),
        "units": "N*m",
        "covariance": COV,
        "spectrum": Spectrum.WHITE,
    }
    kwargs.update(overrides)
    return DisturbanceSpec(**kwargs)  # type: ignore[arg-type]


class TestConstruction:
    def test_frozen(self) -> None:
        spec = _spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.units = "rad"  # type: ignore[misc]

    def test_covariance_is_read_only_copy(self) -> None:
        cov = COV.copy()
        spec = _spec(covariance=cov)
        cov[0, 0] = 99.0
        assert spec.covariance[0, 0] == 4.0
        with pytest.raises(ValueError):
            spec.covariance[0, 0] = 1.0

    @pytest.mark.parametrize("stage", list(InjectionStage))
    def test_every_declared_stage_constructs(self, stage: InjectionStage) -> None:
        assert _spec(stage=stage).stage is stage

    def test_stage_set_matches_issue(self) -> None:
        assert {s.value for s in InjectionStage} == {
            "torque",
            "activation",
            "initial_state",
            "timing",
            "grip_split",
            "external_hand_wrench",
        }

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"units": ""}, "units"),
            ({"name": ""}, "name"),
            ({"channels": ("u1",)}, "channels"),
            ({"channels": ("u1", "u1")}, "unique"),
            ({"covariance": np.eye(3)}, "shape"),
            ({"covariance": np.array([[1.0, 2.0], [0.0, 1.0]])}, "symmetric"),
            ({"covariance": np.array([[1.0, 2.0], [2.0, 1.0]])}, "positive semi"),
            ({"covariance": np.array([[np.nan, 0.0], [0.0, 1.0]])}, "finite"),
            ({"covariance": -np.eye(2)}, "positive semi"),
            ({"spectrum": None}, "persistent"),
            ({"persistent": True}, "persistent"),
            ({"spectrum": Spectrum.WHITE, "bandwidth_hz": 5.0}, "bandwidth"),
            ({"spectrum": Spectrum.ORNSTEIN_UHLENBECK}, "bandwidth_hz"),
            (
                {"spectrum": Spectrum.ORNSTEIN_UHLENBECK, "bandwidth_hz": -1.0},
                "bandwidth_hz",
            ),
            ({"spectrum": Spectrum.PINK}, "band_hz"),
            ({"spectrum": Spectrum.PINK, "band_hz": (5.0, 1.0)}, "band_hz"),
            ({"signal_gain": (-1.0, 0.0), "signal_bounds": ((0, 1), (0, 1))}, "gain"),
            ({"signal_gain": (1.0, 0.0)}, "signal_bounds"),
            ({"signal_gain": (1.0, 0.0), "signal_bounds": ((1, 0), (0, 1))}, "bounds"),
            (
                {
                    "spectrum": Spectrum.ORNSTEIN_UHLENBECK,
                    "bandwidth_hz": 4.0,
                    "signal_gain": (1.0, 0.0),
                    "signal_bounds": ((0, 1), (0, 1)),
                },
                "signal-dependent",
            ),
        ],
    )
    def test_invalid_declarations_rejected(
        self, overrides: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _spec(**overrides)

    def test_wrong_stage_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            _spec(stage="torque")

    def test_psd_units_label(self) -> None:
        assert _spec().covariance_units == "(N*m)^2*s"
        persistent = _spec(spectrum=None, persistent=True)
        assert persistent.covariance_units == "(N*m)^2"


class TestTimestepConsistency:
    def test_white_discrete_covariance_is_psd_over_dt(self) -> None:
        spec = _spec()
        np.testing.assert_allclose(spec.discrete_covariance(0.01), COV / 0.01)
        np.testing.assert_allclose(spec.discrete_covariance(0.005), COV / 0.005)

    def test_persistent_covariance_does_not_depend_on_dt(self) -> None:
        spec = _spec(spectrum=None, persistent=True)
        np.testing.assert_allclose(spec.discrete_covariance(0.01), COV)
        np.testing.assert_allclose(spec.discrete_covariance(0.001), COV)

    def test_halving_dt_leaves_integrated_white_variance_unchanged(self) -> None:
        horizon = 0.4  # s
        n_trials = 20000
        spec = _spec()
        expected = np.diag(COV) * horizon  # Var of the integral of white noise
        rel_tol = Z * np.sqrt(2.0 / (n_trials - 1))
        measured = {}
        for dt in (0.01, 0.005):
            n_steps = round(horizon / dt)
            w = spec.sample(n_trials=n_trials, n_steps=n_steps, dt=dt, seed=7)
            integral = w.sum(axis=1) * dt  # (n_trials, channels)
            measured[dt] = integral.var(axis=0, ddof=1)
            np.testing.assert_allclose(measured[dt], expected, rtol=rel_tol)
        # the two timesteps agree with each other within the same statistics
        np.testing.assert_allclose(
            measured[0.01], measured[0.005], rtol=np.sqrt(2) * rel_tol
        )

    def test_ou_stationary_variance_is_dt_invariant(self) -> None:
        spec = _spec(spectrum=Spectrum.ORNSTEIN_UHLENBECK, bandwidth_hz=2.0)
        n_trials = 20000
        expected = np.diag(COV) / (2.0 * spec.ou_time_constant)
        rel_tol = Z * np.sqrt(2.0 / (n_trials - 1))
        for dt in (0.02, 0.01):
            w = spec.sample(n_trials=n_trials, n_steps=60, dt=dt, seed=3)
            last = w[:, -1, :]
            np.testing.assert_allclose(last.var(axis=0, ddof=1), expected, rtol=rel_tol)

    def test_ou_lag_one_autocorrelation_matches_exp_dt_over_tau(self) -> None:
        spec = _spec(spectrum=Spectrum.ORNSTEIN_UHLENBECK, bandwidth_hz=2.0)
        dt, n_trials = 0.02, 20000
        w = spec.sample(n_trials=n_trials, n_steps=40, dt=dt, seed=11)
        phi = np.exp(-dt / spec.ou_time_constant)
        rho = (w[:, -1, 0] * w[:, -2, 0]).mean() / w[:, -1, 0].var()
        # Fisher-z standard error of a correlation estimate: 1/sqrt(n - 3)
        assert abs(rho - phi) < Z * (1 - phi**2) / np.sqrt(n_trials - 3)

    def test_pink_ensemble_variance_matches_declared_band(self) -> None:
        spec = _spec(spectrum=Spectrum.PINK, band_hz=(1.0, 20.0))
        n_trials, dt, n_steps = 20000, 0.01, 200
        w = spec.sample(n_trials=n_trials, n_steps=n_steps, dt=dt, seed=5)
        expected = np.diag(COV) * spec.pink_band_variance_factor(dt, n_steps)
        rel_tol = Z * np.sqrt(2.0 / (n_trials - 1))
        np.testing.assert_allclose(
            w[:, 10, :].var(axis=0, ddof=1), expected, rtol=rel_tol
        )

    def test_pink_band_must_sit_below_nyquist(self) -> None:
        spec = _spec(spectrum=Spectrum.PINK, band_hz=(1.0, 80.0))
        with pytest.raises(ValueError, match="Nyquist"):
            spec.sample(n_trials=2, n_steps=100, dt=0.01, seed=0)


class TestSampledMoments:
    def test_persistent_sample_recovers_declared_covariance(self) -> None:
        n = 40000
        spec = _spec(spectrum=None, persistent=True)
        draws = spec.sample(n_trials=n, n_steps=1, dt=0.01, seed=1)
        assert draws.shape == (n, 2)
        sample_cov = np.cov(draws.T)
        se = np.sqrt((np.outer(np.diag(COV), np.diag(COV)) + COV**2) / n)
        assert np.all(np.abs(sample_cov - COV) < Z * se)
        assert np.all(np.abs(draws.mean(axis=0)) < Z * np.sqrt(np.diag(COV) / n))

    def test_white_per_step_sample_recovers_cross_channel_covariance(self) -> None:
        n, dt = 4000, 0.01
        spec = _spec()
        w = spec.sample(n_trials=n, n_steps=10, dt=dt, seed=2)
        assert w.shape == (n, 10, 2)
        flat = w.reshape(-1, 2)
        target = COV / dt
        m = flat.shape[0]
        se = np.sqrt((np.outer(np.diag(target), np.diag(target)) + target**2) / m)
        assert np.all(np.abs(np.cov(flat.T) - target) < Z * se)

    def test_sampling_is_deterministic_for_a_seed(self) -> None:
        a = _spec().sample(n_trials=5, n_steps=4, dt=0.01, seed=9)
        b = _spec().sample(n_trials=5, n_steps=4, dt=0.01, seed=9)
        c = _spec().sample(n_trials=5, n_steps=4, dt=0.01, seed=10)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_singular_covariance_has_no_variance_in_null_direction(self) -> None:
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank one, PSD
        spec = _spec(covariance=cov, spectrum=None, persistent=True)
        draws = spec.sample(n_trials=2000, n_steps=1, dt=0.01, seed=4)
        assert np.max(np.abs(draws[:, 0] - draws[:, 1])) < 1e-12


class TestPersistenceShape:
    def test_persistent_is_one_constant_state_per_trial(self) -> None:
        spec = _spec(spectrum=None, persistent=True)
        draws = spec.sample(n_trials=6, n_steps=50, dt=0.01, seed=0)
        assert draws.shape == (6, 2)  # no time axis: constant random states

    def test_per_step_has_time_axis(self) -> None:
        w = _spec().sample(n_trials=6, n_steps=50, dt=0.01, seed=0)
        assert w.shape == (6, 50, 2)
        assert not np.allclose(w[:, 0, :], w[:, 1, :])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_trials": 0, "n_steps": 5, "dt": 0.01},
            {"n_trials": 2, "n_steps": 0, "dt": 0.01},
            {"n_trials": 2, "n_steps": 5, "dt": 0.0},
            {"n_trials": 2, "n_steps": 5, "dt": float("nan")},
        ],
    )
    def test_sample_preconditions(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            _spec().sample(seed=0, **kwargs)  # type: ignore[arg-type]


class TestSignalDependentLaw:
    """M1: sigma_i^2 = sigma0_i^2 + k_i^2 u_i^2 (sec:var-mechanisms)."""

    def _m1(self, **overrides: object) -> DisturbanceSpec:
        return _spec(
            covariance=np.diag([0.25, 0.01]),
            signal_gain=(0.5, 0.0),
            signal_bounds=((-10.0, 10.0), (-10.0, 10.0)),
            **overrides,
        )

    def test_covariance_at_signal_adds_k_squared_u_squared(self) -> None:
        spec = self._m1()
        cov = spec.covariance_at(np.array([4.0, 7.0]))
        np.testing.assert_allclose(np.diag(cov), [0.25 + 0.25 * 16.0, 0.01])

    def test_zero_gain_is_additive_and_signal_independent(self) -> None:
        spec = self._m1()
        a = spec.covariance_at(np.array([0.0, 1.0]))
        b = spec.covariance_at(np.array([0.0, -9.0]))
        np.testing.assert_allclose(a, b)

    def test_cross_channel_covariance_preserved(self) -> None:
        spec = _spec(signal_gain=(1.0, 1.0), signal_bounds=((-5, 5), (-5, 5)))
        cov = spec.covariance_at(np.array([1.0, 2.0]))
        assert cov[0, 1] == COV[0, 1]
        np.testing.assert_allclose(np.diag(cov), np.diag(COV) + [1.0, 4.0])

    def test_sampled_variance_follows_the_signal(self) -> None:
        n, dt = 40000, 0.01
        spec = self._m1(spectrum=None, persistent=True)
        draws = spec.sample(
            n_trials=n, n_steps=1, dt=dt, seed=6, signal=np.array([4.0, 0.0])
        )
        expected = 0.25 + 0.25 * 16.0
        assert draws[:, 0].var(ddof=1) == pytest.approx(
            expected, rel=Z * np.sqrt(2.0 / (n - 1))
        )

    def test_white_per_step_signal_changes_variance_over_time(self) -> None:
        n, dt = 20000, 0.01
        spec = self._m1()
        signal = np.zeros((2, 2))
        signal[1, 0] = 8.0  # step 1 has 8x the drive on channel 0
        w = spec.sample(n_trials=n, n_steps=2, dt=dt, seed=8, signal=signal)
        rel = Z * np.sqrt(2.0 / (n - 1))
        assert w[:, 0, 0].var(ddof=1) == pytest.approx(0.25 / dt, rel=rel)
        assert w[:, 1, 0].var(ddof=1) == pytest.approx(
            (0.25 + 0.25 * 64.0) / dt, rel=rel
        )

    def test_signal_outside_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="bounds"):
            self._m1().covariance_at(np.array([11.0, 0.0]))

    def test_signal_required_when_law_declared(self) -> None:
        with pytest.raises(ValueError, match="signal"):
            self._m1().sample(n_trials=2, n_steps=2, dt=0.01, seed=0)

    def test_signal_rejected_when_no_law_declared(self) -> None:
        with pytest.raises(ValueError, match="signal"):
            _spec().sample(n_trials=2, n_steps=2, dt=0.01, seed=0, signal=np.zeros(2))
