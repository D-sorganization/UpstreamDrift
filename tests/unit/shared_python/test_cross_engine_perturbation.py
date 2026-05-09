"""Unit tests for cross_engine_perturbation module.

Covers GH2021: Cross-engine perturbation consistency normalisation.

All tests use mock engines — no real physics engine dependencies required.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEnginePerturbationRunner,
    CrossEngineRunResult,
    CrossEngineSimConfig,
)
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_profile,
)

# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------


class MockEngine:
    """Minimal mock that implements SteppableEngine protocol.

    Simulates a simple 2-DOF system: velocities accumulate proportionally
    to the applied torque. Different ``gain`` values simulate different physics.
    """

    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain
        self._q = np.zeros(2)
        self._v = np.zeros(2)

    def reset(self) -> None:
        self._q = np.zeros(2)
        self._v = np.zeros(2)

    def set_control(self, u: np.ndarray) -> None:
        self._u = np.atleast_1d(np.array(u, dtype=float))

    def step(self, dt: float | None = None) -> None:
        step = dt if dt is not None else 0.01
        u = getattr(self, "_u", np.zeros(2))
        u2 = u if len(u) >= 2 else np.array([u[0], u[0]])
        self._v = self._v + self._gain * u2[:2] * step
        self._q = self._q + self._v * step

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), self._v.copy()


# ---------------------------------------------------------------------------
# CrossEngineSimConfig tests
# ---------------------------------------------------------------------------


class TestCrossEngineSimConfig:
    """Validate CrossEngineSimConfig defaults and constraints."""

    @pytest.mark.unit
    def test_cross_engine_perturbation_defaults(self) -> None:
        """Default config should have expected values from GH2021 diagnostics."""
        cfg = CrossEngineSimConfig()
        assert cfg.t_end == pytest.approx(1.5)
        assert cfg.dt == pytest.approx(0.01)
        assert cfg.noise_amplitude == pytest.approx(0.1)
        assert cfg.n_trials == 10
        assert cfg.seed == 42

    @pytest.mark.unit
    def test_cross_engine_perturbation_custom_config(self) -> None:
        """Custom values should be accepted."""
        cfg = CrossEngineSimConfig(
            t_end=2.0, dt=0.005, noise_amplitude=0.5, n_trials=5, seed=7
        )
        assert cfg.t_end == pytest.approx(2.0)
        assert cfg.dt == pytest.approx(0.005)

    @pytest.mark.unit
    def test_invalid_dt_zero(self) -> None:
        """dt=0 must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            CrossEngineSimConfig(dt=0.0)

    @pytest.mark.unit
    def test_invalid_dt_negative(self) -> None:
        """Negative dt must raise ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            CrossEngineSimConfig(dt=-0.01)

    @pytest.mark.unit
    def test_invalid_t_end_zero(self) -> None:
        """t_end=0 must raise ValueError."""
        with pytest.raises(ValueError, match="t_end must be positive"):
            CrossEngineSimConfig(t_end=0.0)

    @pytest.mark.unit
    def test_t_end_must_exceed_dt(self) -> None:
        """t_end <= dt must raise ValueError."""
        with pytest.raises(ValueError, match="t_end .* must be greater than dt"):
            CrossEngineSimConfig(t_end=0.005, dt=0.01)

    @pytest.mark.unit
    def test_invalid_noise_amplitude_negative(self) -> None:
        """Negative noise_amplitude must raise ValueError."""
        with pytest.raises(ValueError, match="noise_amplitude must be non-negative"):
            CrossEngineSimConfig(noise_amplitude=-0.1)

    @pytest.mark.unit
    def test_invalid_n_trials_zero(self) -> None:
        """n_trials=0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            CrossEngineSimConfig(n_trials=0)


# ---------------------------------------------------------------------------
# Runner registration tests
# ---------------------------------------------------------------------------


class TestRunnerRegistration:
    """Test engine registration."""

    @pytest.mark.unit
    def test_register_single_engine(self) -> None:
        """Registered engine name should appear in internal registry."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        runner.register_engine("pendulum", MockEngine())
        assert "pendulum" in runner._engines

    @pytest.mark.unit
    def test_register_multiple_engines(self) -> None:
        """Multiple engines can be registered."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        runner.register_engine("eng_a", MockEngine(gain=1.0))
        runner.register_engine("eng_b", MockEngine(gain=2.0))
        assert len(runner._engines) == 2

    @pytest.mark.unit
    def test_register_empty_name_raises(self) -> None:
        """Empty name must raise ValueError."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        with pytest.raises(ValueError, match="Engine name must be non-empty"):
            runner.register_engine("", MockEngine())

    @pytest.mark.unit
    def test_register_duplicate_name_raises(self) -> None:
        """Duplicate name must raise ValueError."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        runner.register_engine("eng", MockEngine())
        with pytest.raises(ValueError, match="already registered"):
            runner.register_engine("eng", MockEngine())

    @pytest.mark.unit
    def test_run_comparison_no_engines_raises(self) -> None:
        """run_comparison without registered engines must raise RuntimeError."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        profile = np.zeros(round(cfg.t_end / cfg.dt))
        with pytest.raises(RuntimeError, match="No engines registered"):
            runner.run_comparison(profile)


# ---------------------------------------------------------------------------
# run_comparison tests
# ---------------------------------------------------------------------------


class TestRunComparison:
    """Test the main comparison loop."""

    def _make_runner(
        self, n_trials: int = 3, noise_amplitude: float = 0.1
    ) -> CrossEnginePerturbationRunner:
        cfg = CrossEngineSimConfig(
            t_end=0.1,
            dt=0.01,
            n_trials=n_trials,
            noise_amplitude=noise_amplitude,
            seed=0,
        )
        runner = CrossEnginePerturbationRunner(cfg)
        return runner

    @pytest.mark.unit
    def test_returns_result_per_engine(self) -> None:
        """run_comparison should return one result per registered engine."""
        runner = self._make_runner()
        runner.register_engine("eng_a", MockEngine())
        runner.register_engine("eng_b", MockEngine())
        n_steps = round(runner.config.t_end / runner.config.dt)
        results = runner.run_comparison(np.ones(n_steps))
        assert set(results.keys()) == {"eng_a", "eng_b"}

    @pytest.mark.unit
    def test_trial_count_matches_config(self) -> None:
        """Each engine result should have n_trials trial metrics."""
        runner = self._make_runner(n_trials=5)
        runner.register_engine("eng", MockEngine())
        n_steps = round(runner.config.t_end / runner.config.dt)
        results = runner.run_comparison(np.ones(n_steps))
        assert len(results["eng"].metrics_per_trial) == 5

    @pytest.mark.unit
    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical results on repeated calls."""
        runner = self._make_runner(n_trials=2, noise_amplitude=0.5)
        runner.register_engine("eng", MockEngine())
        n_steps = round(runner.config.t_end / runner.config.dt)
        profile = np.ones(n_steps) * 2.0
        r1 = runner.run_comparison(profile)

        runner2 = self._make_runner(n_trials=2, noise_amplitude=0.5)
        runner2.register_engine("eng", MockEngine())
        r2 = runner2.run_comparison(profile)

        assert r1["eng"].mean_total_energy_final == pytest.approx(
            r2["eng"].mean_total_energy_final, rel=1e-10
        )

    @pytest.mark.unit
    def test_identical_engines_low_cv(self) -> None:
        """Two engines with identical physics should produce near-zero CV across engines."""
        runner = self._make_runner(n_trials=3, noise_amplitude=0.0)
        runner.register_engine("eng_a", MockEngine(gain=1.0))
        runner.register_engine("eng_b", MockEngine(gain=1.0))
        n_steps = round(runner.config.t_end / runner.config.dt)
        results = runner.run_comparison(np.ones(n_steps))
        cv = runner.compute_cv_summary(results)
        # Identical engines with identical noise → CV should be 0
        assert cv["cv_end_effector_speed_final"] < 1e-10

    @pytest.mark.unit
    def test_different_engines_nonzero_cv(self) -> None:
        """Engines with different gains should produce nonzero CV."""
        runner = self._make_runner(n_trials=3, noise_amplitude=0.1)
        runner.register_engine("slow", MockEngine(gain=1.0))
        runner.register_engine("fast", MockEngine(gain=10.0))
        n_steps = round(runner.config.t_end / runner.config.dt)
        results = runner.run_comparison(np.ones(n_steps))
        cv = runner.compute_cv_summary(results)
        # Very different gains → high CV
        assert cv["cv_end_effector_speed_final"] > 0.1

    @pytest.mark.unit
    def test_profile_shape_mismatch_raises(self) -> None:
        """Wrong-length profile must raise ValueError."""
        runner = self._make_runner()
        runner.register_engine("eng", MockEngine())
        wrong_profile = np.ones(5)  # too short
        with pytest.raises(ValueError, match="expected"):
            runner.run_comparison(wrong_profile)

    @pytest.mark.unit
    def test_dt_override_respected(self) -> None:
        """Engine must be called with the config dt, not its internal default."""
        cfg = CrossEngineSimConfig(t_end=0.05, dt=0.01, n_trials=1, noise_amplitude=0.0)
        runner = CrossEnginePerturbationRunner(cfg)
        n_steps = round(cfg.t_end / cfg.dt)  # 5

        class StepRecorder:
            """Records dt values passed to step()."""

            def __init__(self) -> None:
                self.dt_calls: list[float | None] = []
                self._q = np.zeros(2)
                self._v = np.zeros(2)
                self._u = np.zeros(2)

            def reset(self) -> None:
                self._q = np.zeros(2)
                self._v = np.zeros(2)

            def set_control(self, u: np.ndarray) -> None:
                self._u = np.atleast_1d(u)

            def step(self, dt: float | None = None) -> None:
                self.dt_calls.append(dt)

            def get_state(self) -> tuple[np.ndarray, np.ndarray]:
                return self._q.copy(), self._v.copy()

        recorder = StepRecorder()
        runner.register_engine("rec", recorder)
        profile = np.zeros(n_steps)
        runner.run_comparison(profile)

        assert all(d == pytest.approx(0.01) for d in recorder.dt_calls)


# ---------------------------------------------------------------------------
# compute_cv_summary tests
# ---------------------------------------------------------------------------


class TestComputeCvSummary:
    """Test CV computation."""

    @pytest.mark.unit
    def test_zero_variance_returns_zero_cv(self) -> None:
        """When all engines produce same mean, CV must be 0."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        results = {
            "a": CrossEngineRunResult(
                engine_name="a",
                mean_total_energy_final=5.0,
                mean_end_effector_speed_final=3.0,
                mean_peak_end_effector_speed=4.0,
            ),
            "b": CrossEngineRunResult(
                engine_name="b",
                mean_total_energy_final=5.0,
                mean_end_effector_speed_final=3.0,
                mean_peak_end_effector_speed=4.0,
            ),
        }
        cv = runner.compute_cv_summary(results)
        assert cv["cv_total_energy_final"] == pytest.approx(0.0, abs=1e-12)
        assert cv["cv_end_effector_speed_final"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.unit
    def test_zero_mean_returns_zero_cv(self) -> None:
        """When mean is near zero, CV must return 0.0 (not division by zero)."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        results = {
            "a": CrossEngineRunResult(
                engine_name="a",
                mean_total_energy_final=0.0,
                mean_end_effector_speed_final=0.0,
                mean_peak_end_effector_speed=0.0,
            ),
            "b": CrossEngineRunResult(
                engine_name="b",
                mean_total_energy_final=0.0,
                mean_end_effector_speed_final=0.0,
                mean_peak_end_effector_speed=0.0,
            ),
        }
        cv = runner.compute_cv_summary(results)
        assert cv["cv_total_energy_final"] == 0.0

    @pytest.mark.unit
    def test_empty_results_raises(self) -> None:
        """Empty results dict must raise ValueError."""
        cfg = CrossEngineSimConfig(n_trials=1)
        runner = CrossEnginePerturbationRunner(cfg)
        with pytest.raises(ValueError, match="non-empty"):
            runner.compute_cv_summary({})


# ---------------------------------------------------------------------------
# perturb_torque_profile tests
# ---------------------------------------------------------------------------


class TestPerturbTorqueProfile:
    """Test the additive torque profile perturbation function."""

    @pytest.mark.unit
    def test_shape_preserved(self) -> None:
        """Output shape must equal input shape."""
        profile = np.ones(150)
        result = perturb_torque_profile(profile, noise_amplitude=0.1, seed=0)
        assert result.shape == profile.shape

    @pytest.mark.unit
    def test_zero_amplitude_no_change(self) -> None:
        """Zero amplitude must return an unchanged copy."""
        profile = np.linspace(0, 10, 100)
        result = perturb_torque_profile(profile, noise_amplitude=0.0)
        np.testing.assert_array_equal(result, profile)

    @pytest.mark.unit
    def test_seed_reproducibility(self) -> None:
        """Same seed must produce identical output."""
        profile = np.ones(50)
        r1 = perturb_torque_profile(profile, noise_amplitude=1.0, seed=7)
        r2 = perturb_torque_profile(profile, noise_amplitude=1.0, seed=7)
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.unit
    def test_different_seeds_different_output(self) -> None:
        """Different seeds must produce different outputs (statistically)."""
        profile = np.ones(50)
        r1 = perturb_torque_profile(profile, noise_amplitude=1.0, seed=1)
        r2 = perturb_torque_profile(profile, noise_amplitude=1.0, seed=2)
        assert not np.allclose(r1, r2)

    @pytest.mark.unit
    def test_mean_noise_near_zero(self) -> None:
        """Over many samples, mean of added noise should be near zero."""
        profile = np.zeros(10000)
        result = perturb_torque_profile(profile, noise_amplitude=1.0, seed=0)
        np.testing.assert_allclose(np.mean(result), 0.0, atol=0.05)

    @pytest.mark.unit
    def test_invalid_noise_type_raises(self) -> None:
        """Unsupported noise_type must raise ValueError."""
        with pytest.raises(ValueError, match="noise_type must be 'additive'"):
            perturb_torque_profile(np.ones(10), noise_amplitude=0.1, noise_type="pink")

    @pytest.mark.unit
    def test_cross_engine_perturbation_negative_amplitude_raises(self) -> None:
        """Negative amplitude must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            perturb_torque_profile(np.ones(10), noise_amplitude=-1.0)

    @pytest.mark.unit
    def test_empty_profile_raises(self) -> None:
        """Empty profile must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            perturb_torque_profile(np.array([]), noise_amplitude=0.1)
