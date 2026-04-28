"""Unit tests for MuJoCoPerturbationAnalyzer (#1980).

All tests run when mujoco is installed.  Tests skip gracefully when not.

Design by Contract
------------------
- Pre: set_base_torque_profile() must be called before run_batch() / perturb_torque().
- Post: run_batch() returns PerturbationSummary with all MANDATORY_METRICS.
- Post: extract_metrics() returns all MANDATORY_METRICS for valid MuJoCoSimResult.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.perturbation.analyzer import (
    MANDATORY_METRICS,
    MuJoCoSimResult,
)
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
)
from src.shared.python.perturbation.analyzer_base import ComparisonReport
from src.shared.python.perturbation.config import PerturbationConfig

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

try:
    import mujoco as _mujoco  # noqa: F401

    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False

_skip_no_mujoco = pytest.mark.skipif(
    not _MUJOCO_AVAILABLE, reason="mujoco not installed"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SMALL_CONFIG = PerturbationConfig(
    n_trials=3,
    noise_type="white",
    noise_amplitude=0.05,
    perturb_mode="additive",
    seed=42,
)

_ZERO_PROFILE: dict = {"coeffs": [[0.0, 0.0], [0.0, 0.0]]}


# ---------------------------------------------------------------------------
# Pure-Python helpers (no mujoco required)
# ---------------------------------------------------------------------------


class TestMandatoryMetrics:
    def test_non_empty(self) -> None:
        assert len(MANDATORY_METRICS) >= 10

    def test_contains_required_names(self) -> None:
        required = {
            "end_effector_position_final",
            "end_effector_velocity_final",
            "end_effector_speed_final",
            "peak_end_effector_speed",
            "total_energy_final",
            "joint_angles_final",
            "joint_velocities_final",
            "trajectory_rmse",
            "trajectory_max_deviation",
            "motion_duration",
        }
        assert required.issubset(set(MANDATORY_METRICS))

    def test_no_duplicates(self) -> None:
        assert len(MANDATORY_METRICS) == len(set(MANDATORY_METRICS))


class TestPerturbCoeffsByMode:
    def test_additive_zero_amplitude_no_change(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1, noise_amplitude=0.0, noise_type="white", perturb_mode="additive"
        )
        result = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=0,
        )
        flat_orig = [c for j in coeffs for c in j]
        flat_res = [c for j in result for c in j]
        assert all(abs(a - b) < 1e-12 for a, b in zip(flat_orig, flat_res, strict=True))

    def test_multiplicative_same_shape(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1,
            noise_amplitude=0.1,
            noise_type="white",
            perturb_mode="multiplicative",
        )
        result = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=0,
        )
        assert len(result) == len(coeffs)
        for orig, res in zip(coeffs, result, strict=True):
            assert len(res) == len(orig)

    def test_both_mode_same_shape(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1, noise_amplitude=0.1, noise_type="white", perturb_mode="both"
        )
        result = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=0,
        )
        assert len(result) == len(coeffs)

    def test_reproducible_with_seed(self) -> None:
        coeffs = [[1.0, 2.0], [3.0, 4.0]]
        config = PerturbationConfig(
            n_trials=1, noise_amplitude=0.3, noise_type="white", perturb_mode="additive"
        )
        r1 = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=5,
        )
        r2 = perturb_torque_coeffs(
            coeffs,
            noise_amplitude=config.noise_amplitude,
            noise_type=config.noise_type,
            perturb_mode=config.perturb_mode,
            seed=5,
        )
        assert r1 == r2


class TestMuJoCoSimResult:
    def _make(self, n: int = 5, nq: int = 2, nv: int = 2) -> MuJoCoSimResult:
        t = np.linspace(0.0, 1.0, n)
        qpos = np.zeros((n, nq))
        qvel = np.zeros((n, nv))
        ee_pos = np.zeros((n, 3))
        ee_vel = np.zeros((n, 3))
        ke = np.ones(n)
        pe = np.zeros(n)
        return MuJoCoSimResult(
            t=t,
            qpos_traj=qpos,
            qvel_traj=qvel,
            ee_pos_traj=ee_pos,
            ee_vel_traj=ee_vel,
            kinetic_energy_traj=ke,
            potential_energy_traj=pe,
        )

    def test_n_steps(self) -> None:
        r = self._make(n=10)
        assert r.n_steps == 10

    def test_trajectory_shapes(self) -> None:
        r = self._make(n=5, nq=2, nv=2)
        assert r.qpos_traj.shape == (5, 2)
        assert r.qvel_traj.shape == (5, 2)


class TestComparisonReport:
    def test_default_fields(self) -> None:
        report = ComparisonReport(winner="A", confidence=0.9)
        assert report.winner == "A"
        assert report.confidence == 0.9
        assert isinstance(report.metric_comparisons, dict)
        assert isinstance(report.pvalues, dict)


# ---------------------------------------------------------------------------
# MuJoCo-dependent tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analyzer():  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.mujoco.python.perturbation.analyzer import (
        MuJoCoPerturbationAnalyzer,
    )

    return MuJoCoPerturbationAnalyzer(t_end=0.1)


@pytest.fixture(scope="module")
def analyzer_with_profile(analyzer):  # type: ignore[no-untyped-def]
    analyzer.set_base_torque_profile(_ZERO_PROFILE)
    return analyzer


@_skip_no_mujoco
class TestEngineMetadata:
    def test_engine_name(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer.ENGINE_NAME == "mujoco"

    def test_nq_positive(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer._nq > 0

    def test_nu_positive(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer._nu > 0


@_skip_no_mujoco
class TestSetBaseProfile:
    def test_accepts_valid_profile(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(_ZERO_PROFILE)  # no exception

    def test_requires_dict(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((ValueError, AssertionError, TypeError)):
            analyzer.set_base_torque_profile("not_a_dict")  # type: ignore[arg-type]

    def test_requires_coeffs_key(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((ValueError, AssertionError, KeyError)):
            analyzer.set_base_torque_profile({"bad_key": []})

    def test_stores_base_coeffs(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(_ZERO_PROFILE)
        assert analyzer._base_coeffs is not None


@_skip_no_mujoco
class TestPerturbTorque:
    def test_returns_dict_with_coeffs(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
        assert isinstance(result, dict)
        assert "coeffs" in result

    def test_same_shape_as_base(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        base = analyzer_with_profile._base_coeffs
        result = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=0)
        assert len(result["coeffs"]) == len(base)

    def test_reproducible_with_seed(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        r1 = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=7)
        r2 = analyzer_with_profile.perturb_torque(_SMALL_CONFIG, seed=7)
        assert r1["coeffs"] == r2["coeffs"]


@_skip_no_mujoco
class TestExtractMetrics:
    def test_returns_all_mandatory_metrics(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        for name in MANDATORY_METRICS:
            assert name in metrics, f"Missing metric: {name}"

    def test_values_are_finite(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        for name, val in metrics.items():
            if isinstance(val, np.ndarray):
                assert np.all(np.isfinite(val)), f"Non-finite array metric: {name}"
            else:
                assert np.isfinite(float(val)), f"Non-finite scalar metric: {name}"

    def test_rejects_invalid_input(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((ValueError, AssertionError, AttributeError)):
            analyzer_with_profile.extract_metrics("bad_input")  # type: ignore[arg-type]

    def test_motion_duration_positive(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        assert float(metrics["motion_duration"]) > 0.0

    def test_speed_non_negative(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        coeffs = analyzer_with_profile._base_coeffs
        sim = analyzer_with_profile._simulate(coeffs)
        metrics = analyzer_with_profile.extract_metrics(sim)
        assert float(metrics["end_effector_speed_final"]) >= 0.0
        assert float(metrics["peak_end_effector_speed"]) >= 0.0


@_skip_no_mujoco
class TestRunBatch:
    def test_returns_perturbation_summary(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        from src.shared.python.perturbation.config import PerturbationSummary

        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert isinstance(result, PerturbationSummary)

    def test_engine_name_matches(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert result.engine_name == "mujoco"

    def test_success_rate_in_range(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert 0.0 <= result.success_rate <= 1.0

    def test_robustness_score_in_range(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert 0.0 <= result.robustness_score <= 1.0

    def test_execution_time_non_negative(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert result.execution_time_sec >= 0.0

    def test_zero_amplitude_full_success(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(_ZERO_PROFILE)
        config = PerturbationConfig(
            n_trials=3, noise_amplitude=0.0, noise_type="white", seed=0
        )
        result = analyzer.run_batch(config)
        assert result.success_rate == 1.0

    def test_requires_profile_set(self) -> None:
        from src.engines.physics_engines.mujoco.python.perturbation.analyzer import (
            MuJoCoPerturbationAnalyzer,
        )

        fresh = MuJoCoPerturbationAnalyzer(t_end=0.1)
        with pytest.raises((ValueError, AssertionError, AttributeError)):
            fresh.run_batch(_SMALL_CONFIG)

    def test_contains_scalar_metrics(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        scalar_expected = {
            "end_effector_speed_final",
            "peak_end_effector_speed",
            "total_energy_final",
            "trajectory_rmse",
            "trajectory_max_deviation",
            "motion_duration",
        }
        for name in scalar_expected:
            assert name in result.metrics, f"Missing metric: {name}"
