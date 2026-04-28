"""Unit tests for PinocchioPerturbationAnalyzer (#1978).

Tests are split into two groups:
1. Pure-Python tests that run without pinocchio installed (helpers, data structures).
2. Pinocchio-dependent tests that are skipped if pinocchio is not available.

Design by Contract
------------------
- Pre: set_base_torque_profile() must be called before run_batch() / perturb_torque().
- Post: run_batch() returns PerturbationSummary with all MANDATORY_METRICS.
- Post: extract_metrics() returns all MANDATORY_METRICS for valid PinocchioSimResult.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.engines.physics_engines.pinocchio.python.perturbation.analyzer import (
    MANDATORY_METRICS,
    PinocchioSimResult,
)
from src.shared.python.pendulum_simulator.perturbation_analysis import (
    perturb_torque_coeffs,
)
from src.shared.python.perturbation.analyzer_base import ComparisonReport
from src.shared.python.perturbation.config import PerturbationConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOLFER_URDF = (
    Path(__file__).parents[5]
    / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)

_SMALL_CONFIG = PerturbationConfig(
    n_trials=3,
    noise_type="white",
    noise_amplitude=0.05,
    perturb_mode="additive",
    seed=42,
)


# ---------------------------------------------------------------------------
# Pure-Python helpers (no pinocchio required)
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

    def test_multiplicative_returns_same_shape(self) -> None:
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

    def test_both_mode_returns_same_shape(self) -> None:
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

    def test_additive_reproducible_with_same_seed(self) -> None:
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


class TestPinocchioSimResult:
    def _make_sim_result(
        self, n: int = 5, nq: int = 7, nv: int = 6
    ) -> PinocchioSimResult:
        t = np.linspace(0.0, 1.0, n)
        q = np.zeros((n, nq))
        v = np.zeros((n, nv))
        ee_pos = np.zeros((n, 3))
        ee_vel = np.zeros((n, 3))
        ke = np.ones(n)
        pe = np.zeros(n)
        return PinocchioSimResult(
            t=t,
            q_traj=q,
            v_traj=v,
            ee_pos_traj=ee_pos,
            ee_vel_traj=ee_vel,
            kinetic_energy_traj=ke,
            potential_energy_traj=pe,
        )

    def test_n_steps(self) -> None:
        r = self._make_sim_result(n=10)
        assert r.n_steps == 10

    def test_trajectory_shape(self) -> None:
        r = self._make_sim_result(n=5, nq=7, nv=6)
        assert r.q_traj.shape == (5, 7)
        assert r.v_traj.shape == (5, 6)


class TestComparisonReport:
    def test_default_fields(self) -> None:
        report = ComparisonReport(winner="A", confidence=0.8)
        assert report.winner == "A"
        assert report.confidence == 0.8
        assert isinstance(report.metric_comparisons, dict)
        assert isinstance(report.pvalues, dict)


# ---------------------------------------------------------------------------
# Pinocchio-dependent tests
# ---------------------------------------------------------------------------

try:
    import pinocchio as _pin  # noqa: F401

    _PINOCCHIO_AVAILABLE = True
except ImportError:
    _PINOCCHIO_AVAILABLE = False

_skip_no_pinocchio = pytest.mark.skipif(
    not _PINOCCHIO_AVAILABLE, reason="pinocchio not installed"
)


@pytest.fixture(scope="module")
def urdf_path() -> Path:
    if not _GOLFER_URDF.exists():
        pytest.skip(f"Golfer URDF not found: {_GOLFER_URDF}")
    return _GOLFER_URDF


@pytest.fixture(scope="module")
def analyzer(urdf_path: Path):  # type: ignore[no-untyped-def]
    from src.engines.physics_engines.pinocchio.python.perturbation.analyzer import (
        PinocchioPerturbationAnalyzer,
    )

    return PinocchioPerturbationAnalyzer(urdf_path=urdf_path, t_end=0.1, dt=0.01)


@pytest.fixture(scope="module")
def simple_profile(analyzer) -> dict:  # type: ignore[no-untyped-def]
    nv = analyzer._nv
    return {"coeffs": [[0.0, 0.0] for _ in range(nv)]}


@pytest.fixture(scope="module")
def analyzer_with_profile(analyzer, simple_profile):  # type: ignore[no-untyped-def]
    analyzer.set_base_torque_profile(simple_profile)
    return analyzer


@_skip_no_pinocchio
class TestEngineMetadata:
    def test_engine_name(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer.ENGINE_NAME == "pinocchio"

    def test_nv_positive(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer._nv > 0

    def test_nq_positive(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        assert analyzer._nq > 0


@_skip_no_pinocchio
class TestSetBaseProfile:
    def test_accepts_valid_profile(self, analyzer, simple_profile) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(simple_profile)  # no exception

    def test_requires_dict(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((AssertionError, TypeError)):
            analyzer.set_base_torque_profile("not_a_dict")  # type: ignore[arg-type]

    def test_requires_coeffs_key(self, analyzer) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises((AssertionError, KeyError)):
            analyzer.set_base_torque_profile({"bad_key": []})

    def test_stores_base_coeffs(self, analyzer, simple_profile) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(simple_profile)
        assert analyzer._base_coeffs is not None


@_skip_no_pinocchio
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


@_skip_no_pinocchio
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
        with pytest.raises((AssertionError, AttributeError)):
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


@_skip_no_pinocchio
class TestRunBatch:
    def test_returns_perturbation_summary(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        from src.shared.python.perturbation.config import PerturbationSummary

        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert isinstance(result, PerturbationSummary)

    def test_engine_name_matches(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert result.engine_name == "pinocchio"

    def test_success_rate_in_range(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert 0.0 <= result.success_rate <= 1.0

    def test_robustness_score_in_range(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert 0.0 <= result.robustness_score <= 1.0

    def test_execution_time_non_negative(self, analyzer_with_profile) -> None:  # type: ignore[no-untyped-def]
        result = analyzer_with_profile.run_batch(_SMALL_CONFIG)
        assert result.execution_time_sec >= 0.0

    def test_requires_profile_set(self, urdf_path) -> None:  # type: ignore[no-untyped-def]
        from src.engines.physics_engines.pinocchio.python.perturbation.analyzer import (
            PinocchioPerturbationAnalyzer,
        )

        fresh = PinocchioPerturbationAnalyzer(urdf_path=urdf_path, t_end=0.1, dt=0.01)
        with pytest.raises((AssertionError, AttributeError)):
            fresh.run_batch(_SMALL_CONFIG)

    def test_zero_amplitude_full_success(self, analyzer, simple_profile) -> None:  # type: ignore[no-untyped-def]
        analyzer.set_base_torque_profile(simple_profile)
        config = PerturbationConfig(
            n_trials=3, noise_amplitude=0.0, noise_type="white", seed=0
        )
        result = analyzer.run_batch(config)
        assert result.success_rate == 1.0
