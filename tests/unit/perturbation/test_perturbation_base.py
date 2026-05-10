"""Unit tests for PerturbationAnalyzerBase (#2273).

Tests the shared abstract base class in
``src.shared.python.perturbation.perturbation_base``.  A minimal concrete
subclass (``StubAnalyzer``) is used to exercise all base-class logic
without requiring any physics engine to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from src.shared.python.perturbation.config import PerturbationConfig
from src.shared.python.perturbation.perturbation_base import (
    _ARRAY_METRICS,
    MANDATORY_METRICS,
    ComparisonReport,
    PerturbationAnalyzerBase,
    _compute_cv_values,
)
from src.shared.python.perturbation.statistics import MetricStatistics

# ---------------------------------------------------------------------------
# Minimal stub sim-result
# ---------------------------------------------------------------------------


@dataclass
class _StubSimResult:
    """Trivial sim-result that satisfies the SimResultProtocol duck-type."""

    t: np.ndarray
    q_traj: np.ndarray
    v_traj: np.ndarray
    ee_pos_traj: np.ndarray
    ee_vel_traj: np.ndarray
    kinetic_energy_traj: np.ndarray
    potential_energy_traj: np.ndarray

    @property
    def n_steps(self) -> int:
        return len(self.t)


def _make_stub_result(n: int = 5, nq: int = 2) -> _StubSimResult:
    """Create a deterministic stub simulation result."""
    t = np.linspace(0.0, 0.1 * n, n)
    q = np.tile(np.arange(nq, dtype=float), (n, 1)) * 0.01
    v = np.zeros((n, nq))
    ee_pos = np.column_stack([np.linspace(0, 0.1, n), np.zeros(n), np.zeros(n)])
    ee_vel = np.zeros((n, 3))
    ke = np.linspace(0.1, 0.5, n)
    pe = np.linspace(0.5, 0.1, n)
    return _StubSimResult(
        t=t,
        q_traj=q,
        v_traj=v,
        ee_pos_traj=ee_pos,
        ee_vel_traj=ee_vel,
        kinetic_energy_traj=ke,
        potential_energy_traj=pe,
    )


# ---------------------------------------------------------------------------
# Minimal concrete subclass of the base
# ---------------------------------------------------------------------------


class StubAnalyzer(PerturbationAnalyzerBase):
    """Concrete implementation of PerturbationAnalyzerBase for testing."""

    ENGINE_NAME = "stub"

    def __init__(self, nq: int = 2) -> None:
        super().__init__()
        self._nq = nq
        self._call_count = 0

    def _simulate(self, coeffs: list[list[float]]) -> _StubSimResult:
        self._call_count += 1
        return _make_stub_result(nq=self._nq)

    def _get_q_traj(self, sim_result: _StubSimResult) -> np.ndarray:
        return sim_result.q_traj

    def _get_v_traj(self, sim_result: _StubSimResult) -> np.ndarray:
        return sim_result.v_traj

    def _validate_sim_result_type(self, sim_result: object) -> None:
        if not isinstance(sim_result, _StubSimResult):
            raise ValueError(
                f"sim_result must be _StubSimResult, got {type(sim_result)}"
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def analyzer() -> StubAnalyzer:
    return StubAnalyzer()


@pytest.fixture()
def configured_analyzer(analyzer: StubAnalyzer) -> StubAnalyzer:
    analyzer.set_base_torque_profile({"coeffs": [[0.1, 0.0], [0.05, 0.0]]})
    return analyzer


@pytest.fixture()
def small_config() -> PerturbationConfig:
    return PerturbationConfig(n_trials=3, noise_amplitude=0.01, seed=42)


# ---------------------------------------------------------------------------
# Tests: MANDATORY_METRICS constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMandatoryMetrics:
    def test_contains_ten_metrics(self) -> None:
        assert len(MANDATORY_METRICS) == 10

    def test_all_expected_keys_present(self) -> None:
        expected = {
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
        assert set(MANDATORY_METRICS) == expected

    def test_array_metrics_are_subset(self) -> None:
        assert _ARRAY_METRICS.issubset(set(MANDATORY_METRICS))


# ---------------------------------------------------------------------------
# Tests: set_base_torque_profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetBaseTorqueProfile:
    def test_perturbation_base_accepts_valid_profile(
        self, analyzer: StubAnalyzer
    ) -> None:
        analyzer.set_base_torque_profile({"coeffs": [[1.0, 0.0]]})
        assert analyzer._base_coeffs == [[1.0, 0.0]]

    def test_caches_nominal_result(self, analyzer: StubAnalyzer) -> None:
        analyzer.set_base_torque_profile({"coeffs": [[1.0]]})
        assert analyzer._nominal_result is not None

    def test_raises_on_non_dict(self, analyzer: StubAnalyzer) -> None:
        with pytest.raises(ValueError, match="profile must be a dict"):
            analyzer.set_base_torque_profile([1.0, 2.0])  # type: ignore[arg-type]

    def test_raises_on_missing_coeffs_key(self, analyzer: StubAnalyzer) -> None:
        with pytest.raises(ValueError, match="'coeffs' key missing"):
            analyzer.set_base_torque_profile({"wrong_key": []})

    def test_raises_on_empty_coeffs(self, analyzer: StubAnalyzer) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            analyzer.set_base_torque_profile({"coeffs": []})

    def test_raises_on_non_list_coeffs(self, analyzer: StubAnalyzer) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            analyzer.set_base_torque_profile({"coeffs": "not_a_list"})


# ---------------------------------------------------------------------------
# Tests: perturb_torque
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerturbTorque:
    def test_raises_before_profile_set(self, analyzer: StubAnalyzer) -> None:
        cfg = PerturbationConfig(n_trials=1)
        with pytest.raises(ValueError, match="set_base_torque_profile"):
            analyzer.perturb_torque(cfg, seed=0)

    def test_perturbation_base_returns_dict_with_coeffs(
        self, configured_analyzer: StubAnalyzer, small_config: PerturbationConfig
    ) -> None:
        result = configured_analyzer.perturb_torque(small_config, seed=0)
        assert isinstance(result, dict)
        assert "coeffs" in result

    def test_same_seed_gives_same_result(
        self, configured_analyzer: StubAnalyzer, small_config: PerturbationConfig
    ) -> None:
        r1 = configured_analyzer.perturb_torque(small_config, seed=7)
        r2 = configured_analyzer.perturb_torque(small_config, seed=7)
        assert r1["coeffs"] == r2["coeffs"]

    def test_perturbation_base_different_seeds_differ(
        self, configured_analyzer: StubAnalyzer, small_config: PerturbationConfig
    ) -> None:
        r1 = configured_analyzer.perturb_torque(small_config, seed=1)
        r2 = configured_analyzer.perturb_torque(small_config, seed=99)
        # At least one coefficient should differ
        flat_a = [c for sub in r1["coeffs"] for c in sub]
        flat_b = [c for sub in r2["coeffs"] for c in sub]
        assert flat_a != flat_b


# ---------------------------------------------------------------------------
# Tests: extract_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractMetrics:
    def test_all_mandatory_keys_present(
        self, configured_analyzer: StubAnalyzer
    ) -> None:
        result = _make_stub_result()
        metrics = configured_analyzer.extract_metrics(result)
        for key in MANDATORY_METRICS:
            assert key in metrics, f"Missing metric: {key}"

    def test_scalar_values_are_finite(self, configured_analyzer: StubAnalyzer) -> None:
        result = _make_stub_result()
        metrics = configured_analyzer.extract_metrics(result)
        scalar_keys = [k for k in MANDATORY_METRICS if k not in _ARRAY_METRICS]
        for key in scalar_keys:
            assert np.isfinite(metrics[key]), f"Non-finite value for {key}"

    def test_raises_on_wrong_type(self, configured_analyzer: StubAnalyzer) -> None:
        with pytest.raises(ValueError, match="_StubSimResult"):
            configured_analyzer.extract_metrics(object())  # type: ignore[arg-type]

    def test_raises_on_single_step(self, configured_analyzer: StubAnalyzer) -> None:
        result = _make_stub_result(n=1)
        with pytest.raises(ValueError, match=">= 2 steps"):
            configured_analyzer.extract_metrics(result)

    def test_motion_duration_is_positive(
        self, configured_analyzer: StubAnalyzer
    ) -> None:
        result = _make_stub_result(n=5)
        metrics = configured_analyzer.extract_metrics(result)
        assert metrics["motion_duration"] > 0.0

    def test_trajectory_rmse_zero_for_nominal(
        self, configured_analyzer: StubAnalyzer
    ) -> None:
        # When nominal is the same as result, RMSE should be 0
        nominal = configured_analyzer._nominal_result
        assert nominal is not None
        metrics = configured_analyzer.extract_metrics(nominal)
        assert metrics["trajectory_rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_total_energy_is_ke_plus_pe(
        self, configured_analyzer: StubAnalyzer
    ) -> None:
        result = _make_stub_result(n=5)
        metrics = configured_analyzer.extract_metrics(result)
        last = result.n_steps - 1
        expected = float(
            result.kinetic_energy_traj[last] + result.potential_energy_traj[last]
        )
        assert metrics["total_energy_final"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests: run_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunBatch:
    def test_raises_before_profile_set(
        self, analyzer: StubAnalyzer, small_config: PerturbationConfig
    ) -> None:
        with pytest.raises(ValueError, match="set_base_torque_profile"):
            analyzer.run_batch(small_config)

    def test_perturbation_base_returns_perturbation_summary(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        from src.shared.python.perturbation.config import PerturbationSummary

        summary = configured_analyzer.run_batch(small_config)
        assert isinstance(summary, PerturbationSummary)

    def test_success_rate_in_unit_interval(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        summary = configured_analyzer.run_batch(small_config)
        assert 0.0 <= summary.success_rate <= 1.0

    def test_robustness_score_in_unit_interval(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        summary = configured_analyzer.run_batch(small_config)
        assert 0.0 <= summary.robustness_score <= 1.0

    def test_engine_name_propagated(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        summary = configured_analyzer.run_batch(small_config)
        assert summary.engine_name == "stub"

    def test_metrics_dict_has_scalar_keys(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        summary = configured_analyzer.run_batch(small_config)
        scalar_keys = [m for m in MANDATORY_METRICS if m not in _ARRAY_METRICS]
        for key in scalar_keys:
            assert key in summary.metrics

    def test_zero_robustness_when_all_fail(
        self, small_config: PerturbationConfig
    ) -> None:
        """Analyzer whose trial _simulate calls all raise should return RS=0."""

        class _FailCounter(StubAnalyzer):
            """First call (nominal) succeeds; all subsequent calls raise."""

            def __init__(self) -> None:
                super().__init__()
                self._call_count = 0

            def _simulate(self, coeffs: list[list[float]]) -> _StubSimResult:
                self._call_count += 1
                if self._call_count > 1:
                    raise RuntimeError("forced trial failure")
                return _make_stub_result()

        fa = _FailCounter()
        fa.set_base_torque_profile({"coeffs": [[0.1]]})
        summary = fa.run_batch(small_config)
        assert summary.robustness_score == 0.0
        assert summary.success_rate == 0.0


# ---------------------------------------------------------------------------
# Tests: compare_profiles
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompareProfiles:
    def test_returns_comparison_report(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        profile_a = {"coeffs": [[0.1, 0.0], [0.05, 0.0]]}
        profile_b = {"coeffs": [[0.2, 0.0], [0.10, 0.0]]}
        report = configured_analyzer.compare_profiles(
            profile_a, profile_b, small_config
        )
        assert isinstance(report, ComparisonReport)

    def test_winner_is_a_or_b(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        profile_a = {"coeffs": [[0.1]]}
        profile_b = {"coeffs": [[0.2]]}
        report = configured_analyzer.compare_profiles(
            profile_a, profile_b, small_config, name_a="A", name_b="B"
        )
        assert report.winner in ("A", "B")

    def test_confidence_in_unit_interval(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        profile_a = {"coeffs": [[0.1]]}
        profile_b = {"coeffs": [[0.2]]}
        report = configured_analyzer.compare_profiles(
            profile_a, profile_b, small_config
        )
        assert 0.0 <= report.confidence <= 1.0

    def test_pvalues_keys_are_scalar_metrics(
        self,
        configured_analyzer: StubAnalyzer,
        small_config: PerturbationConfig,
    ) -> None:
        profile_a = {"coeffs": [[0.1]]}
        profile_b = {"coeffs": [[0.1]]}
        report = configured_analyzer.compare_profiles(
            profile_a, profile_b, small_config
        )
        expected_metrics = {
            "end_effector_speed_final",
            "peak_end_effector_speed",
            "total_energy_final",
            "trajectory_rmse",
            "trajectory_max_deviation",
        }
        assert set(report.pvalues.keys()) == expected_metrics


# ---------------------------------------------------------------------------
# Tests: ComparisonReport dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComparisonReport:
    def test_perturbation_base_default_fields(self) -> None:
        report = ComparisonReport(winner="A", confidence=0.9)
        assert report.metric_comparisons == {}
        assert report.pvalues == {}

    def test_winner_stored(self) -> None:
        report = ComparisonReport(winner="B", confidence=0.5)
        assert report.winner == "B"
        assert report.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: _compute_cv_values helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeCvValues:
    def test_empty_dict_returns_empty_list(self) -> None:
        assert _compute_cv_values({}) == []

    def _make_stats(
        self, mean: float, std: float, median: float = 0.0
    ) -> MetricStatistics:
        """Helper: build a MetricStatistics with the real dataclass fields."""
        cv = std / abs(mean) if abs(mean) > 1e-12 else 0.0
        return MetricStatistics(
            mean=mean,
            std=std,
            cv=cv,
            min_val=mean - std,
            max_val=mean + std,
            median=median or mean,
            iqr=std,
            p5=mean - 2 * std,
            p95=mean + 2 * std,
        )

    def test_zero_std_gives_zero_cv(self) -> None:
        stats = self._make_stats(mean=1.0, std=0.0)
        cvs = _compute_cv_values({"m": stats})
        assert cvs == [0.0]

    def test_zero_mean_gives_zero_cv(self) -> None:
        stats = self._make_stats(mean=0.0, std=0.1)
        cvs = _compute_cv_values({"m": stats})
        assert cvs == [0.0]

    def test_nonzero_cv(self) -> None:
        stats = self._make_stats(mean=2.0, std=0.4)
        cvs = _compute_cv_values({"m": stats})
        assert cvs == [pytest.approx(0.4 / 2.0)]

    def test_multiple_entries(self) -> None:
        s1 = self._make_stats(mean=1.0, std=0.1)
        s2 = self._make_stats(mean=0.0, std=0.0)
        cvs = _compute_cv_values({"a": s1, "b": s2})
        assert len(cvs) == 2


# ---------------------------------------------------------------------------
# Tests: subclass ENGINE_NAME propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEngineNamePropagation:
    def test_engine_name_from_subclass(
        self, configured_analyzer: StubAnalyzer, small_config: PerturbationConfig
    ) -> None:
        summary = configured_analyzer.run_batch(small_config)
        assert summary.engine_name == StubAnalyzer.ENGINE_NAME

    def test_different_engine_names(self) -> None:
        class AlphaAnalyzer(StubAnalyzer):
            ENGINE_NAME = "alpha"

        class BetaAnalyzer(StubAnalyzer):
            ENGINE_NAME = "beta"

        assert AlphaAnalyzer.ENGINE_NAME != BetaAnalyzer.ENGINE_NAME


# ---------------------------------------------------------------------------
# Tests: abstract method contract enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAbstractMethods:
    def test_cannot_instantiate_base_directly(self) -> None:
        with pytest.raises(TypeError):
            PerturbationAnalyzerBase()  # type: ignore[abstract]

    def test_partial_implementation_raises(self) -> None:
        """A class missing any abstract method should raise TypeError."""

        class Partial(PerturbationAnalyzerBase):
            def _simulate(self, coeffs) -> _StubSimResult:  # type: ignore[override]
                return _make_stub_result()

            def _get_q_traj(self, r) -> np.ndarray:  # type: ignore[override]
                return r.q_traj

            # Missing _get_v_traj and _validate_sim_result_type

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]
