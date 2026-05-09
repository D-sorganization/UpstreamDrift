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


# ---------------------------------------------------------------------------
# Tests: set_base_torque_profile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: perturb_torque
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: extract_metrics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: ComparisonReport dataclass
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: _compute_cv_values helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: subclass ENGINE_NAME propagation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: abstract method contract enforcement
# ---------------------------------------------------------------------------
