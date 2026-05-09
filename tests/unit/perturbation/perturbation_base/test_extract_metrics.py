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
