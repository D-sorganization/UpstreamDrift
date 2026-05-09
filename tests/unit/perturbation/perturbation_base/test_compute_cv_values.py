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


# ---------------------------------------------------------------------------
# Tests: compare_profiles
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: ComparisonReport dataclass
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: abstract method contract enforcement
# ---------------------------------------------------------------------------
