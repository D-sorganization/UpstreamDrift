"""Unit tests for trial failure reporting in PerturbationAnalyzerBase (#3055).

Verifies that:
- Failed trials are collected in ``TrialFailure`` records instead of silently
  disappearing.
- A WARNING is logged when any trial fails.
- A ``PartialResultsWarning`` is raised when the failure rate exceeds the
  threshold (> 5%).
- Successful trials still return correct results in partial-failure scenarios.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.perturbation.config import PerturbationConfig
from src.shared.python.perturbation.perturbation_base import (
    FAILURE_RATE_ERROR_THRESHOLD,
    PartialResultsWarning,
    PerturbationAnalyzerBase,
    TrialFailure,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared stub infrastructure (mirrors test_perturbation_base.py)
# ---------------------------------------------------------------------------


@dataclass
class _StubSimResult:
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


class _BaseStubAnalyzer(PerturbationAnalyzerBase):
    """Minimal concrete subclass shared by all tests in this module."""

    ENGINE_NAME = "stub"

    def _simulate(self, coeffs: list[list[float]]) -> _StubSimResult:
        return _make_stub_result()

    def _get_q_traj(self, sim_result: _StubSimResult) -> np.ndarray:
        return sim_result.q_traj

    def _get_v_traj(self, sim_result: _StubSimResult) -> np.ndarray:
        return sim_result.v_traj

    def _validate_sim_result_type(self, sim_result: object) -> None:
        if not isinstance(sim_result, _StubSimResult):
            raise ValueError(
                f"sim_result must be _StubSimResult, got {type(sim_result)}"
            )


class _PartialFailAnalyzer(_BaseStubAnalyzer):
    """Fails a configurable set of trial indices.

    The first call (nominal simulation in ``set_base_torque_profile``) always
    succeeds so initialisation does not raise.
    """

    def __init__(self, fail_indices: set[int]) -> None:
        super().__init__()
        self._fail_indices = fail_indices
        self._call_count = 0

    def _simulate(self, coeffs: list[list[float]]) -> _StubSimResult:
        idx = self._call_count
        self._call_count += 1
        # call_count 0 is the nominal simulation; trial i corresponds to call i+1
        trial_index = idx - 1
        if trial_index >= 0 and trial_index in self._fail_indices:
            raise ValueError(f"injected failure for trial {trial_index}")
        return _make_stub_result()


# ---------------------------------------------------------------------------
# Tests: TrialFailure dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrialFailure:
    def test_fields_stored(self) -> None:
        tf = TrialFailure(trial_index=3, exception_type="ValueError", message="oops")
        assert tf.trial_index == 3
        assert tf.exception_type == "ValueError"
        assert tf.message == "oops"


# ---------------------------------------------------------------------------
# Tests: PartialResultsWarning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialResultsWarning:
    def test_is_user_warning_subclass(self) -> None:
        assert issubclass(PartialResultsWarning, UserWarning)


# ---------------------------------------------------------------------------
# Tests: run_batch failure reporting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunBatchFailureReporting:
    """run_batch must surface trial failures instead of silently swallowing."""

    def _analyzer_with_single_fail(self) -> _PartialFailAnalyzer:
        """Analyzer that fails trial 0 only (failure rate = 1/n_trials)."""
        a = _PartialFailAnalyzer(fail_indices={0})
        a.set_base_torque_profile({"coeffs": [[0.1, 0.0], [0.05, 0.0]]})
        return a

    def test_failed_trial_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single failed trial must produce at least one WARNING log entry."""
        import logging

        analyzer = self._analyzer_with_single_fail()
        cfg = PerturbationConfig(n_trials=5, noise_amplitude=0.01, seed=0)

        with caplog.at_level(logging.WARNING), warnings.catch_warnings():
            warnings.simplefilter("always")
            analyzer.run_batch(cfg)

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("failed" in m.lower() for m in warning_messages), (
            f"No failure WARNING found. Records: {warning_messages}"
        )

    def test_successful_trials_return_correct_results(self) -> None:
        """Partial failures must not corrupt the results of successful trials."""
        # Only trial 1 fails; trials 0, 2 succeed -> success_rate = 2/3
        analyzer = _PartialFailAnalyzer(fail_indices={1})
        analyzer.set_base_torque_profile({"coeffs": [[0.1]]})
        cfg = PerturbationConfig(n_trials=3, noise_amplitude=0.0, seed=0)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            summary = analyzer.run_batch(cfg)

        assert summary.success_rate == pytest.approx(2 / 3, abs=1e-9)
        assert 0.0 <= summary.robustness_score <= 1.0

    def test_partial_results_warning_raised_above_threshold(self) -> None:
        """When failure rate > threshold, PartialResultsWarning must be raised."""
        n_trials = 20
        # Fail 3 out of 20 -> 15% failure rate > 5% threshold
        fail_set = {0, 1, 2}
        analyzer = _PartialFailAnalyzer(fail_indices=fail_set)
        analyzer.set_base_torque_profile({"coeffs": [[0.1]]})
        cfg = PerturbationConfig(n_trials=n_trials, noise_amplitude=0.0, seed=0)

        with pytest.warns(PartialResultsWarning, match="trials failed"):
            analyzer.run_batch(cfg)

    def test_no_warning_raised_below_threshold(self) -> None:
        """When failure rate <= threshold no PartialResultsWarning is emitted."""
        n_trials = 100
        # Fail exactly 1 out of 100 -> 1% failure rate <= 5% threshold
        analyzer = _PartialFailAnalyzer(fail_indices={0})
        analyzer.set_base_torque_profile({"coeffs": [[0.1]]})
        cfg = PerturbationConfig(n_trials=n_trials, noise_amplitude=0.0, seed=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            analyzer.run_batch(cfg)

        partial_warnings = [
            w for w in caught if issubclass(w.category, PartialResultsWarning)
        ]
        assert partial_warnings == [], (
            f"Unexpected PartialResultsWarning at or below threshold: {partial_warnings}"
        )

    def test_failure_rate_threshold_value(self) -> None:
        """The threshold constant must be 5%."""
        assert pytest.approx(0.05) == FAILURE_RATE_ERROR_THRESHOLD

    def test_zero_success_returns_zero_robustness(self) -> None:
        """All trials failing -> robustness_score=0 and success_rate=0."""

        class _AllFail(_BaseStubAnalyzer):
            def __init__(self) -> None:
                super().__init__()
                self._first = True

            def _simulate(self, coeffs: list[list[float]]) -> _StubSimResult:
                if self._first:
                    self._first = False
                    return _make_stub_result()
                raise RuntimeError("always fails")

        a = _AllFail()
        a.set_base_torque_profile({"coeffs": [[0.1]]})
        cfg = PerturbationConfig(n_trials=5, noise_amplitude=0.0, seed=0)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            summary = a.run_batch(cfg)

        assert summary.robustness_score == 0.0
        assert summary.success_rate == 0.0
