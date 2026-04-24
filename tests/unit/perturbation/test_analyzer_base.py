from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.perturbation.analyzer_base import (
    MANDATORY_METRICS,
    ComparisonReport,
    PartialResultsError,
    PartialResultsWarning,
    PerturbationAnalyzerBase,
)
from src.shared.python.perturbation.config import PerturbationConfig


class _RecordingAnalyzer(PerturbationAnalyzerBase):
    ENGINE_NAME = "recording"

    def __init__(self, failure_trials: set[int] | None = None) -> None:
        self._failure_trials = failure_trials or set()
        self._call_count = 0
        self._base_coeffs = [[0.0, 0.0]]

    def _simulate(self, coeffs: list[list[float]]) -> float:
        trial_index = self._call_count
        self._call_count += 1
        if trial_index in self._failure_trials:
            raise RuntimeError(f"boom-{trial_index}")
        return float(trial_index + sum(sum(joint) for joint in coeffs))

    def extract_metrics(self, sim_result: object) -> dict[str, float | np.ndarray]:
        if not isinstance(sim_result, float):
            raise TypeError("expected float simulation result")
        scalar = float(sim_result)
        vector = np.array([scalar, scalar + 1.0, scalar + 2.0])
        return {
            "end_effector_position_final": vector,
            "end_effector_velocity_final": vector + 0.5,
            "end_effector_speed_final": scalar + 0.1,
            "peak_end_effector_speed": scalar + 0.2,
            "total_energy_final": scalar + 0.3,
            "joint_angles_final": np.array([scalar, scalar + 0.4]),
            "joint_velocities_final": np.array([scalar + 0.5, scalar + 0.6]),
            "trajectory_rmse": scalar + 0.7,
            "trajectory_max_deviation": scalar + 0.8,
            "motion_duration": scalar + 0.9,
        }

    def set_base_torque_profile(self, profile: object) -> None:
        if not isinstance(profile, dict) or "coeffs" not in profile:
            raise ValueError("profile must be a dict with coeffs")
        self._base_coeffs = profile["coeffs"]


class TestRunBatchPartialResults:
    def test_run_batch_records_trial_failures(self) -> None:
        analyzer = _RecordingAnalyzer(failure_trials={1})
        config = PerturbationConfig(
            n_trials=3,
            noise_amplitude=0.0,
            seed=7,
            min_success_rate=0.9,
        )

        with pytest.warns(PartialResultsWarning, match="run_batch"):
            summary = analyzer.run_batch(config)

        assert summary.engine_name == "recording"
        assert summary.success_rate == pytest.approx(2 / 3)
        assert set(summary.metrics) == {
            metric
            for metric in MANDATORY_METRICS
            if metric
            not in (
                "end_effector_position_final",
                "end_effector_velocity_final",
                "joint_angles_final",
                "joint_velocities_final",
            )
        }
        assert len(summary.failures) == 1
        failure = summary.failures[0]
        assert failure.trial_index == 1
        assert failure.seed == 8
        assert failure.stage == "run_batch"
        assert failure.error_type == "RuntimeError"
        assert failure.message == "boom-1"

    def test_run_batch_can_raise_for_partial_results(self) -> None:
        analyzer = _RecordingAnalyzer(failure_trials={0})
        config = PerturbationConfig(
            n_trials=2,
            noise_amplitude=0.0,
            min_success_rate=1.0,
            raise_on_partial_results=True,
        )

        with pytest.raises(PartialResultsError, match="run_batch") as exc_info:
            analyzer.run_batch(config)

        assert exc_info.value.success_rate == pytest.approx(0.5)
        assert exc_info.value.threshold == pytest.approx(1.0)
        assert len(exc_info.value.failures) == 1


class TestCompareProfilesPartialResults:
    def test_compare_profiles_surfaces_trial_failures(self) -> None:
        analyzer = _RecordingAnalyzer(failure_trials={0})
        config = PerturbationConfig(
            n_trials=1,
            noise_amplitude=0.0,
            min_success_rate=1.0,
        )

        with pytest.warns(PartialResultsWarning, match="compare_profiles"):
            report = analyzer.compare_profiles(
                {"coeffs": [[0.0, 0.0]]},
                {"coeffs": [[0.0, 0.0]]},
                config,
            )

        assert isinstance(report, ComparisonReport)
        assert len(report.failures) == 1
        assert report.failures[0].stage == "compare_profiles:A:end_effector_speed_final"
        assert report.metric_comparisons["end_effector_speed_final"]["failures_a"] == 1
        assert report.metric_comparisons["end_effector_speed_final"]["failures_b"] == 0
