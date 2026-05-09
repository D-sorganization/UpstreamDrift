"""Tests for src.shared.python.analysis.coordination_metrics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.coordination_metrics import CoordinationMetricsMixin


def _make_instance(n_samples: int = 50, n_joints: int = 3) -> CoordinationMetricsMixin:
    """Create a concrete instance of the mixin with synthetic data."""

    class _Concrete(CoordinationMetricsMixin):
        pass

    obj = _Concrete()
    obj.times = np.linspace(0.0, 1.0, n_samples)
    obj.dt = obj.times[1] - obj.times[0]
    # Sinusoidal joint motion for testing
    t = obj.times
    obj.joint_positions = np.column_stack(
        [np.sin(2 * np.pi * (i + 1) * t) for i in range(n_joints)]
    )
    obj.joint_velocities = np.column_stack(
        [2 * np.pi * (i + 1) * np.cos(2 * np.pi * (i + 1) * t) for i in range(n_joints)]
    )
    obj.joint_torques = np.column_stack(
        [0.1 * np.sin(2 * np.pi * (i + 1) * t) for i in range(n_joints)]
    )
    return obj


class TestComputeCouplingAngles:
    def test_coordination_metrics_returns_array(self) -> None:
        obj = _make_instance()
        result = obj.compute_coupling_angles(0, 1)
        assert isinstance(result, np.ndarray)

    def test_values_in_0_360(self) -> None:
        obj = _make_instance()
        result = obj.compute_coupling_angles(0, 1)
        assert np.all(result >= 0.0)
        assert np.all(result < 360.0)

    def test_out_of_range_index_returns_empty(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_coupling_angles(0, 5)
        assert len(result) == 0

    def test_same_length_as_times(self) -> None:
        obj = _make_instance()
        result = obj.compute_coupling_angles(0, 1)
        assert len(result) == len(obj.times)


class TestComputeCoordinationMetrics:
    def test_returns_object(self) -> None:
        obj = _make_instance()
        result = obj.compute_coordination_metrics(0, 1)
        assert result is not None

    def test_percentages_sum_to_100(self) -> None:
        obj = _make_instance()
        result = obj.compute_coordination_metrics(0, 1)
        assert result is not None
        pct_sum = (
            result.in_phase_pct
            + result.anti_phase_pct
            + result.proximal_leading_pct
            + result.distal_leading_pct
        )
        assert pct_sum == pytest.approx(100.0, abs=1e-4)

    def test_mean_coupling_angle_in_range(self) -> None:
        obj = _make_instance()
        result = obj.compute_coordination_metrics(0, 1)
        assert result is not None
        assert 0.0 <= result.mean_coupling_angle < 360.0

    def test_coordination_variability_non_negative(self) -> None:
        obj = _make_instance()
        result = obj.compute_coordination_metrics(0, 1)
        assert result is not None
        assert result.coordination_variability >= 0.0

    def test_coordination_metrics_out_of_range_returns_none(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_coordination_metrics(0, 5)
        assert result is None


class TestComputePhaseAngle:
    def test_coordination_metrics_returns_array(self) -> None:
        obj = _make_instance()
        result = obj.compute_phase_angle(0)
        assert isinstance(result, np.ndarray)

    def test_coordination_metrics_out_of_range_returns_empty(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_phase_angle(5)
        assert len(result) == 0


class TestComputeContinuousRelativePhase:
    def test_coordination_metrics_returns_array(self) -> None:
        obj = _make_instance()
        result = obj.compute_continuous_relative_phase(0, 1)
        assert isinstance(result, np.ndarray)

    def test_coordination_metrics_out_of_range_returns_empty(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_continuous_relative_phase(0, 5)
        assert len(result) == 0


class TestComputeCorrelations:
    def test_coordination_metrics_returns_tuple(self) -> None:
        obj = _make_instance()
        matrix, labels = obj.compute_correlations("velocity")
        assert isinstance(matrix, np.ndarray)
        assert isinstance(labels, list)

    def test_matrix_is_square(self) -> None:
        obj = _make_instance(n_joints=3)
        matrix, labels = obj.compute_correlations("velocity")
        assert matrix.shape[0] == matrix.shape[1]

    def test_labels_match_matrix(self) -> None:
        obj = _make_instance(n_joints=3)
        matrix, labels = obj.compute_correlations("velocity")
        assert len(labels) == matrix.shape[0]

    def test_diagonal_is_one(self) -> None:
        obj = _make_instance(n_joints=3)
        matrix, _ = obj.compute_correlations("velocity")
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-10)

    def test_position_data_type(self) -> None:
        obj = _make_instance()
        matrix, labels = obj.compute_correlations("position")
        assert matrix.shape[0] > 0

    def test_torque_data_type(self) -> None:
        obj = _make_instance()
        matrix, labels = obj.compute_correlations("torque")
        assert matrix.shape[0] > 0


class TestComputeRollingCorrelation:
    def test_coordination_metrics_returns_tuple(self) -> None:
        obj = _make_instance(n_samples=100)
        times, corrs = obj.compute_rolling_correlation(0, 1, window_size=10)
        assert isinstance(times, np.ndarray)
        assert isinstance(corrs, np.ndarray)

    def test_same_length(self) -> None:
        obj = _make_instance(n_samples=100)
        times, corrs = obj.compute_rolling_correlation(0, 1, window_size=10)
        assert len(times) == len(corrs)

    def test_values_in_minus1_to_1(self) -> None:
        obj = _make_instance(n_samples=100)
        _, corrs = obj.compute_rolling_correlation(0, 1, window_size=10)
        if len(corrs) > 0:
            assert np.all(corrs >= -1.0 - 1e-6)
            assert np.all(corrs <= 1.0 + 1e-6)

    def test_small_window_raises(self) -> None:
        obj = _make_instance(n_samples=100)
        with pytest.raises((ValueError, TypeError, AssertionError)):
            obj.compute_rolling_correlation(0, 1, window_size=1)
