"""Tests for src.shared.python.analysis.power_work_metrics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.analysis.power_work_metrics import PowerWorkMetricsMixin


def _make_instance(n_samples: int = 50, n_joints: int = 3) -> PowerWorkMetricsMixin:
    """Create a concrete instance of the mixin with synthetic data."""

    class _Concrete(PowerWorkMetricsMixin):
        pass

    obj = _Concrete()
    t = np.linspace(0.0, 1.0, n_samples)
    obj.times = t
    obj.dt = t[1] - t[0]
    obj.joint_positions = np.column_stack(
        [np.sin(2 * np.pi * (i + 1) * t) for i in range(n_joints)]
    )
    obj.joint_velocities = np.column_stack(
        [2 * np.pi * (i + 1) * np.cos(2 * np.pi * (i + 1) * t) for i in range(n_joints)]
    )
    obj.joint_torques = np.column_stack(
        [np.sin(2 * np.pi * (i + 1) * t) for i in range(n_joints)]
    )
    obj.ground_forces = None
    obj._work_metrics_cache = {}
    return obj


class TestComputeWorkMetrics:
    def test_power_work_metrics_returns_dict(self) -> None:
        obj = _make_instance()
        result = obj.compute_work_metrics(0)
        assert isinstance(result, dict)

    def test_power_work_metrics_has_expected_keys(self) -> None:
        obj = _make_instance()
        result = obj.compute_work_metrics(0)
        assert result is not None
        assert "positive_work" in result
        assert "negative_work" in result
        assert "net_work" in result

    def test_positive_work_non_negative(self) -> None:
        obj = _make_instance()
        result = obj.compute_work_metrics(0)
        assert result is not None
        assert result["positive_work"] >= 0.0

    def test_negative_work_non_positive(self) -> None:
        obj = _make_instance()
        result = obj.compute_work_metrics(0)
        assert result is not None
        assert result["negative_work"] <= 0.0

    def test_net_work_finite(self) -> None:
        obj = _make_instance()
        result = obj.compute_work_metrics(0)
        assert result is not None
        assert np.isfinite(result["net_work"])

    def test_power_work_metrics_out_of_range_returns_none(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_work_metrics(5)
        assert result is None

    def test_caching_returns_same_result(self) -> None:
        obj = _make_instance()
        result1 = obj.compute_work_metrics(0)
        result2 = obj.compute_work_metrics(0)
        assert result1 is result2  # same object from cache


class TestComputeJointPowerMetrics:
    def test_returns_object(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_power_metrics(0)
        assert result is not None

    def test_peak_generation_non_negative(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_power_metrics(0)
        assert result is not None
        assert result.peak_generation >= 0.0

    def test_peak_absorption_non_positive(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_power_metrics(0)
        assert result is not None
        assert result.peak_absorption <= 0.0

    def test_power_work_metrics_durations_non_negative(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_power_metrics(0)
        assert result is not None
        assert result.generation_duration >= 0.0
        assert result.absorption_duration >= 0.0

    def test_net_work_finite(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_power_metrics(0)
        assert result is not None
        assert np.isfinite(result.net_work)

    def test_power_work_metrics_out_of_range_returns_none(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_joint_power_metrics(5)
        assert result is None


class TestComputeImpulseMetrics:
    def test_returns_object_for_torque(self) -> None:
        obj = _make_instance()
        result = obj.compute_impulse_metrics("torque", 0)
        assert result is not None

    def test_net_impulse_finite(self) -> None:
        obj = _make_instance()
        result = obj.compute_impulse_metrics("torque", 0)
        assert result is not None
        assert np.isfinite(result.net_impulse)

    def test_positive_impulse_non_negative(self) -> None:
        obj = _make_instance()
        result = obj.compute_impulse_metrics("torque", 0)
        assert result is not None
        assert result.positive_impulse >= 0.0

    def test_negative_impulse_non_positive(self) -> None:
        obj = _make_instance()
        result = obj.compute_impulse_metrics("torque", 0)
        assert result is not None
        assert result.negative_impulse <= 0.0

    def test_force_without_ground_forces_returns_none(self) -> None:
        obj = _make_instance()
        # ground_forces is None
        result = obj.compute_impulse_metrics("force", 0)
        assert result is None

    def test_unknown_type_returns_none(self) -> None:
        obj = _make_instance()
        result = obj.compute_impulse_metrics("unknown", 0)
        assert result is None


class TestComputePhaseSpacePathLength:
    def test_power_work_metrics_returns_float(self) -> None:
        obj = _make_instance()
        result = obj.compute_phase_space_path_length(0)
        assert isinstance(result, float)

    def test_non_negative(self) -> None:
        obj = _make_instance()
        result = obj.compute_phase_space_path_length(0)
        assert result >= 0.0

    def test_out_of_range_returns_zero(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_phase_space_path_length(5)
        assert result == 0.0

    def test_nonzero_for_sinusoidal_motion(self) -> None:
        obj = _make_instance(n_samples=100)
        result = obj.compute_phase_space_path_length(0)
        assert result > 0.0


class TestComputeJointStiffness:
    def test_returns_object(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_stiffness(0)
        assert result is not None

    def test_has_stiffness_attribute(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_stiffness(0)
        assert result is not None
        assert np.isfinite(result.stiffness)

    def test_r_squared_in_0_1(self) -> None:
        obj = _make_instance()
        result = obj.compute_joint_stiffness(0)
        assert result is not None
        assert 0.0 <= result.r_squared <= 1.0

    def test_power_work_metrics_out_of_range_returns_none(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_joint_stiffness(5)
        assert result is None


class TestComputeDynamicStiffness:
    def test_power_work_metrics_returns_tuple_of_three(self) -> None:
        obj = _make_instance(n_samples=100)
        result = obj.compute_dynamic_stiffness(0, window_size=10)
        assert len(result) == 3

    def test_arrays_same_length(self) -> None:
        obj = _make_instance(n_samples=100)
        times, stiffness, r2 = obj.compute_dynamic_stiffness(0, window_size=10)
        assert len(times) == len(stiffness) == len(r2)

    def test_finite_stiffness_values(self) -> None:
        obj = _make_instance(n_samples=100)
        _, stiffness, _ = obj.compute_dynamic_stiffness(0, window_size=10)
        if len(stiffness) > 0:
            assert np.all(np.isfinite(stiffness))

    def test_power_work_metrics_out_of_range_returns_empty(self) -> None:
        obj = _make_instance(n_joints=2)
        times, stiffness, r2 = obj.compute_dynamic_stiffness(5)
        assert len(times) == 0
