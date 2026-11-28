"""Tests for src.shared.python.analysis.power_work_metrics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
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

    @pytest.mark.unit
    def test_caching_returns_defensive_copies(self) -> None:
        obj = _make_instance()
        result1 = obj.compute_work_metrics(0)
        result2 = obj.compute_work_metrics(0)
        assert result1 == result2
        assert result1 is not result2
        result1["net_work"] = -999.0
        result3 = obj.compute_work_metrics(0)
        assert result3["net_work"] != -999.0


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


@pytest.mark.unit
class TestPowerWorkGuards15343:
    """FC-012 guards for hysteresis, power durations, and work-metrics cache (#15343)."""

    def test_open_hookean_cycle_reports_no_hysteresis(self) -> None:
        obj = _make_instance(n_samples=50)
        theta = np.linspace(0.0, np.pi, len(obj.times))
        k = 10.0
        obj.joint_positions[:, 0] = theta
        obj.joint_torques[:, 0] = k * theta
        metrics = obj.compute_joint_stiffness(0)
        assert metrics is not None
        assert metrics.hysteresis_area == pytest.approx(0.0, abs=1e-6)

    def test_power_durations_respect_total_interval(self) -> None:
        obj = _make_instance(n_samples=100)
        obj.dt = 0.5
        obj.joint_torques[:, 0] = 1.0
        obj.joint_velocities[:, 0] = 1.0
        metrics = obj.compute_joint_power_metrics(0)
        assert metrics is not None
        t_total = float(obj.times[-1] - obj.times[0])
        assert metrics.generation_duration <= t_total + 1e-9
        assert metrics.absorption_duration <= t_total + 1e-9
        assert (
            metrics.generation_duration + metrics.absorption_duration <= t_total + 1e-9
        )


def _circle_cycle_instance(theta_0: float) -> PowerWorkMetricsMixin:
    """One full 1 Hz circle of radius 1 rad, torque 20 N·m a quarter out of
    phase (exact loop area pi*1*20 = 62.83 J), with the documented 0.03 deg
    end-of-cycle angle jitter that #15864 must tolerate regardless of where
    theta is zeroed."""
    obj = _make_instance(n_samples=200)
    phi = 2.0 * np.pi * obj.times
    obj.joint_positions[:, 0] = theta_0 + np.sin(phi)
    obj.joint_positions[-1, 0] += np.deg2rad(0.03)  # closure jitter
    obj.joint_torques[:, 0] = 20.0 * np.cos(phi)
    return obj


@pytest.mark.unit
class TestClosedAngleCycleOrigin15864:
    """FC-012a-2: closed-angle detection must not depend on |angles[0]| (#15864)."""

    @pytest.mark.parametrize("theta_0", [1.0, 0.0, -np.pi])
    def test_cycle_is_closed_for_any_angle_origin(self, theta_0: float) -> None:
        """Zeroing theta elsewhere must not decide whether the loop is closed.

        FC-012a-2 (#15864): the old ``np.isclose(a0, aN, rtol)`` compared
        against ``rtol*|angles[0]|``, so a trace starting at 0 rad failed
        closure under 0.03 deg jitter while the same trace at 1 rad passed.
        """
        from src.shared.python.analysis.power_work_metrics import _closed_angle_cycle

        obj = _circle_cycle_instance(theta_0)
        assert _closed_angle_cycle(obj.joint_positions[:, 0])

    def test_hysteresis_work_is_invariant_under_angle_origin_shift(self) -> None:
        """The 62.82 J closed-cycle work is unchanged by the angle origin.

        FC-012a-2 (#15864): shifting ``theta_0`` from 1.0 rad to 0.0 rad used
        to silently drop the hysteresis loop to 0 J.
        """
        reference = _circle_cycle_instance(1.0).compute_joint_stiffness(0)
        shifted = _circle_cycle_instance(0.0).compute_joint_stiffness(0)
        assert reference is not None and shifted is not None
        # Not a zero-vs-zero coincidence: the loop really is detected.
        assert reference.hysteresis_area == pytest.approx(np.pi * 20.0, abs=0.1)
        assert shifted.hysteresis_area == pytest.approx(
            reference.hysteresis_area, abs=1e-9
        )
