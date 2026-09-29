"""Tests for src.shared.python.analysis.swing_metrics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.analysis.swing_metrics import SwingMetricsMixin


def _make_instance(n_samples: int = 100, n_joints: int = 3) -> SwingMetricsMixin:
    """Create a concrete instance of the mixin with synthetic data."""

    class _Concrete(SwingMetricsMixin):
        pass

    obj = _Concrete()
    t = np.linspace(0.0, 2.0, n_samples)
    obj.times = t
    obj.dt = t[1] - t[0]
    # Sinusoidal joint angles (radians)
    obj.joint_positions = np.column_stack(
        [np.sin(2 * np.pi * (i + 1) / 2.0 * t) for i in range(n_joints)]
    )
    # Simulate club head speed: ramp up to a peak then drop
    mid = n_samples // 2
    obj.club_head_speed = np.concatenate(
        [np.linspace(0, 10, mid), np.linspace(10, 2, n_samples - mid)]
    )
    return obj


class TestComputeRangeOfMotion:
    def test_swing_metrics_returns_tuple_of_three(self) -> None:
        obj = _make_instance()
        result = obj.compute_range_of_motion(0)
        assert len(result) == 3

    def test_rom_non_negative(self) -> None:
        obj = _make_instance()
        _, _, rom = obj.compute_range_of_motion(0)
        assert rom >= 0.0

    def test_max_angle_ge_min_angle(self) -> None:
        obj = _make_instance()
        min_angle, max_angle, _ = obj.compute_range_of_motion(0)
        assert max_angle >= min_angle

    def test_values_are_degrees(self) -> None:
        obj = _make_instance()
        # sin(t) has range [-1, 1] radians = [-57.3, 57.3] degrees
        min_angle, max_angle, rom = obj.compute_range_of_motion(0)
        # Should be well above 1 (i.e., in degrees not radians)
        assert rom > 2.0

    def test_out_of_range_returns_zeros(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_range_of_motion(5)
        assert result == (0.0, 0.0, 0.0)


class TestComputeTempo:
    def test_returns_tuple_or_none(self) -> None:
        obj = _make_instance(n_samples=100)
        result = obj.compute_tempo()
        assert result is None or len(result) == 3

    def test_swing_metrics_durations_non_negative(self) -> None:
        obj = _make_instance(n_samples=200)
        result = obj.compute_tempo()
        if result is not None:
            backswing, downswing, ratio = result
            assert backswing >= 0.0
            assert downswing >= 0.0
            assert ratio >= 0.0

    def test_no_club_speed_returns_none(self) -> None:
        obj = _make_instance()
        obj.club_head_speed = None
        result = obj.compute_tempo()
        assert result is None

    def test_short_club_speed_returns_none(self) -> None:
        obj = _make_instance()
        obj.club_head_speed = np.array([1.0, 2.0, 3.0])
        result = obj.compute_tempo()
        assert result is None


class TestComputeXFactor:
    def test_swing_metrics_returns_array(self) -> None:
        obj = _make_instance(n_joints=3)
        result = obj.compute_x_factor(0, 1)
        assert isinstance(result, np.ndarray)

    def test_same_length_as_times(self) -> None:
        obj = _make_instance(n_joints=3)
        result = obj.compute_x_factor(0, 1)
        assert result is not None
        assert len(result) == len(obj.times)

    def test_swing_metrics_out_of_range_returns_none(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_x_factor(0, 5)
        assert result is None

    def test_values_are_degrees(self) -> None:
        obj = _make_instance(n_joints=3)
        result = obj.compute_x_factor(0, 1)
        assert result is not None
        # Amplitudes should be up to ~57 degrees (rad2deg of 1 rad)
        assert np.max(np.abs(result)) > 1.0


class TestComputeXFactorStretch:
    def test_swing_metrics_returns_tuple(self) -> None:
        obj = _make_instance(n_joints=3)
        result = obj.compute_x_factor_stretch(0, 1)
        assert result is not None
        assert len(result) == 2

    def test_peak_stretch_rate_non_negative(self) -> None:
        obj = _make_instance(n_joints=3)
        result = obj.compute_x_factor_stretch(0, 1)
        assert result is not None
        _, peak = result
        assert peak >= 0.0

    def test_swing_metrics_out_of_range_returns_none(self) -> None:
        obj = _make_instance(n_joints=2)
        result = obj.compute_x_factor_stretch(0, 5)
        assert result is None
