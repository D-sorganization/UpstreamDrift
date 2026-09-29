"""Tests for src.shared.python.signal_toolkit.limits (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.signal_toolkit.core import Signal
from src.shared.python.signal_toolkit.limits import (
    SaturationMode,
    apply_deadband,
    apply_rate_limiter,
    apply_saturation,
)


def _make_signal(amp: float = 2.0, n: int = 100) -> Signal:
    t = np.linspace(0.0, 1.0, n)
    y = amp * np.sin(2 * np.pi * t)
    return Signal(time=t, values=y, name="test", units="m")


class TestApplySaturation:
    def test_limits_returns_signal(self) -> None:
        sig = _make_signal()
        result = apply_saturation(sig, lower=-1.0, upper=1.0)
        assert isinstance(result, Signal)

    def test_hard_clip_values_within_limits(self) -> None:
        sig = _make_signal(amp=5.0)
        result = apply_saturation(sig, lower=-1.0, upper=1.0, mode=SaturationMode.HARD)
        assert np.all(result.values >= -1.0)
        assert np.all(result.values <= 1.0)

    def test_tanh_mode_values_within_limits(self) -> None:
        sig = _make_signal(amp=5.0)
        result = apply_saturation(sig, lower=-1.0, upper=1.0, mode=SaturationMode.TANH)
        assert np.all(result.values >= -1.0 - 1e-10)
        assert np.all(result.values <= 1.0 + 1e-10)

    def test_sigmoid_mode_runs(self) -> None:
        sig = _make_signal(amp=3.0)
        result = apply_saturation(
            sig, lower=-2.0, upper=2.0, mode=SaturationMode.SIGMOID
        )
        assert isinstance(result, Signal)

    def test_cubic_mode_runs(self) -> None:
        sig = _make_signal(amp=3.0)
        result = apply_saturation(sig, lower=-2.0, upper=2.0, mode=SaturationMode.CUBIC)
        assert isinstance(result, Signal)

    def test_lower_gte_upper_raises(self) -> None:
        sig = _make_signal()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            apply_saturation(sig, lower=1.0, upper=-1.0)

    def test_lower_equals_upper_raises(self) -> None:
        sig = _make_signal()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            apply_saturation(sig, lower=0.0, upper=0.0)

    def test_limits_name_updated(self) -> None:
        sig = _make_signal()
        result = apply_saturation(sig, lower=-1.0, upper=1.0)
        assert "saturated" in result.name

    def test_small_signal_unchanged(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = 0.5 * np.ones(100)  # all within [-1, 1]
        sig = Signal(time=t, values=y)
        result = apply_saturation(sig, lower=-1.0, upper=1.0, mode=SaturationMode.HARD)
        np.testing.assert_allclose(result.values, 0.5, atol=1e-12)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=150)
        result = apply_saturation(sig, lower=-1.0, upper=1.0)
        assert len(result.values) == 150


class TestApplyRateLimiter:
    def test_limits_returns_signal(self) -> None:
        sig = _make_signal()
        result = apply_rate_limiter(sig, max_rate=10.0)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=100)
        result = apply_rate_limiter(sig, max_rate=10.0)
        assert len(result.values) == 100

    def test_rate_limited_signal_changes_slower(self) -> None:
        # Create a step signal that changes instantaneously
        t = np.linspace(0.0, 1.0, 201)
        y = np.zeros(201)
        y[100:] = 10.0  # step at t=0.5
        sig = Signal(time=t, values=y)
        result = apply_rate_limiter(sig, max_rate=1.0, smooth_transition=False)
        # The maximum rate of change should be approximately limited
        diffs = np.abs(np.diff(result.values))
        dt = sig.dt
        # Should be approximately 1.0 * dt per step
        assert np.max(diffs) <= 1.0 * dt + 1e-10

    def test_negative_max_rate_raises(self) -> None:
        sig = _make_signal()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            apply_rate_limiter(sig, max_rate=-5.0)

    def test_zero_max_rate_raises(self) -> None:
        sig = _make_signal()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            apply_rate_limiter(sig, max_rate=0.0)

    def test_first_value_preserved(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = np.ones(100) * 7.0
        sig = Signal(time=t, values=y)
        result = apply_rate_limiter(sig, max_rate=5.0)
        assert result.values[0] == pytest.approx(7.0)

    def test_limits_name_updated(self) -> None:
        sig = _make_signal()
        result = apply_rate_limiter(sig, max_rate=10.0)
        assert "rate_limited" in result.name


class TestApplyDeadband:
    def test_limits_returns_signal(self) -> None:
        sig = _make_signal()
        result = apply_deadband(sig, threshold=0.5)
        assert isinstance(result, Signal)

    def test_output_length_preserved(self) -> None:
        sig = _make_signal(n=80)
        result = apply_deadband(sig, threshold=0.1)
        assert len(result.values) == 80

    def test_values_within_deadband_zeroed(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        y = np.ones(100) * 0.05  # all below threshold of 0.1
        sig = Signal(time=t, values=y)
        result = apply_deadband(sig, threshold=0.1, center=0.0, smooth=False)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-12)

    def test_limits_all_values_finite(self) -> None:
        sig = _make_signal(amp=3.0)
        result = apply_deadband(sig, threshold=0.5)
        assert np.all(np.isfinite(result.values))
