"""Unit tests for signal_toolkit/limits.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.core import Signal, SignalGenerator
from src.shared.python.signal_toolkit.limits import (
    SaturationMode,
    apply_backlash,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
    create_saturation_function,
    visualize_saturation_curves,
)


@pytest.fixture
def t() -> np.ndarray:
    return np.linspace(0, 4 * np.pi, 500)


@pytest.fixture
def sine_signal(t: np.ndarray) -> Signal:
    return SignalGenerator.sinusoid(t, amplitude=2.0, frequency=0.5, name="sin")


class TestApplySaturation:
    """Tests for apply_saturation function."""

    def test_hard_clipping(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.HARD
        )
        assert np.all(saturated.values >= -1.0)
        assert np.all(saturated.values <= 1.0)

    def test_tanh_saturation(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.TANH
        )
        assert np.all(saturated.values >= -1.0 - 1e-9)
        assert np.all(saturated.values <= 1.0 + 1e-9)

    def test_sigmoid_saturation(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.SIGMOID
        )
        assert saturated.values.shape == sine_signal.values.shape

    def test_atan_saturation(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.ATAN
        )
        assert saturated.values.shape == sine_signal.values.shape

    def test_soft_saturation(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.SOFT
        )
        assert saturated.values.shape == sine_signal.values.shape

    def test_cubic_saturation(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.CUBIC
        )
        assert saturated.values.shape == sine_signal.values.shape

    def test_exponential_saturation(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(
            sine_signal, lower=-1.0, upper=1.0, mode=SaturationMode.EXPONENTIAL
        )
        assert saturated.values.shape == sine_signal.values.shape

    def test_invalid_bounds_raises(self, sine_signal: Signal) -> None:
        with pytest.raises((ValueError, AssertionError)):
            apply_saturation(sine_signal, lower=1.0, upper=-1.0)

    def test_output_name_updated(self, sine_signal: Signal) -> None:
        saturated = apply_saturation(sine_signal, lower=-1.0, upper=1.0)
        assert "_saturated" in saturated.name

    def test_no_clipping_within_bounds(self, t: np.ndarray) -> None:
        """Signal within bounds should be unchanged for hard clip."""
        small_sig = Signal(time=t, values=0.5 * np.sin(t), name="small")
        result = apply_saturation(
            small_sig, lower=-1.0, upper=1.0, mode=SaturationMode.HARD
        )
        assert np.allclose(result.values, small_sig.values)


class TestApplyRateLimiter:
    """Tests for apply_rate_limiter function."""

    def test_rate_limited_output(self, sine_signal: Signal) -> None:
        limited = apply_rate_limiter(sine_signal, max_rate=1.0)
        # Compute actual rate of change
        rates = np.abs(np.diff(limited.values)) / sine_signal.dt
        assert np.all(rates <= 1.0 + 1e-6)

    def test_hard_rate_limiting(self, sine_signal: Signal) -> None:
        limited = apply_rate_limiter(sine_signal, max_rate=0.5, smooth_transition=False)
        rates = np.abs(np.diff(limited.values)) / sine_signal.dt
        assert np.all(rates <= 0.5 + 1e-6)

    def test_negative_max_rate_raises(self, sine_signal: Signal) -> None:
        with pytest.raises((ValueError, AssertionError)):
            apply_rate_limiter(sine_signal, max_rate=-1.0)

    def test_name_updated(self, sine_signal: Signal) -> None:
        limited = apply_rate_limiter(sine_signal, max_rate=5.0)
        assert "_rate_limited" in limited.name


class TestApplyDeadband:
    """Tests for apply_deadband function."""

    def test_hard_deadband(self, t: np.ndarray) -> None:
        """Values inside deadband → 0."""
        small_sig = Signal(time=t, values=0.3 * np.sin(t), name="s")
        result = apply_deadband(small_sig, threshold=0.5, center=0.0, smooth=False)
        # All values within ±0.3, so all should map to center
        assert np.all(np.abs(result.values) < 1e-6)

    def test_smooth_deadband(self, t: np.ndarray) -> None:
        sig = Signal(time=t, values=2.0 * np.sin(t), name="s")
        result = apply_deadband(sig, threshold=0.5, smooth=True)
        assert result.values.shape == t.shape

    def test_deadband_name(self, sine_signal: Signal) -> None:
        result = apply_deadband(sine_signal, threshold=0.5)
        assert "_deadband" in result.name

    def test_negative_threshold_raises(self, sine_signal: Signal) -> None:
        with pytest.raises((ValueError, AssertionError)):
            apply_deadband(sine_signal, threshold=-1.0)


class TestApplyHysteresis:
    """Tests for apply_hysteresis function."""

    def test_hysteresis_output_binary(self, sine_signal: Signal) -> None:
        result = apply_hysteresis(sine_signal, threshold_up=1.0, threshold_down=-1.0)
        # Output should only be 0 or 1
        assert set(np.unique(np.round(result.values, 6))).issubset({0.0, 1.0})

    def test_hysteresis_initial_high(self, sine_signal: Signal) -> None:
        result = apply_hysteresis(
            sine_signal, threshold_up=1.0, threshold_down=-1.0, initial_state=True
        )
        assert result.values.shape == sine_signal.values.shape

    def test_hysteresis_smooth(self, sine_signal: Signal) -> None:
        result = apply_hysteresis(
            sine_signal, threshold_up=1.0, threshold_down=-1.0, smooth=True
        )
        assert result.values.shape == sine_signal.values.shape

    def test_custom_output_levels(self, sine_signal: Signal) -> None:
        result = apply_hysteresis(
            sine_signal,
            threshold_up=1.0,
            threshold_down=-1.0,
            output_high=5.0,
            output_low=-5.0,
        )
        # should contain -5 or 5
        assert np.max(result.values) <= 5.0 + 1e-3

    def test_name_updated(self, sine_signal: Signal) -> None:
        result = apply_hysteresis(sine_signal, threshold_up=1.0, threshold_down=-1.0)
        assert "_hysteresis" in result.name


class TestApplyBacklash:
    """Tests for apply_backlash function."""

    def test_backlash_basic(self, sine_signal: Signal) -> None:
        result = apply_backlash(sine_signal, backlash_width=0.2)
        assert result.values.shape == sine_signal.values.shape

    def test_backlash_no_smooth(self, sine_signal: Signal) -> None:
        result = apply_backlash(sine_signal, backlash_width=0.5, smooth=False)
        assert result.values.shape == sine_signal.values.shape

    def test_negative_backlash_raises(self, sine_signal: Signal) -> None:
        with pytest.raises(ValueError):
            apply_backlash(sine_signal, backlash_width=-0.1)

    def test_nonpositive_smoothness_raises(self, sine_signal: Signal) -> None:
        with pytest.raises(ValueError):
            apply_backlash(sine_signal, backlash_width=0.1, smoothness=0.0)

    def test_name_updated(self, sine_signal: Signal) -> None:
        result = apply_backlash(sine_signal, backlash_width=0.1)
        assert "_backlash" in result.name


class TestSaturationUtilities:
    """Tests for create_saturation_function and visualize_saturation_curves."""

    def test_create_saturation_function(self) -> None:
        sat_fn = create_saturation_function(
            lower=-1.0, upper=1.0, mode=SaturationMode.HARD
        )
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = sat_fn(x)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_visualize_saturation_curves(self) -> None:
        curves = visualize_saturation_curves(lower=-1.0, upper=1.0)
        assert len(curves) == len(SaturationMode)
        for x, y in curves.values():
            assert len(x) == len(y)
