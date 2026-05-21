"""Comprehensive tests for signal_toolkit.limits module.

Covers all saturation modes, rate limiting, deadband, hysteresis, backlash,
plus the factory and visualization helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.core import Signal
from signal_toolkit.limits import (
    SaturationMode,
    _apply_saturation_values,
    _cubic_clip,
    _exponential_clip,
    _soft_clip,
    apply_backlash,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
    create_saturation_function,
    visualize_saturation_curves,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def t() -> np.ndarray:
    return np.linspace(0, 1, 200)


@pytest.fixture
def ramp_signal(t: np.ndarray) -> Signal:
    """A ramp that goes from -2 to 2."""
    return Signal(time=t, values=np.linspace(-2.0, 2.0, len(t)), name="ramp")


@pytest.fixture
def sine_signal(t: np.ndarray) -> Signal:
    return Signal(time=t, values=np.sin(2 * np.pi * 3 * t), name="sine")


# ──────────────────────────────────────────────────────────────────────────────
# SaturationMode enum
# ──────────────────────────────────────────────────────────────────────────────


class TestSaturationMode:
    def test_all_modes_exist(self):
        modes = {m.value for m in SaturationMode}
        assert {
            "hard",
            "soft",
            "tanh",
            "sigmoid",
            "atan",
            "cubic",
            "exponential",
        } == modes


# ──────────────────────────────────────────────────────────────────────────────
# _apply_saturation_values (private helper; covers the else branch too)
# ──────────────────────────────────────────────────────────────────────────────


class TestApplySaturationValues:
    def test_hard_mode(self):
        x = np.array([-3.0, 0.0, 3.0])
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.HARD, 1.0)
        np.testing.assert_array_equal(result, np.array([-1.0, 0.0, 1.0]))

    def test_zero_range_returns_center(self):
        """When upper == lower, the range is zero → all values map to center."""
        x = np.array([-2.0, 0.0, 5.0])
        # upper == lower == 0.5 → center = 0.5, half_range = 0
        result = _apply_saturation_values(x, 0.5, 0.5, SaturationMode.TANH, 1.0)
        np.testing.assert_array_almost_equal(result, np.full(3, 0.5))

    def test_tanh_mode_within_bounds(self):
        x = np.linspace(-2.0, 2.0, 50)
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.TANH, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_sigmoid_mode_within_bounds(self):
        x = np.linspace(-3.0, 3.0, 50)
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.SIGMOID, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_atan_mode_within_bounds(self):
        x = np.linspace(-5.0, 5.0, 50)
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.ATAN, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_soft_mode_within_bounds(self):
        x = np.linspace(-2.0, 2.0, 50)
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.SOFT, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_cubic_mode_within_bounds(self):
        x = np.linspace(-2.0, 2.0, 50)
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.CUBIC, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_exponential_mode_within_bounds(self):
        x = np.linspace(-3.0, 3.0, 50)
        result = _apply_saturation_values(x, -1.0, 1.0, SaturationMode.EXPONENTIAL, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_else_branch_fallback(self):
        """The else branch clips normalized to [-1, 1]."""
        from unittest.mock import MagicMock

        fake_mode = MagicMock()
        # Make all == comparisons with SaturationMode members return False
        fake_mode.__eq__ = lambda self, other: False
        x = np.array([-2.0, 0.0, 2.0])
        result = _apply_saturation_values(x, -1.0, 1.0, fake_mode, 1.0)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# apply_saturation
# ──────────────────────────────────────────────────────────────────────────────


class TestApplySaturation:
    def test_hard_saturation(self, ramp_signal: Signal):
        result = apply_saturation(
            ramp_signal, lower=-0.5, upper=0.5, mode=SaturationMode.HARD
        )
        assert result.name == "ramp_saturated"
        assert np.all(result.values >= -0.5)
        assert np.all(result.values <= 0.5)

    def test_all_modes_run_without_error(self, ramp_signal: Signal):
        for mode in SaturationMode:
            result = apply_saturation(ramp_signal, lower=-1.0, upper=1.0, mode=mode)
            assert result.name == "ramp_saturated"

    def test_name_is_correct(self, ramp_signal: Signal):
        result = apply_saturation(ramp_signal)
        assert result.name == "ramp_saturated"

    def test_preserves_time(self, ramp_signal: Signal):
        result = apply_saturation(ramp_signal, lower=-0.5, upper=0.5)
        np.testing.assert_array_equal(result.time, ramp_signal.time)

    def test_units_preserved(self, t: np.ndarray):
        sig = Signal(time=t, values=np.ones(len(t)), name="test", units="m/s")
        result = apply_saturation(sig)
        assert result.units == "m/s"


# ──────────────────────────────────────────────────────────────────────────────
# Private clip helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestPrivateClipHelpers:
    def test_soft_clip_linear_region(self):
        x = np.array([-0.3, 0.0, 0.3])
        result = _soft_clip(x, k=1.0)
        # In linear region, output ≈ input
        np.testing.assert_array_almost_equal(result, x, decimal=5)

    def test_soft_clip_saturated_region(self):
        x = np.array([-5.0, 5.0])
        result = _soft_clip(x, k=1.0)
        assert np.all(np.abs(result) <= 1.0)

    def test_cubic_clip_within_unit(self):
        x = np.linspace(-0.9, 0.9, 20)
        result = _cubic_clip(x, k=1.0)
        assert np.all(np.abs(result) <= 1.0)

    def test_cubic_clip_outside_unit(self):
        x = np.array([-5.0, 5.0])
        result = _cubic_clip(x, k=1.0)
        assert np.all(np.abs(result) <= 1.0)

    def test_exponential_clip_normalizer_near_zero(self):
        """Very small k triggers the normalizer guard."""
        x = np.linspace(-2.0, 2.0, 20)
        result = _exponential_clip(x, k=1e-15)  # normalizer < 1e-10
        assert np.all(np.isfinite(result))


# ──────────────────────────────────────────────────────────────────────────────
# apply_rate_limiter
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyRateLimiter:
    def test_rate_limiter_limits_changes(self, ramp_signal: Signal):
        result = apply_rate_limiter(ramp_signal, max_rate=0.5, smooth_transition=False)
        assert result.name == "ramp_rate_limited"
        assert len(result.values) == len(ramp_signal.values)

    def test_rate_limiter_smooth(self, ramp_signal: Signal):
        result = apply_rate_limiter(ramp_signal, max_rate=0.5, smooth_transition=True)
        assert len(result.values) == len(ramp_signal.values)

    def test_rate_limiter_preserves_slow_signals(self, t: np.ndarray):
        """A very slow signal should pass through unchanged."""
        slow = Signal(time=t, values=t * 0.001, name="slow")
        result = apply_rate_limiter(slow, max_rate=100.0, smooth_transition=False)
        np.testing.assert_array_almost_equal(result.values, slow.values, decimal=5)

    def test_rate_limiter_negative_rate_limiting(self, t: np.ndarray):
        """Test the negative rate branch (delta < -max_delta)."""
        # Steeply decreasing signal
        steep_down = Signal(
            time=t, values=np.linspace(10.0, -10.0, len(t)), name="steepdown"
        )
        result = apply_rate_limiter(steep_down, max_rate=1.0, smooth_transition=False)
        assert len(result.values) == len(t)

    def test_rate_limiter_units_preserved(self, ramp_signal: Signal):
        result = apply_rate_limiter(ramp_signal, max_rate=1.0)
        assert result.units == ramp_signal.units


# ──────────────────────────────────────────────────────────────────────────────
# apply_deadband
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyDeadband:
    def test_deadband_hard_zero_zone(self, t: np.ndarray):
        sig = Signal(time=t, values=np.linspace(-0.5, 0.5, len(t)), name="sig")
        result = apply_deadband(sig, threshold=0.1, center=0.0, smooth=False)
        assert result.name == "sig_deadband"
        # Values in deadband → 0 (center)
        in_band = np.where(np.abs(sig.values) <= 0.1)
        np.testing.assert_array_almost_equal(result.values[in_band], 0.0)

    def test_deadband_smooth_mode(self, t: np.ndarray):
        sig = Signal(time=t, values=np.linspace(-1.0, 1.0, len(t)), name="sig")
        result = apply_deadband(sig, threshold=0.2, center=0.0, smooth=True)
        assert len(result.values) == len(t)

    def test_deadband_preserves_time(self, sine_signal: Signal):
        result = apply_deadband(sine_signal, threshold=0.3)
        np.testing.assert_array_equal(result.time, sine_signal.time)


# ──────────────────────────────────────────────────────────────────────────────
# apply_hysteresis
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyHysteresis:
    def test_hysteresis_transitions(self, t: np.ndarray):
        sig = Signal(time=t, values=np.sin(2 * np.pi * 2 * t), name="sine")
        result = apply_hysteresis(
            sig, threshold_up=0.5, threshold_down=-0.5, output_high=1.0, output_low=0.0
        )
        assert result.name == "sine_hysteresis"
        assert set(np.unique(result.values)).issubset({0.0, 1.0})

    def test_hysteresis_initial_state_high(self, t: np.ndarray):
        # Start high – values above threshold_down stay high initially
        sig = Signal(time=t, values=np.zeros(len(t)), name="zero")
        result = apply_hysteresis(
            sig, threshold_up=0.5, threshold_down=-0.5, initial_state=True
        )
        assert result.values[0] == 1.0

    def test_hysteresis_smooth(self, t: np.ndarray):
        """smooth=True applies convolution smoothing."""
        sig = Signal(time=t, values=np.sin(2 * np.pi * 2 * t), name="sine")
        result = apply_hysteresis(
            sig,
            threshold_up=0.3,
            threshold_down=-0.3,
            output_high=1.0,
            output_low=0.0,
            smooth=True,
            smoothness=5.0,
        )
        # Smoothed output should have values between 0 and 1
        assert np.all(result.values >= -0.1)
        assert np.all(result.values <= 1.1)

    def test_hysteresis_units_preserved(self, sine_signal: Signal):
        result = apply_hysteresis(sine_signal, threshold_up=0.5, threshold_down=-0.5)
        assert result.units == sine_signal.units


# ──────────────────────────────────────────────────────────────────────────────
# apply_backlash
# ──────────────────────────────────────────────────────────────────────────────


class TestApplyBacklash:
    def test_backlash_no_backlash(self, ramp_signal: Signal):
        result = apply_backlash(ramp_signal, backlash_width=0.0, smooth=False)
        assert result.name == "ramp_backlash"
        # Zero backlash → output tracks input exactly
        np.testing.assert_array_almost_equal(
            result.values, ramp_signal.values, decimal=5
        )

    def test_backlash_with_width(self, ramp_signal: Signal):
        result = apply_backlash(ramp_signal, backlash_width=0.5, smooth=False)
        assert len(result.values) == len(ramp_signal.values)

    def test_backlash_smooth_mode(self, ramp_signal: Signal):
        result = apply_backlash(
            ramp_signal, backlash_width=0.2, smooth=True, smoothness=5.0
        )
        assert len(result.values) == len(ramp_signal.values)

    def test_backlash_negative_width_raises(self, sine_signal: Signal):
        with pytest.raises(ValueError, match="non-negative"):
            apply_backlash(sine_signal, backlash_width=-0.1)

    def test_backlash_non_positive_smoothness_raises(self, sine_signal: Signal):
        with pytest.raises(ValueError, match="positive"):
            apply_backlash(sine_signal, backlash_width=0.1, smoothness=0.0)

    def test_backlash_output_decreasing_ramp(self, t: np.ndarray):
        """Test the negative delta branch inside the backlash loop."""
        sig = Signal(time=t, values=np.linspace(2.0, -2.0, len(t)), name="neg_ramp")
        result = apply_backlash(sig, backlash_width=0.5, smooth=False)
        assert len(result.values) == len(t)


# ──────────────────────────────────────────────────────────────────────────────
# create_saturation_function
# ──────────────────────────────────────────────────────────────────────────────


class TestCreateSaturationFunction:
    def test_returns_callable(self):
        fn = create_saturation_function(-1.0, 1.0, SaturationMode.TANH)
        assert callable(fn)

    def test_function_applies_saturation(self):
        fn = create_saturation_function(-0.5, 0.5, SaturationMode.HARD)
        x = np.array([-2.0, 0.0, 2.0])
        result = fn(x)
        assert np.all(result >= -0.5)
        assert np.all(result <= 0.5)

    def test_default_mode_is_tanh(self):
        fn = create_saturation_function(-1.0, 1.0)
        x = np.linspace(-5.0, 5.0, 20)
        result = fn(x)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# visualize_saturation_curves
# ──────────────────────────────────────────────────────────────────────────────


class TestVisualizeSaturationCurves:
    def test_returns_all_modes(self):
        curves = visualize_saturation_curves(-1.0, 1.0, 1.0, 100)
        assert set(curves.keys()) == {m.value for m in SaturationMode}

    def test_each_curve_has_correct_shape(self):
        curves = visualize_saturation_curves(-1.0, 1.0, 1.0, 200)
        for x, y in curves.values():
            assert len(x) == 200
            assert len(y) == 200

    def test_values_within_bounds(self):
        curves = visualize_saturation_curves(-2.0, 2.0, 1.0, 100)
        for _x, y in curves.values():
            assert np.all(y >= -2.0) and np.all(y <= 2.0)
