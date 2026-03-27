"""Unit tests for signal_toolkit/core.py (Signal, SignalGenerator)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.core import Signal, SignalGenerator


@pytest.fixture
def t100() -> np.ndarray:
    return np.linspace(0.0, 1.0, 100)


@pytest.fixture
def basic_signal(t100: np.ndarray) -> Signal:
    return Signal(time=t100, values=np.sin(2 * np.pi * t100), name="sin", units="V")


class TestSignal:
    """Tests for the Signal dataclass."""

    def test_creation_1d(self, t100: np.ndarray) -> None:
        sig = Signal(time=t100, values=np.zeros(100))
        assert sig.n_samples == 100

    def test_creation_2d_values(self, t100: np.ndarray) -> None:
        vals = np.zeros((100, 3))
        sig = Signal(time=t100, values=vals)
        assert sig.values.shape == (100, 3)

    def test_invalid_time_not_1d(self) -> None:
        with pytest.raises((ValueError, AssertionError)):
            Signal(time=np.zeros((2, 3)), values=np.zeros(6))

    def test_invalid_length_mismatch(self, t100: np.ndarray) -> None:
        with pytest.raises((ValueError, AssertionError)):
            Signal(time=t100, values=np.zeros(50))

    def test_fs_property(self, basic_signal: Signal) -> None:
        assert np.isclose(basic_signal.fs, 99.0, rtol=0.02)

    def test_dt_property(self, basic_signal: Signal) -> None:
        assert np.isclose(basic_signal.dt, 1.0 / 99.0, rtol=0.02)

    def test_duration_property(self, basic_signal: Signal) -> None:
        assert np.isclose(basic_signal.duration, 1.0, atol=0.02)

    def test_n_samples_property(self, basic_signal: Signal) -> None:
        assert basic_signal.n_samples == 100

    def test_copy_independent(self, basic_signal: Signal) -> None:
        copied = basic_signal.copy()
        copied.values[0] = 999.0
        assert basic_signal.values[0] != 999.0

    def test_slice(self, basic_signal: Signal) -> None:
        sliced = basic_signal.slice(0.25, 0.75)
        assert sliced.time[0] >= 0.25
        assert sliced.time[-1] <= 0.75

    def test_resample(self, basic_signal: Signal) -> None:
        resampled = basic_signal.resample(50.0)
        assert np.isclose(resampled.fs, 50.0, rtol=0.1)

    def test_add_constant(self, basic_signal: Signal) -> None:
        result = basic_signal + 1.0
        assert np.allclose(result.values, basic_signal.values + 1.0)

    def test_add_signal(self, basic_signal: Signal) -> None:
        result = basic_signal + basic_signal
        assert np.allclose(result.values, 2 * basic_signal.values)

    def test_add_signal_mismatch_raises(
        self, basic_signal: Signal, t100: np.ndarray
    ) -> None:
        other = Signal(time=t100 + 1.0, values=np.zeros(100))
        with pytest.raises(ValueError):
            _ = basic_signal + other

    def test_mul_constant(self, basic_signal: Signal) -> None:
        result = basic_signal * 3.0
        assert np.allclose(result.values, 3.0 * basic_signal.values)

    def test_mul_signal(self, basic_signal: Signal) -> None:
        result = basic_signal * basic_signal
        assert np.allclose(result.values, basic_signal.values**2)

    def test_mul_signal_mismatch_raises(
        self, basic_signal: Signal, t100: np.ndarray
    ) -> None:
        other = Signal(time=t100 + 2.0, values=np.zeros(100))
        with pytest.raises(ValueError):
            _ = basic_signal * other

    def test_neg(self, basic_signal: Signal) -> None:
        neg = -basic_signal
        assert np.allclose(neg.values, -basic_signal.values)

    def test_single_sample_properties(self) -> None:
        sig = Signal(time=np.array([0.0]), values=np.array([1.0]))
        assert sig.fs == 1.0
        assert sig.dt == 1.0
        assert sig.duration == 0.0


class TestSignalGenerator:
    """Tests for all SignalGenerator factory methods."""

    def test_constant(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.constant(t100, value=5.0)
        assert np.allclose(sig.values, 5.0)

    def test_sinusoid(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.sinusoid(t100, amplitude=2.0, frequency=3.0)
        assert np.isclose(np.max(np.abs(sig.values)), 2.0, atol=0.1)

    def test_cosine(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.cosine(t100, amplitude=1.0, frequency=2.0)
        assert np.isclose(sig.values[0], 1.0, atol=0.05)

    def test_exponential(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.exponential(t100, amplitude=1.0, decay_rate=1.0)
        # starts at 1, decays toward 0
        assert sig.values[0] > sig.values[-1]

    def test_linear(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.linear(t100, slope=2.0, intercept=1.0)
        assert np.isclose(sig.values[0], 1.0, atol=1e-9)

    def test_polynomial(self, t100: np.ndarray) -> None:
        # y = 0 + 0*t + 1*t^2 => c=[0, 0, 1]
        sig = SignalGenerator.polynomial(t100, coefficients=[0.0, 0.0, 1.0])
        t_shifted = t100 - t100[0]
        assert np.allclose(sig.values, t_shifted**2, atol=1e-8)

    def test_step(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.step(t100, step_time=0.5)
        # Before step_time=0.5: value=0; after: value=1
        assert sig.values[0] == 0.0
        assert sig.values[-1] == 1.0

    def test_pulse(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.pulse(t100, start_time=0.3, duration=0.2, amplitude=3.0)
        assert sig.values[0] == 0.0
        center_idx = np.argmin(np.abs(t100 - 0.4))
        assert np.isclose(sig.values[center_idx], 3.0)

    def test_chirp_linear(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.chirp(t100, f0=1.0, f1=5.0, method="linear")
        assert sig.values.shape == t100.shape

    def test_chirp_exponential(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.chirp(t100, f0=1.0, f1=10.0, method="exponential")
        assert sig.values.shape == t100.shape

    def test_chirp_unknown_method_raises(self, t100: np.ndarray) -> None:
        with pytest.raises(ValueError):
            SignalGenerator.chirp(t100, method="bad_method")

    def test_chirp_nonpositive_freq_exponential_raises(self, t100: np.ndarray) -> None:
        with pytest.raises(ValueError):
            SignalGenerator.chirp(t100, f0=0.0, f1=5.0, method="exponential")

    def test_sawtooth(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.sawtooth(t100, frequency=2.0, amplitude=1.0)
        assert sig.values.shape == t100.shape

    def test_triangle(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.triangle(t100, frequency=2.0, amplitude=1.0)
        assert sig.values.shape == t100.shape

    def test_square(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.square(t100, frequency=2.0, duty_cycle=0.5)
        # All values should be ±1
        assert set(np.unique(sig.values.astype(float))).issubset({-1.0, 1.0})

    def test_from_function(self, t100: np.ndarray) -> None:
        sig = SignalGenerator.from_function(t100, func=lambda t: t**2)
        t_shifted = t100 - t100[0]
        assert np.allclose(sig.values, t_shifted**2, atol=1e-10)

    def test_superposition_two(self, t100: np.ndarray) -> None:
        s1 = SignalGenerator.sinusoid(t100, amplitude=1.0, frequency=1.0)
        s2 = SignalGenerator.sinusoid(t100, amplitude=2.0, frequency=2.0)
        sup = SignalGenerator.superposition([s1, s2])
        assert np.allclose(sup.values, s1.values + s2.values)

    def test_superposition_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalGenerator.superposition([])

    def test_superposition_time_mismatch_raises(self, t100: np.ndarray) -> None:
        s1 = Signal(time=t100, values=np.zeros(100))
        s2 = Signal(time=t100 + 5.0, values=np.zeros(100))
        with pytest.raises(ValueError):
            SignalGenerator.superposition([s1, s2])
