"""Tests for the signal processing toolkit.

This module contains comprehensive tests for all signal toolkit components:
- Core signal classes
- Function fitting
- Limits and saturation
- Calculus (differentiation/integration)
- Noise generation
- Filters
- Import/export
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    Integrator,
    compute_derivative,
    compute_integral,
    compute_tangent_line,
)
from src.shared.python.signal_toolkit.core import Signal, SignalGenerator
from src.shared.python.signal_toolkit.filters import (
    FilterDesigner,
    FilterType,
    apply_filter,
    apply_median_filter,
    apply_moving_average,
    apply_savgol,
)
from src.shared.python.signal_toolkit.fitting import (
    ExponentialFitter,
    FunctionFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)
from src.shared.python.signal_toolkit.io import (
    SignalExporter,
    SignalImporter,
)
from src.shared.python.signal_toolkit.limits import (
    SaturationMode,
    apply_deadband,
    apply_hysteresis,
    apply_rate_limiter,
    apply_saturation,
)
from src.shared.python.signal_toolkit.noise import (
    DisturbanceSimulator,
    NoiseGenerator,
    NoiseType,
    add_noise_to_signal,
)

# =============================================================================
# Core Signal Tests
# =============================================================================


class TestSignal:
    """Tests for the Signal class."""

    def test_signal_creation(self) -> None:
        """Test basic signal creation."""
        t = np.linspace(0, 10, 100)
        values = np.sin(t)

        signal = Signal(t, values, name="test", units="rad")

        assert len(signal.time) == 100
        assert len(signal.values) == 100
        assert signal.name == "test"
        assert signal.units == "rad"

    def test_signal_properties(self) -> None:
        """Test signal property calculations."""
        t = np.linspace(0, 10, 1001)
        values = np.zeros(1001)

        signal = Signal(t, values)

        assert signal.n_samples == 1001
        assert signal.duration == pytest.approx(10.0, rel=1e-3)
        assert signal.fs == pytest.approx(100.0, rel=1e-2)
        assert signal.dt == pytest.approx(0.01, rel=1e-2)

    def test_signal_copy(self) -> None:
        """Test signal copying."""
        t = np.linspace(0, 10, 100)
        values = np.sin(t)

        original = Signal(t, values, name="original")
        copy = original.copy()

        # Modify copy
        copy.values[0] = 999
        copy.name = "modified"

        # Original should be unchanged
        assert original.values[0] != 999
        assert original.name == "original"

    def test_signal_slice(self) -> None:
        """Test signal slicing."""
        t = np.linspace(0, 10, 101)
        values = t  # Linear ramp

        signal = Signal(t, values)
        sliced = signal.slice(2.0, 5.0)

        assert sliced.time[0] >= 2.0
        assert sliced.time[-1] <= 5.0
        assert len(sliced.time) < len(signal.time)

    def test_signal_arithmetic(self) -> None:
        """Test signal arithmetic operations."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 2)
        s2 = Signal(t, np.ones(100) * 3)

        # Addition
        s3 = s1 + s2
        assert np.allclose(s3.values, 5.0)

        # Multiplication
        s4 = s1 * s2
        assert np.allclose(s4.values, 6.0)

        # Scalar operations
        s5 = s1 + 10
        assert np.allclose(s5.values, 12.0)

        s6 = s1 * 5
        assert np.allclose(s6.values, 10.0)


class TestSignalGenerator:
    """Tests for the SignalGenerator class."""

    def test_sinusoid_generation(self) -> None:
        """Test sinusoid generation."""
        t = np.linspace(0, 1, 1000)

        signal = SignalGenerator.sinusoid(
            t,
            amplitude=2.0,
            frequency=5.0,
            phase=0.0,
            offset=1.0,
        )

        assert signal.name == "sinusoid"
        # Check amplitude
        assert max(signal.values) == pytest.approx(3.0, rel=1e-2)
        assert min(signal.values) == pytest.approx(-1.0, rel=1e-2)

    def test_polynomial_generation(self) -> None:
        """Test polynomial generation."""
        t = np.linspace(0, 2, 100)

        # y = 1 + 2*t + 3*t^2
        signal = SignalGenerator.polynomial(t, [1, 2, 3])

        # At t=0: y = 1
        assert signal.values[0] == pytest.approx(1.0)
        # At t=2: y = 1 + 4 + 12 = 17
        assert signal.values[-1] == pytest.approx(17.0, rel=1e-2)

    def test_exponential_generation(self) -> None:
        """Test exponential generation."""
        t = np.linspace(0, 5, 100)

        signal = SignalGenerator.exponential(t, amplitude=10.0, decay_rate=1.0)

        # At t=0: y = 10
        assert signal.values[0] == pytest.approx(10.0)
        # Decays over time
        assert signal.values[-1] < signal.values[0]

    def test_step_generation(self) -> None:
        """Test step generation."""
        t = np.linspace(0, 10, 100)

        signal = SignalGenerator.step(
            t, step_time=5.0, step_value=2.0, initial_value=0.0
        )

        # Before step
        assert signal.values[0] == 0.0
        # After step
        assert signal.values[-1] == 2.0

    def test_chirp_generation(self) -> None:
        """Test chirp (frequency sweep) generation."""
        t = np.linspace(0, 5, 1000)

        signal = SignalGenerator.chirp(t, f0=1.0, f1=10.0, amplitude=1.0)

        # Check amplitude bounds
        assert max(np.abs(signal.values)) <= 1.1

    def test_custom_function(self) -> None:
        """Test custom function generation."""
        t = np.linspace(0, 10, 100)

        signal = SignalGenerator.from_function(t, lambda x: x**2 + 1)

        assert signal.values[0] == pytest.approx(1.0)
        assert signal.values[-1] == pytest.approx(101.0)


# =============================================================================
# Function Fitting Tests
# =============================================================================


class TestFunctionFitting:
    """Tests for function fitting."""

    def test_linear_fit(self) -> None:
        """Test linear function fitting."""
        t = np.linspace(0, 10, 100)
        values = 2.0 * t + 5.0 + np.random.normal(0, 0.1, 100)

        signal = Signal(t, values)
        fitter = LinearFitter()
        result = fitter.fit(signal)

        assert result.success
        assert result.r_squared > 0.99
        assert result.parameters["slope"] == pytest.approx(2.0, rel=0.1)
        assert result.parameters["intercept"] == pytest.approx(5.0, rel=0.1)

    def test_polynomial_fit(self) -> None:
        """Test polynomial fitting."""
        t = np.linspace(0, 5, 100)
        values = 1 + 2 * t + 0.5 * t**2

        signal = Signal(t, values)
        fitter = PolynomialFitter(order=2)
        result = fitter.fit(signal)

        assert result.success
        assert result.r_squared > 0.999
        assert result.parameters["c0"] == pytest.approx(1.0, rel=0.1)
        assert result.parameters["c1"] == pytest.approx(2.0, rel=0.1)
        assert result.parameters["c2"] == pytest.approx(0.5, rel=0.1)

    def test_sinusoid_fit(self) -> None:
        """Test sinusoidal fitting."""
        t = np.linspace(0, 2, 200)
        values = 3.0 * np.sin(2 * np.pi * 2.0 * t + 0.5) + 1.0

        signal = Signal(t, values)
        fitter = SinusoidFitter()
        result = fitter.fit(signal)

        assert result.success
        assert result.r_squared > 0.99
        assert result.parameters["amplitude"] == pytest.approx(3.0, rel=0.1)
        assert result.parameters["frequency"] == pytest.approx(2.0, rel=0.1)

    def test_exponential_fit(self) -> None:
        """Test exponential decay fitting."""
        t = np.linspace(0, 5, 100)
        values = 10.0 * np.exp(-0.5 * t) + 2.0

        signal = Signal(t, values)
        fitter = ExponentialFitter()
        result = fitter.fit_decay(signal)

        assert result.success
        assert result.r_squared > 0.99
        assert result.parameters["amplitude"] == pytest.approx(10.0, rel=0.2)
        assert result.parameters["decay_rate"] == pytest.approx(0.5, rel=0.2)

    def test_auto_fit(self) -> None:
        """Test automatic best fit detection."""
        t = np.linspace(0, 10, 100)
        values = 3.0 * t + 7.0

        signal = Signal(t, values)
        fitter = FunctionFitter()
        best_type, result = fitter.auto_fit(signal)

        assert best_type == "linear"
        assert result.r_squared > 0.99


# =============================================================================
# Limits and Saturation Tests
# =============================================================================


class TestLimits:
    """Tests for limits and saturation."""

    def test_hard_saturation(self) -> None:
        """Test hard clipping saturation."""
        t = np.linspace(0, 10, 100)
        values = np.linspace(-2, 2, 100)

        signal = Signal(t, values)
        saturated = apply_saturation(
            signal, lower=-1.0, upper=1.0, mode=SaturationMode.HARD
        )

        assert max(saturated.values) <= 1.0
        assert min(saturated.values) >= -1.0

    def test_soft_saturation(self) -> None:
        """Test soft (tanh) saturation."""
        t = np.linspace(0, 10, 100)
        values = np.linspace(-5, 5, 100)

        signal = Signal(t, values)
        saturated = apply_saturation(
            signal, lower=-1.0, upper=1.0, mode=SaturationMode.TANH
        )

        # Tanh smoothly approaches limits
        assert max(saturated.values) <= 1.0
        assert min(saturated.values) >= -1.0

    def test_rate_limiter(self) -> None:
        """Test rate limiting."""
        t = np.linspace(0, 1, 100)
        # Instantaneous step
        values = np.where(t >= 0.5, 10.0, 0.0)

        signal = Signal(t, values)
        limited = apply_rate_limiter(signal, max_rate=5.0)

        # Rate should be limited
        diffs = np.diff(limited.values) / np.diff(limited.time)
        assert all(np.abs(diffs) <= 5.1)  # Allow small tolerance

    def test_deadband(self) -> None:
        """Test deadband application."""
        t = np.linspace(0, 10, 100)
        values = np.linspace(-2, 2, 100)

        signal = Signal(t, values)
        result = apply_deadband(signal, threshold=0.5, center=0.0, smooth=False)

        # Values within deadband should be at center
        mid_idx = len(result.values) // 2
        assert result.values[mid_idx] == pytest.approx(0.0, abs=0.1)

    def test_hysteresis(self) -> None:
        """Test hysteresis (Schmitt trigger)."""
        t = np.linspace(0, 10, 200)
        # Sinusoid crossing thresholds
        values = np.sin(2 * np.pi * 0.5 * t) * 2

        signal = Signal(t, values)
        result = apply_hysteresis(
            signal,
            threshold_up=0.5,
            threshold_down=-0.5,
            output_high=1.0,
            output_low=0.0,
        )

        # Should be binary output
        assert set(np.unique(result.values)) <= {0.0, 1.0}


# =============================================================================
# Calculus Tests
# =============================================================================


class TestCalculus:
    """Tests for differentiation and integration."""

    def test_derivative_of_linear(self) -> None:
        """Test derivative of linear function."""
        t = np.linspace(0, 10, 1000)
        values = 2.0 * t + 5.0

        signal = Signal(t, values)
        derivative = compute_derivative(signal)

        # Derivative of 2*t + 5 = 2
        assert np.mean(derivative.values[10:-10]) == pytest.approx(2.0, rel=0.1)

    def test_derivative_of_sine(self) -> None:
        """Test derivative of sine function."""
        t = np.linspace(0, 2 * np.pi, 1000)
        values = np.sin(t)

        signal = Signal(t, values)
        derivative = compute_derivative(signal)

        # Derivative of sin(t) = cos(t)
        expected = np.cos(t)
        correlation = np.corrcoef(derivative.values[10:-10], expected[10:-10])[0, 1]
        assert correlation > 0.99

    def test_integral_of_constant(self) -> None:
        """Test integral of constant function."""
        t = np.linspace(0, 10, 100)
        values = np.ones(100) * 5.0

        signal = Signal(t, values)
        result = compute_integral(signal, lower_bound=0, upper_bound=10)

        # Integral of 5 from 0 to 10 = 50
        assert result.value == pytest.approx(50.0, rel=0.01)

    def test_integral_of_linear(self) -> None:
        """Test integral of linear function."""
        t = np.linspace(0, 4, 1000)
        values = t  # y = t

        signal = Signal(t, values)
        result = compute_integral(signal, lower_bound=0, upper_bound=4)

        # Integral of t from 0 to 4 = t^2/2 |_0^4 = 8
        assert result.value == pytest.approx(8.0, rel=0.01)

    def test_tangent_line(self) -> None:
        """Test tangent line computation."""
        t = np.linspace(0, 10, 1000)
        values = t**2

        signal = Signal(t, values)
        tangent = compute_tangent_line(signal, t_point=5.0)

        # At t=5, y=25, slope=10
        assert tangent.t_point == pytest.approx(5.0, rel=0.1)
        assert tangent.y_point == pytest.approx(25.0, rel=0.1)
        assert tangent.slope == pytest.approx(10.0, rel=0.1)

    def test_differentiation_methods(self) -> None:
        """Test different differentiation methods."""
        t = np.linspace(0, 10, 500)
        values = np.sin(t)

        signal = Signal(t, values)

        for method in DifferentiationMethod:
            diff = Differentiator(method=method)
            result = diff.differentiate(signal)
            # All methods should give reasonable results
            assert len(result.values) == len(signal.values)


# =============================================================================
# Noise Tests
# =============================================================================


class TestNoise:
    """Tests for noise generation."""

    def test_white_noise(self) -> None:
        """Test white noise generation."""
        t = np.linspace(0, 10, 10000)
        generator = NoiseGenerator(seed=42)

        noise = generator.generate(t, NoiseType.WHITE, amplitude=1.0)

        # Should have zero mean and specified std
        assert abs(np.mean(noise.values)) < 0.1
        assert np.std(noise.values) == pytest.approx(1.0, rel=0.1)

    def test_noise_types(self) -> None:
        """Test all noise types generate without error."""
        t = np.linspace(0, 10, 1000)
        generator = NoiseGenerator(seed=42)

        for noise_type in NoiseType:
            noise = generator.generate(t, noise_type, amplitude=1.0)
            assert len(noise.values) == len(t)

    def test_add_noise_with_snr(self) -> None:
        """Test adding noise with specified SNR."""
        t = np.linspace(0, 10, 1000)
        values = np.sin(2 * np.pi * t)

        signal = Signal(t, values)
        noisy = add_noise_to_signal(signal, NoiseType.WHITE, snr_db=20, seed=42)

        # Signal should still be recognizable
        correlation = np.corrcoef(signal.values, noisy.values)[0, 1]
        assert correlation > 0.9

    def test_disturbance_simulator(self) -> None:
        """Test disturbance simulator."""
        t = np.linspace(0, 10, 1000)

        sim = DisturbanceSimulator(seed=42)
        sim.add_noise(NoiseType.WHITE, amplitude=0.1)
        sim.add_step(step_time=5.0, magnitude=1.0)

        disturbance = sim.generate(t)

        assert len(disturbance.values) == len(t)
        # Should have step at t=5
        assert disturbance.values[-1] > disturbance.values[0]


# =============================================================================
# Filter Tests
# =============================================================================


class TestFilters:
    """Tests for signal filtering."""

    def test_butterworth_lowpass(self) -> None:
        """Test Butterworth lowpass filter."""
        t = np.linspace(0, 10, 1000)
        fs = 100.0
        # Signal with low and high frequency components
        values = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 20 * t)

        signal = Signal(t, values)

        spec = FilterDesigner.butterworth(
            FilterType.LOWPASS, cutoff=5.0, fs=fs, order=4
        )
        filtered = apply_filter(signal, spec)

        # High frequency should be attenuated
        assert np.std(filtered.values) < np.std(signal.values)

    def test_butterworth_highpass(self) -> None:
        """Test Butterworth highpass filter."""
        t = np.linspace(0, 10, 1000)
        fs = 100.0
        # DC component + high frequency
        values = 5.0 + np.sin(2 * np.pi * 20 * t)

        signal = Signal(t, values)

        spec = FilterDesigner.butterworth(
            FilterType.HIGHPASS, cutoff=10.0, fs=fs, order=4
        )
        filtered = apply_filter(signal, spec)

        # DC should be removed
        assert abs(np.mean(filtered.values)) < 1.0

    def test_bandpass_filter(self) -> None:
        """Test bandpass filter."""
        t = np.linspace(0, 10, 2000)
        fs = 200.0
        # Components at 5Hz, 15Hz, and 30Hz
        values = (
            np.sin(2 * np.pi * 5 * t)
            + np.sin(2 * np.pi * 15 * t)
            + np.sin(2 * np.pi * 30 * t)
        )

        signal = Signal(t, values)

        # Bandpass 10-20 Hz should keep only 15Hz
        spec = FilterDesigner.butterworth(
            FilterType.BANDPASS, cutoff=(10.0, 20.0), fs=fs, order=4
        )
        filtered = apply_filter(signal, spec)

        # Energy should be reduced
        assert np.var(filtered.values) < np.var(signal.values)

    def test_moving_average(self) -> None:
        """Test moving average filter."""
        t = np.linspace(0, 10, 100)
        values = np.random.randn(100)

        signal = Signal(t, values)
        filtered = apply_moving_average(signal, window_size=5)

        # Should be smoother
        diffs_original = np.abs(np.diff(signal.values))
        diffs_filtered = np.abs(np.diff(filtered.values))
        assert np.mean(diffs_filtered) < np.mean(diffs_original)

    def test_savgol_filter(self) -> None:
        """Test Savitzky-Golay filter."""
        t = np.linspace(0, 10, 100)
        values = np.sin(t) + np.random.randn(100) * 0.5

        signal = Signal(t, values)
        filtered = apply_savgol(signal, window_length=11, polyorder=3)

        # Should be smoother
        assert np.std(filtered.values) < np.std(signal.values)

    def test_median_filter(self) -> None:
        """Test median filter for impulse removal."""
        t = np.linspace(0, 10, 100)
        values = np.sin(t).copy()
        # Add impulse noise
        values[25] = 100
        values[75] = -100

        signal = Signal(t, values)
        filtered = apply_median_filter(signal, kernel_size=5)

        # Impulses should be removed
        assert max(np.abs(filtered.values)) < 10


# =============================================================================
# I/O Tests
# =============================================================================


class TestIO:
    """Tests for signal import/export."""

    def test_csv_roundtrip(self) -> None:
        """Test CSV export and import."""
        t = np.linspace(0, 10, 100)
        values = np.sin(t)

        original = Signal(t, values, name="test_signal")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"

            # Export
            SignalExporter.to_csv(original, path)

            # Import
            imported = SignalImporter.from_csv(path, time_column=0, value_columns=1)

            # Compare
            assert np.allclose(original.time, imported.time, rtol=1e-4)
            assert np.allclose(original.values, imported.values, rtol=1e-4)

    def test_json_roundtrip(self) -> None:
        """Test JSON export and import."""
        t = np.linspace(0, 10, 100)
        values = np.cos(t)

        original = Signal(t, values, name="json_test", units="rad/s")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            # Export
            SignalExporter.to_json(original, path)

            # Import
            imported = SignalImporter.from_json(path)

            # Compare
            assert np.allclose(original.time, imported.time, rtol=1e-4)
            assert np.allclose(original.values, imported.values, rtol=1e-4)
            assert imported.name == "json_test"
            assert imported.units == "rad/s"

    def test_npz_roundtrip(self) -> None:
        """Test NPZ export and import."""
        t = np.linspace(0, 10, 100)
        values = np.exp(-t)

        original = Signal(t, values, name="npz_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"

            # Export
            SignalExporter.to_npz(original, path)

            # Import
            imported = SignalImporter.from_npz(
                path, time_key="time", value_key="npz_test"
            )

            # Compare
            assert np.allclose(original.time, imported.time)
            assert np.allclose(original.values, imported.values)

    def test_dict_conversion(self) -> None:
        """Test dictionary conversion."""
        t = np.linspace(0, 10, 100)
        values = t**2

        original = Signal(t, values, name="dict_test")

        # To dict
        data = SignalExporter.to_dict(original)

        # From dict
        restored = SignalImporter.from_dict(data)

        assert np.allclose(original.time, np.array(restored.time))
        assert np.allclose(original.values, np.array(restored.values))

    def test_multiple_signals_csv(self) -> None:
        """Test exporting/importing multiple signals."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.sin(t), name="sin")
        s2 = Signal(t, np.cos(t), name="cos")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.csv"

            # Export multiple
            SignalExporter.to_csv([s1, s2], path)

            # Import all columns
            imported = SignalImporter.from_csv(path)

            if isinstance(imported, list):
                assert len(imported) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_generate_fit_export(self) -> None:
        """Test generating, fitting, and exporting a signal."""
        t = np.linspace(0, 10, 500)

        # Generate
        signal = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=1.0, offset=0.5)

        # Add noise
        noisy = add_noise_to_signal(signal, NoiseType.WHITE, snr_db=30, seed=42)

        # Filter
        filtered = apply_savgol(noisy)

        # Fit
        fitter = FunctionFitter()
        result = fitter.fit_sinusoid(filtered)

        # Should recover parameters
        assert result.parameters["amplitude"] == pytest.approx(2.0, rel=0.2)
        assert result.parameters["frequency"] == pytest.approx(1.0, rel=0.2)

    def test_signal_processing_pipeline(self) -> None:
        """Test a complete signal processing pipeline."""
        t = np.linspace(0, 10, 1000)

        # Generate base signal
        signal = SignalGenerator.sinusoid(t, amplitude=5.0, frequency=2.0)

        # Apply saturation
        saturated = apply_saturation(
            signal, lower=-3.0, upper=3.0, mode=SaturationMode.TANH
        )

        # Add noise
        noisy = add_noise_to_signal(saturated, NoiseType.WHITE, snr_db=20)

        # Filter
        spec = FilterDesigner.butterworth(FilterType.LOWPASS, cutoff=5.0, fs=100.0)
        filtered = apply_filter(noisy, spec)

        # Compute derivative
        derivative = compute_derivative(filtered)

        # Verify pipeline
        assert len(derivative.values) == len(t)
        assert max(saturated.values) <= 3.0
        assert min(saturated.values) >= -3.0

    def test_calculus_consistency(self) -> None:
        """Test that integration and differentiation are inverse operations."""
        t = np.linspace(0, 10, 1000)
        values = np.sin(t)

        signal = Signal(t, values)

        # Differentiate then integrate
        derivative = compute_derivative(signal)
        integrator = Integrator()
        integral = integrator.cumulative_integral(derivative)

        # Should approximately recover original (minus constant)
        # Normalize both
        recovered = integral.values - np.mean(integral.values)
        original = signal.values - np.mean(signal.values)

        correlation = np.corrcoef(recovered[50:-50], original[50:-50])[0, 1]
        assert correlation > 0.95
