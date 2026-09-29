# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Extended tests for signal toolkit - coverage gaps identified in assessment.

Tests for:
- Signal arithmetic (__sub__, __truediv__, reverse operators)
- Signal.resample() for 2D signals
- apply_backlash()
- compute_curvature(), compute_arc_length()
- find_extrema(), find_inflection_points()
- Filter DbC preconditions
- AdaptiveFilter (LMS/RLS)
- Bessel filter
- BatchProcessor
- Noise spectral properties
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from signal_toolkit.adaptive_filter import AdaptiveFilter as ExtractedAdaptiveFilter
from signal_toolkit.calculus import (
    compute_arc_length,
    compute_curvature,
    find_extrema,
    find_inflection_points,
)
from signal_toolkit.core import Signal, SignalGenerator
from signal_toolkit.filters import (
    AdaptiveFilter,
    FilterDesigner,
    FilterType,
    apply_filter,
)
from signal_toolkit.io import BatchProcessor, SignalExporter
from signal_toolkit.limits import SaturationMode, apply_backlash, apply_saturation
from signal_toolkit.noise import NoiseGenerator, NoiseType

# =============================================================================
# Signal Arithmetic Extended Tests
# =============================================================================


class TestSignalArithmetic:
    """Tests for Signal arithmetic operators."""

    def test_subtraction_signals(self) -> None:
        """Test signal-signal subtraction."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 5)
        s2 = Signal(t, np.ones(100) * 3)

        result = s1 - s2
        assert np.allclose(result.values, 2.0)

    def test_subtraction_scalar(self) -> None:
        """Test signal - scalar."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 5)

        result = s1 - 3
        assert np.allclose(result.values, 2.0)

    def test_division_signals(self) -> None:
        """Test signal / signal."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 10)
        s2 = Signal(t, np.ones(100) * 2)

        result = s1 / s2
        assert np.allclose(result.values, 5.0)

    def test_division_scalar(self) -> None:
        """Test signal / scalar."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 10)

        result = s1 / 2
        assert np.allclose(result.values, 5.0)

    def test_reverse_addition(self) -> None:
        """Test scalar + signal (radd)."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 3)

        result = 5 + s1
        assert np.allclose(result.values, 8.0)

    def test_reverse_subtraction(self) -> None:
        """Test scalar - signal (rsub)."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 3)

        result = 10 - s1
        assert np.allclose(result.values, 7.0)

    def test_reverse_multiplication(self) -> None:
        """Test scalar * signal (rmul)."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 3)

        result = 4 * s1
        assert np.allclose(result.values, 12.0)

    def test_reverse_division(self) -> None:
        """Test scalar / signal (rtruediv)."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.ones(100) * 4)

        result = 20 / s1
        assert np.allclose(result.values, 5.0)

    def test_subtraction_mismatched_time_raises(self) -> None:
        """Test that subtracting signals with different time arrays raises."""
        t1 = np.linspace(0, 10, 100)
        t2 = np.linspace(0, 20, 100)
        s1 = Signal(t1, np.ones(100))
        s2 = Signal(t2, np.ones(100))

        with pytest.raises(ValueError, match="subtraction"):
            _ = s1 - s2

    def test_division_mismatched_time_raises(self) -> None:
        """Test that dividing signals with different time arrays raises."""
        t1 = np.linspace(0, 10, 100)
        t2 = np.linspace(0, 20, 100)
        s1 = Signal(t1, np.ones(100))
        s2 = Signal(t2, np.ones(100))

        with pytest.raises(ValueError, match="division"):
            _ = s1 / s2


# =============================================================================
# Signal.resample() 2D Tests
# =============================================================================


class TestSignalResample2D:
    """Tests for 2D (multi-channel) signal resampling."""

    def test_resample_1d_signal(self) -> None:
        """Test that 1D resampling still works correctly."""
        t = np.linspace(0, 10, 100)
        values = np.sin(t)
        signal = Signal(t, values)

        resampled = signal.resample(20.0)
        assert resampled.values.ndim == 1
        assert resampled.fs == pytest.approx(20.0, rel=0.1)

    def test_resample_2d_signal(self) -> None:
        """Test resampling for 2D multi-channel signals."""
        t = np.linspace(0, 10, 100)
        values = np.column_stack([np.sin(t), np.cos(t), t])
        signal = Signal(t, values)

        resampled = signal.resample(20.0)
        assert resampled.values.ndim == 2
        assert resampled.values.shape[1] == 3
        # First channel should be approximately sin
        assert np.corrcoef(resampled.values[:, 0], np.sin(resampled.time))[0, 1] > 0.99

    def test_resample_preserves_channels(self) -> None:
        """Test that all channels are preserved after resampling."""
        t = np.linspace(0, 5, 500)
        ch1 = np.sin(2 * np.pi * t)
        ch2 = np.cos(2 * np.pi * t)
        values = np.column_stack([ch1, ch2])
        signal = Signal(t, values)

        resampled = signal.resample(50.0)
        assert resampled.values.shape[1] == 2


# =============================================================================
# Backlash Tests
# =============================================================================


class TestBacklash:
    """Tests for apply_backlash."""

    def test_backlash_zero_width(self) -> None:
        """Test that zero backlash is a passthrough."""
        t = np.linspace(0, 10, 100)
        signal = Signal(t, np.sin(t))

        result = apply_backlash(signal, backlash_width=0.0, smooth=False)
        assert np.allclose(result.values, signal.values)

    def test_backlash_positive_width(self) -> None:
        """Test that backlash reduces signal variation."""
        t = np.linspace(0, 10, 1000)
        signal = Signal(t, np.sin(2 * np.pi * t))

        result = apply_backlash(signal, backlash_width=0.5, smooth=False)
        # Output should have less variation than input
        assert np.std(result.values) < np.std(signal.values)

    def test_backlash_negative_width_raises(self) -> None:
        """Test that negative backlash width raises ValueError."""
        t = np.linspace(0, 10, 100)
        signal = Signal(t, np.sin(t))

        with pytest.raises(ValueError, match="non-negative"):
            apply_backlash(signal, backlash_width=-1.0)

    def test_backlash_non_positive_smoothness_raises(self) -> None:
        """Test that non-positive smoothness raises ValueError."""
        t = np.linspace(0, 10, 100)
        signal = Signal(t, np.sin(t))

        with pytest.raises(ValueError, match="positive"):
            apply_backlash(signal, backlash_width=0.1, smoothness=0.0)


# =============================================================================
# Calculus Extended Tests
# =============================================================================


class TestCalculusExtended:
    """Tests for curvature, arc length, extrema, and inflection points."""

    def test_curvature_of_line(self) -> None:
        """Test curvature of a straight line is near zero."""
        t = np.linspace(0, 10, 1000)
        values = 2.0 * t + 5.0
        signal = Signal(t, values)

        curvature = compute_curvature(signal)
        # Curvature of a line should be ~0
        assert np.mean(np.abs(curvature.values[50:-50])) < 0.1

    def test_arc_length_of_line(self) -> None:
        """Test arc length of a straight line segment."""
        t = np.linspace(0, 1, 1000)
        values = t  # y = t, slope = 1
        signal = Signal(t, values)

        arc_len = compute_arc_length(signal)
        # Arc length = integral of sqrt(1 + (dy/dt)^2) dt
        # For y = t: sqrt(1 + 1) * 1 = sqrt(2)
        assert arc_len == pytest.approx(np.sqrt(2), rel=0.1)

    def test_find_extrema_sinusoid(self) -> None:
        """Test finding extrema of a sinusoid."""
        t = np.linspace(0, 2 * np.pi, 1000)
        values = np.sin(t)
        signal = Signal(t, values)

        maxima, minima = find_extrema(signal)
        # Should find 1 maximum near pi/2 and 1 minimum near 3*pi/2
        assert len(maxima) >= 1
        assert len(minima) >= 1

    def test_find_inflection_points_sine(self) -> None:
        """Test finding inflection points of sine."""
        t = np.linspace(0, 2 * np.pi, 2000)
        values = np.sin(t)
        signal = Signal(t, values)

        inflections = find_inflection_points(signal)
        # sin(t) has inflection points at 0, pi, 2*pi
        assert len(inflections) >= 1


# =============================================================================
# Filter DbC Precondition Tests
# =============================================================================


class TestFilterDbC:
    """Tests for filter DbC preconditions."""

    def test_butterworth_invalid_order_raises(self) -> None:
        """Test that order < 1 raises ValueError."""
        with pytest.raises(ValueError, match="order"):
            FilterDesigner.butterworth(
                FilterType.LOWPASS, cutoff=5.0, fs=100.0, order=0
            )

    def test_chebyshev1_invalid_ripple_raises(self) -> None:
        """Test that ripple_db <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="ripple_db"):
            FilterDesigner.chebyshev1(
                FilterType.LOWPASS, cutoff=5.0, fs=100.0, ripple_db=0.0
            )

    def test_chebyshev2_invalid_attenuation_raises(self) -> None:
        """Test that attenuation_db <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="attenuation_db"):
            FilterDesigner.chebyshev2(
                FilterType.LOWPASS, cutoff=5.0, fs=100.0, attenuation_db=-1.0
            )

    def test_elliptic_invalid_params_raises(self) -> None:
        """Test that invalid elliptic params raise ValueError."""
        with pytest.raises(ValueError, match="order"):
            FilterDesigner.elliptic(FilterType.LOWPASS, cutoff=5.0, fs=100.0, order=0)

    def test_bessel_invalid_order_raises(self) -> None:
        """Test that Bessel filter with order < 1 raises."""
        with pytest.raises(ValueError, match="order"):
            FilterDesigner.bessel(FilterType.LOWPASS, cutoff=5.0, fs=100.0, order=0)

    def test_bessel_filter_design(self) -> None:
        """Test Bessel filter can be designed and applied."""
        t = np.linspace(0, 10, 1000)
        signal = Signal(t, np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 20 * t))

        spec = FilterDesigner.bessel(FilterType.LOWPASS, cutoff=5.0, fs=100.0, order=4)
        filtered = apply_filter(signal, spec)

        assert len(filtered.values) == len(signal.values)
        assert np.std(filtered.values) < np.std(signal.values)

    def test_bandpass_requires_tuple(self) -> None:
        """Test that bandpass filter requires (low, high) cutoff tuple."""
        with pytest.raises(ValueError, match="tuple"):
            FilterDesigner.butterworth(FilterType.BANDPASS, cutoff=5.0, fs=100.0)

    def test_negative_fs_raises(self) -> None:
        """Test that negative sampling frequency raises."""
        with pytest.raises(ValueError, match="positive"):
            FilterDesigner.butterworth(FilterType.LOWPASS, cutoff=5.0, fs=-10.0)


# =============================================================================
# Adaptive Filter Tests
# =============================================================================


class TestAdaptiveFilter:
    """Tests for LMS and RLS adaptive filters."""

    def test_extracted_module_matches_public_api(self) -> None:
        """Test that the extracted module preserves the public class identity."""
        assert ExtractedAdaptiveFilter is AdaptiveFilter

    def test_lms_filter_output_length(self) -> None:
        """Test that LMS filter output has correct length."""
        t = np.linspace(0, 10, 500)
        signal = Signal(t, np.sin(2 * np.pi * t))
        reference = Signal(t, np.sin(2 * np.pi * t) + 0.5)

        filtered, error = AdaptiveFilter.lms(signal, reference, order=10)
        assert len(filtered.values) == len(signal.values)
        assert len(error.values) == len(signal.values)

    def test_rls_filter_output_length(self) -> None:
        """Test that RLS filter output has correct length."""
        t = np.linspace(0, 10, 500)
        signal = Signal(t, np.sin(2 * np.pi * t))
        reference = Signal(t, np.sin(2 * np.pi * t) + 0.5)

        filtered, error = AdaptiveFilter.rls(signal, reference, order=10)
        assert len(filtered.values) == len(signal.values)
        assert len(error.values) == len(signal.values)

    def test_lms_converges(self) -> None:
        """Test that LMS filter error decreases over time."""
        t = np.linspace(0, 10, 1000)
        desired = np.sin(2 * np.pi * t)
        reference = Signal(t, desired)
        signal = Signal(t, desired + np.random.randn(1000) * 0.1)

        _, error = AdaptiveFilter.lms(signal, reference, order=20, step_size=0.01)

        # Error at the end should be smaller than at the start
        early_error = np.mean(np.abs(error.values[20:100]))
        late_error = np.mean(np.abs(error.values[800:]))
        assert late_error < early_error


# =============================================================================
# Saturation Postcondition Tests
# =============================================================================


class TestSaturationPostconditions:
    """Test that all saturation modes guarantee bounds."""

    @pytest.mark.parametrize("mode", list(SaturationMode))
    def test_saturation_respects_bounds(self, mode: SaturationMode) -> None:
        """Test that all saturation modes stay within [lower, upper]."""
        t = np.linspace(0, 10, 1000)
        values = np.linspace(-10, 10, 1000)
        signal = Signal(t, values)

        lower, upper = -2.0, 3.0
        saturated = apply_saturation(signal, lower=lower, upper=upper, mode=mode)

        assert max(saturated.values) <= upper + 1e-10
        assert min(saturated.values) >= lower - 1e-10


# =============================================================================
# BatchProcessor Tests
# =============================================================================


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    def test_batch_process_files(self) -> None:
        """Test batch processing of CSV files."""
        t = np.linspace(0, 10, 100)
        s1 = Signal(t, np.sin(t), name="sin")
        s2 = Signal(t, np.cos(t), name="cos")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            p1 = Path(tmpdir) / "sig1.csv"
            p2 = Path(tmpdir) / "sig2.csv"
            SignalExporter.to_csv(s1, p1)
            SignalExporter.to_csv(s2, p2)

            # Batch process with identity transform
            processor = BatchProcessor(tmpdir)
            results = processor.process_all(
                processor=lambda s: s,
                pattern="*.csv",
            )

            assert len(results) >= 2


# =============================================================================
# Noise Spectral Properties Tests
# =============================================================================


class TestNoiseSpectralProperties:
    """Test spectral properties of different noise types."""

    def test_white_noise_flat_spectrum(self) -> None:
        """Test that white noise has approximately flat PSD."""
        t = np.linspace(0, 10, 10000)
        gen = NoiseGenerator(seed=42)
        noise = gen.generate(t, NoiseType.WHITE, amplitude=1.0)

        # Compute PSD
        from scipy.signal import welch

        fs = 1.0 / np.mean(np.diff(t))
        freqs, psd = welch(noise.values, fs=fs)

        # White noise should have roughly flat spectrum
        # Check that max/min ratio isn't too extreme
        valid = psd[psd > 0]
        ratio = np.max(valid) / np.min(valid)
        assert ratio < 100  # Should be relatively flat

    def test_brown_noise_decreasing_spectrum(self) -> None:
        """Test that brown noise has decreasing PSD with frequency."""
        t = np.linspace(0, 10, 10000)
        gen = NoiseGenerator(seed=42)
        noise = gen.generate(t, NoiseType.BROWN, amplitude=1.0)

        from scipy.signal import welch

        fs = 1.0 / np.mean(np.diff(t))
        freqs, psd = welch(noise.values, fs=fs)

        # Brown noise: low frequencies should have more power
        mid = len(psd) // 2
        low_power = np.mean(psd[1:mid])
        high_power = np.mean(psd[mid:])
        assert low_power > high_power

    def test_noise_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same noise."""
        t = np.linspace(0, 10, 1000)
        gen1 = NoiseGenerator(seed=123)
        gen2 = NoiseGenerator(seed=123)

        noise1 = gen1.generate(t, NoiseType.WHITE, amplitude=1.0)
        noise2 = gen2.generate(t, NoiseType.WHITE, amplitude=1.0)

        assert np.allclose(noise1.values, noise2.values)


class TestSignalSliceDbC:
    """DbC tests for Signal.slice()."""

    def test_slice_returns_subset(self) -> None:
        """Test that slice returns correct time range."""
        t = np.linspace(0, 10, 1001)
        signal = Signal(t, np.sin(t))

        sliced = signal.slice(2.0, 5.0)
        assert sliced.time[0] >= 2.0
        assert sliced.time[-1] <= 5.0
        assert len(sliced.time) < len(signal.time)

    def test_slice_preserves_metadata(self) -> None:
        """Test that slice preserves signal metadata."""
        t = np.linspace(0, 10, 100)
        signal = Signal(t, np.sin(t), name="test", units="m/s")

        sliced = signal.slice(2.0, 5.0)
        assert sliced.name == "test"
        assert sliced.units == "m/s"


class TestSignalGeneratorExtended:
    """Extended tests for SignalGenerator."""

    def test_superposition(self) -> None:
        """Test signal superposition."""
        t = np.linspace(0, 10, 100)
        s1 = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0)
        s2 = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=2.0)

        result = SignalGenerator.superposition([s1, s2])
        expected = s1.values + s2.values
        assert np.allclose(result.values, expected)

    def test_superposition_empty_raises(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            SignalGenerator.superposition([])

    def test_sawtooth_wave(self) -> None:
        """Test sawtooth wave generation."""
        t = np.linspace(0, 2, 1000)
        signal = SignalGenerator.sawtooth(t, frequency=2.0, amplitude=1.0)
        assert max(signal.values) <= 1.1
        assert min(signal.values) >= -1.1

    def test_triangle_wave(self) -> None:
        """Test triangle wave generation."""
        t = np.linspace(0, 2, 1000)
        signal = SignalGenerator.triangle(t, frequency=2.0, amplitude=1.0)
        assert max(signal.values) <= 1.1
        assert min(signal.values) >= -1.1

    def test_square_wave(self) -> None:
        """Test square wave generation."""
        t = np.linspace(0, 2, 1000)
        signal = SignalGenerator.square(t, frequency=2.0, amplitude=1.0)
        unique_vals = np.unique(np.round(signal.values, 1))
        assert len(unique_vals) <= 2

    def test_pulse_generation(self) -> None:
        """Test rectangular pulse."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.pulse(t, start_time=2.0, duration=3.0, amplitude=5.0)
        assert signal.values[0] == 0.0  # Before pulse
        # During pulse
        mid_idx = np.argmin(np.abs(t - 3.5))
        assert signal.values[mid_idx] == 5.0


# =============================================================================
# Contracts fallback (Issue #1280)
# =============================================================================


class TestContractsFallback:
    """Tests for contracts package fallback behavior."""

    def test_require_fallback_passes_truthy(self) -> None:
        """Fallback require should pass on truthy condition."""
        from signal_toolkit.core import require

        # Should not raise
        require(True, "should pass")
        require(1, "should pass")

    def test_require_fallback_raises_on_falsy(self) -> None:
        """Fallback require should raise ValueError on falsy condition."""
        from signal_toolkit.core import require

        with pytest.raises((ValueError, Exception)):
            require(False, "this should fail")


# =============================================================================
# Polynomial DRY (Issue #1282)
# =============================================================================


class TestPolynomialDry:
    """Tests verifying polynomial evaluation consistency."""

    def test_polyval_matches_signal_generator(self) -> None:
        """np.polyval and SignalGenerator.polynomial should produce same results."""
        t = np.linspace(0, 10, 100)
        # Ascending order: c0 + c1*t + c2*t^2
        ascending = [1.0, 2.0, 0.5]
        sig = SignalGenerator.polynomial(t, ascending)

        # np.polyval uses descending order
        descending = ascending[::-1]
        t_shifted = t - t[0]
        expected = np.polyval(descending, t_shifted)

        np.testing.assert_allclose(sig.values, expected, atol=1e-10)


# =============================================================================
# Series Expansion (Issue #1279)
# =============================================================================


class TestSeriesExpansionIntegration:
    """Integration tests for SeriesExpansion with Signal data."""

    def test_maclaurin_of_sin(self) -> None:
        """Maclaurin series of sin(x) should converge near x=0."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion(max_terms=10)
        result = expansion.get_series_result(np.sin, center=0.0, n_terms=8)

        # Near zero, approximation should be close
        x_test = np.array([0.1, 0.5])
        approx = result.function(x_test)
        exact = np.sin(x_test)
        np.testing.assert_allclose(approx, exact, atol=1e-4)

    def test_taylor_of_exp(self) -> None:
        """Taylor series of exp(x) centered at 1 should converge near x=1."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion(max_terms=10)
        result = expansion.get_series_result(np.exp, center=1.0, n_terms=8)

        x_test = np.array([0.8, 1.2])
        approx = result.function(x_test)
        exact = np.exp(x_test)
        np.testing.assert_allclose(approx, exact, atol=1e-3)


# =============================================================================
# FilterSpec Bode Plot (Issue #1278)
# =============================================================================


class TestFilterSpecFrequencyResponse:
    """Tests for FilterSpec.get_frequency_response."""

    def test_butterworth_freq_response(self) -> None:
        """Butterworth filter should have monotonic magnitude rolloff."""
        spec = FilterDesigner.butterworth(
            FilterType.LOWPASS, cutoff=10.0, fs=100.0, order=4
        )
        freqs, magnitude, phase = spec.get_frequency_response(256)

        assert len(freqs) == 256
        assert len(magnitude) == 256
        assert len(phase) == 256
        # DC gain should be ~1
        assert magnitude[0] > 0.9
        # High-frequency should be attenuated
        assert magnitude[-1] < 0.01

    def test_bessel_freq_response(self) -> None:
        """Bessel filter should also return valid frequency response."""
        spec = FilterDesigner.bessel(FilterType.LOWPASS, cutoff=10.0, fs=100.0, order=4)
        freqs, magnitude, phase = spec.get_frequency_response(128)

        assert len(freqs) == 128
        assert magnitude[0] > 0.9
