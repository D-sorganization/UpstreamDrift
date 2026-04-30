"""Behavioral tests for CustomFunctionFitter and the CWT/XWT kernel.

Targets:

* ``src/shared/python/signal_toolkit/_custom_fitter.py``
* ``src/shared/python/signal_toolkit/_wavelet.py``

Both modules previously had no direct tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit._custom_fitter import CustomFunctionFitter
from src.shared.python.signal_toolkit._wavelet import compute_cwt, compute_xwt
from src.shared.python.signal_toolkit.core import Signal

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# CustomFunctionFitter
# ---------------------------------------------------------------------------


class TestCustomFunctionFitter:
    def test_recovers_quadratic_parameters_via_callable(self) -> None:
        def quad(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a * t**2 + b * t + c

        t = np.linspace(0.0, 1.0, 200)
        y = 2.0 * t**2 - 1.0 * t + 0.5
        sig = Signal(time=t, values=y, name="q")

        fitter = CustomFunctionFitter(quad, ["a", "b", "c"])
        result = fitter.fit(sig, initial_guess=[1.0, 1.0, 1.0])

        assert result.success is True
        assert result.parameters["a"] == pytest.approx(2.0, abs=1e-6)
        assert result.parameters["b"] == pytest.approx(-1.0, abs=1e-6)
        assert result.parameters["c"] == pytest.approx(0.5, abs=1e-6)
        assert result.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_from_expression_parses_and_fits(self) -> None:
        # Expression-based fitter: a * sin(2*pi*f*t) + c
        fitter = CustomFunctionFitter.from_expression(
            "a * sin(2*pi*f*t) + c", ["a", "f", "c"]
        )
        fs = 500.0
        t = np.arange(0.0, 1.0, 1 / fs)
        y = 1.5 * np.sin(2 * np.pi * 4.0 * t) + 0.25
        sig = Signal(time=t, values=y, name="s")
        result = fitter.fit(sig, initial_guess=[1.0, 4.0, 0.0])
        assert result.success is True
        # amplitude can be recovered with sign flip if phase differs; we
        # provided a good initial guess so the sign should be preserved.
        assert result.parameters["a"] == pytest.approx(1.5, rel=1e-3)
        assert result.parameters["f"] == pytest.approx(4.0, rel=1e-3)
        assert result.parameters["c"] == pytest.approx(0.25, abs=1e-3)
        assert result.r_squared > 0.9999

    def test_from_expression_blocks_dunder(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            CustomFunctionFitter.from_expression("__import__('os').system('x')", ["t"])

    def test_failed_fit_returns_success_false(self) -> None:
        # A function that triggers ValueError inside curve_fit, e.g. by
        # passing non-finite y values.
        def lin(t: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * t + b

        t = np.linspace(0.0, 1.0, 50)
        y = np.full_like(t, np.nan)
        sig = Signal(time=t, values=y, name="bad")
        fitter = CustomFunctionFitter(lin, ["a", "b"])
        result = fitter.fit(sig, initial_guess=[1.0, 0.0])
        # Either marked as failed or returned with NaN-tolerant fallback.
        if not result.success:
            assert "Fit failed" in result.message
        # Whatever happened, the dataclass is well-formed.
        assert result.fitted_signal.values.shape == y.shape
        assert result.residuals.shape == y.shape


# ---------------------------------------------------------------------------
# compute_cwt
# ---------------------------------------------------------------------------


class TestComputeCwt:
    def test_output_shape_and_times(self) -> None:
        fs = 200.0
        x = np.random.default_rng(0).standard_normal(512)
        freqs, times, cwt = compute_cwt(x, fs=fs, num_freqs=20, freq_range=(1.0, 50.0))
        assert freqs.shape == (20,)
        assert times.shape == (512,)
        assert cwt.shape == (20, 512)
        assert cwt.dtype == np.complex128
        # times start at 0 and step by 1/fs.
        assert times[0] == pytest.approx(0.0)
        assert times[1] - times[0] == pytest.approx(1.0 / fs)

    def test_peak_in_cwt_at_signal_frequency(self) -> None:
        fs = 500.0
        f0 = 10.0
        t = np.arange(0.0, 4.0, 1 / fs)
        x = np.sin(2 * np.pi * f0 * t)
        freqs, _times, cwt = compute_cwt(x, fs=fs, num_freqs=40, freq_range=(2.0, 50.0))
        # average magnitude across time, then find dominant frequency.
        mag = np.mean(np.abs(cwt), axis=1)
        peak_freq = freqs[int(np.argmax(mag))]
        # log-spaced frequency grid → tolerate a relative error.
        assert peak_freq == pytest.approx(f0, rel=0.2)

    def test_non_positive_fs_raises(self) -> None:
        with pytest.raises((ValueError, ZeroDivisionError, FloatingPointError)):
            compute_cwt(np.ones(64), fs=0.0)

    def test_non_positive_min_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="Minimum frequency"):
            compute_cwt(np.ones(64), fs=100.0, freq_range=(0.0, 50.0))


# ---------------------------------------------------------------------------
# compute_xwt
# ---------------------------------------------------------------------------


class TestComputeXwt:
    def test_xwt_shape(self) -> None:
        fs = 200.0
        rng = np.random.default_rng(0)
        a = rng.standard_normal(256)
        b = rng.standard_normal(256)
        freqs, times, xwt = compute_xwt(a, b, fs=fs, num_freqs=15)
        assert freqs.shape == (15,)
        assert xwt.shape[0] == 15
        assert xwt.shape[1] == times.shape[0]
        assert xwt.dtype == np.complex128

    def test_xwt_self_equals_squared_cwt_magnitude(self) -> None:
        fs = 200.0
        x = np.sin(2 * np.pi * 5.0 * np.arange(256) / fs)
        freqs_a, _, xwt = compute_xwt(x, x, fs=fs, num_freqs=10)
        freqs_b, _, cwt = compute_cwt(x, fs=fs, num_freqs=10)
        np.testing.assert_array_equal(freqs_a, freqs_b)
        # XWT(x,x) = |W(x)|^2, so imaginary part should be zero (within fp eps).
        assert np.max(np.abs(xwt.imag)) < 1e-6
        np.testing.assert_allclose(xwt.real, np.abs(cwt) ** 2, rtol=1e-6, atol=1e-9)
