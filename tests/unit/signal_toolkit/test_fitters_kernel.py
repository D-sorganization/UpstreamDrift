"""Behavioral tests for the parametric curve-fitting kernels.

Targets the previously-untested modules:

* ``src/shared/python/signal_toolkit/_linear_polynomial_fitters.py``
* ``src/shared/python/signal_toolkit/_exponential_fitter.py``
* ``src/shared/python/signal_toolkit/_sinusoidal_fitters.py``
* ``src/shared/python/signal_toolkit/_fit_result.py``

For each fitter we synthesise a signal from known ground-truth parameters
and assert that:

* the fitted parameters round-trip to the truth within tolerance,
* R^2 is essentially 1.0 on noiseless data,
* RMSE is small,
* the produced ``FitResult`` carries the expected schema and a sensible
  ``fitted_signal`` and ``residuals``,
* error paths raise on empty / malformed input.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.signal_toolkit._exponential_fitter import ExponentialFitter
from src.shared.python.signal_toolkit._fit_result import FitResult
from src.shared.python.signal_toolkit._linear_polynomial_fitters import (
    LinearFitter,
    PolynomialFitter,
)
from src.shared.python.signal_toolkit._sinusoidal_fitters import SinusoidFitter
from src.shared.python.signal_toolkit.core import Signal

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(t: np.ndarray, y: np.ndarray, name: str = "x") -> Signal:
    return Signal(time=t, values=y, name=name, units="m")


# ---------------------------------------------------------------------------
# LinearFitter
# ---------------------------------------------------------------------------


class TestLinearFitter:
    def test_recovers_slope_and_intercept_noiseless(self) -> None:
        t = np.linspace(0.0, 10.0, 200)
        slope, intercept = 2.5, -1.25
        sig = _make_signal(t, slope * t + intercept)
        result = LinearFitter().fit(sig)
        assert result.parameters["slope"] == pytest.approx(slope, rel=1e-9, abs=1e-9)
        assert result.parameters["intercept"] == pytest.approx(
            intercept, rel=1e-9, abs=1e-9
        )
        assert result.r_squared == pytest.approx(1.0, abs=1e-9)
        assert result.rmse == pytest.approx(0.0, abs=1e-9)
        assert result.success is True

    def test_recovers_with_small_noise(self) -> None:
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 5.0, 500)
        sig = _make_signal(t, 1.7 * t + 0.4 + 0.01 * rng.standard_normal(t.size))
        result = LinearFitter().fit(sig)
        assert result.parameters["slope"] == pytest.approx(1.7, rel=5e-3)
        assert result.parameters["intercept"] == pytest.approx(0.4, abs=5e-3)
        assert 0.99 < result.r_squared <= 1.0

    def test_fitted_signal_matches_input_time(self) -> None:
        t = np.linspace(0.0, 1.0, 50)
        sig = _make_signal(t, 3 * t)
        result = LinearFitter().fit(sig)
        assert isinstance(result.fitted_signal, Signal)
        np.testing.assert_array_equal(result.fitted_signal.time, sig.time)
        assert result.fitted_signal.values.shape == sig.values.shape
        assert result.residuals.shape == sig.values.shape

    def test_empty_signal_raises_precondition(self) -> None:
        sig = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises(PreconditionError, match="non-empty"):
            LinearFitter().fit(sig)

    def test_constant_signal_zero_slope(self) -> None:
        t = np.linspace(0.0, 5.0, 100)
        sig = _make_signal(t, np.full_like(t, 7.5))
        result = LinearFitter().fit(sig)
        assert result.parameters["slope"] == pytest.approx(0.0, abs=1e-9)
        assert result.parameters["intercept"] == pytest.approx(7.5, abs=1e-9)
        # ss_tot is zero for a constant signal → r_squared is the documented
        # degenerate-case fallback (0.0).
        assert result.r_squared == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# PolynomialFitter
# ---------------------------------------------------------------------------


class TestPolynomialFitter:
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
    def test_recovers_polynomial_of_each_order(self, order: int) -> None:
        rng = np.random.default_rng(order)
        true_coeffs = rng.uniform(-2.0, 2.0, size=order + 1)  # c0..cn
        t = np.linspace(0.0, 1.0, 200)
        y = sum(c * t**i for i, c in enumerate(true_coeffs))
        sig = _make_signal(t, y)
        result = PolynomialFitter(order=order).fit(sig)
        for i, expected in enumerate(true_coeffs):
            assert result.parameters[f"c{i}"] == pytest.approx(expected, abs=1e-6)
        assert result.r_squared == pytest.approx(1.0, abs=1e-9)
        assert result.rmse == pytest.approx(0.0, abs=1e-6)

    def test_negative_order_raises(self) -> None:
        sig = _make_signal(np.linspace(0, 1, 5), np.zeros(5))
        with pytest.raises(PreconditionError, match="order >= 0"):
            PolynomialFitter(order=2).fit(sig, order=-1)

    def test_empty_signal_raises(self) -> None:
        sig = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises(PreconditionError, match="non-empty"):
            PolynomialFitter().fit(sig)

    def test_order_overrides_default(self) -> None:
        t = np.linspace(0, 1, 100)
        # Pure quadratic; default order is 6 but we ask for 2.
        y = 1.0 + 2.0 * t + 3.0 * t**2
        result = PolynomialFitter(order=6).fit(_make_signal(t, y), order=2)
        assert result.parameters["c0"] == pytest.approx(1.0, abs=1e-6)
        assert result.parameters["c1"] == pytest.approx(2.0, abs=1e-6)
        assert result.parameters["c2"] == pytest.approx(3.0, abs=1e-6)
        assert "c3" not in result.parameters

    def test_get_coefficients_array_round_trip(self) -> None:
        params = {"c0": 1.5, "c2": -0.5, "c1": 0.25}
        coeffs = PolynomialFitter().get_coefficients_array(params)
        assert coeffs.tolist() == [1.5, 0.25, -0.5]

    def test_few_samples_clamps_order_without_error(self) -> None:
        # Only 2 samples but default order is 6 → should clamp & succeed.
        sig = _make_signal(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        result = PolynomialFitter(order=6).fit(sig)
        assert result.success is True
        assert result.parameters["c0"] == pytest.approx(0.0, abs=1e-9)
        assert result.parameters["c1"] == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# ExponentialFitter
# ---------------------------------------------------------------------------


class TestExponentialFitter:
    def test_recovers_decay_parameters(self) -> None:
        t = np.linspace(0.0, 5.0, 500)
        amplitude, decay_rate, offset = 2.0, 1.3, 0.5
        y = amplitude * np.exp(-decay_rate * t) + offset
        sig = _make_signal(t, y)
        result = ExponentialFitter().fit_decay(sig)
        assert result.success is True
        assert result.parameters["amplitude"] == pytest.approx(amplitude, rel=1e-3)
        assert result.parameters["decay_rate"] == pytest.approx(decay_rate, rel=1e-3)
        assert result.parameters["offset"] == pytest.approx(offset, abs=1e-3)
        assert result.r_squared > 0.9999
        assert result.covariance is not None

    def test_recovers_growth_parameters(self) -> None:
        t = np.linspace(0.0, 5.0, 500)
        amplitude, growth_rate, offset = 3.0, 0.8, 1.0
        y = amplitude * (1 - np.exp(-growth_rate * t)) + offset
        sig = _make_signal(t, y)
        result = ExponentialFitter().fit_growth(sig)
        assert result.success is True
        assert result.parameters["amplitude"] == pytest.approx(amplitude, rel=1e-3)
        assert result.parameters["growth_rate"] == pytest.approx(growth_rate, rel=1e-3)
        assert result.parameters["offset"] == pytest.approx(offset, abs=1e-3)
        assert result.r_squared > 0.9999

    def test_decay_empty_signal_raises(self) -> None:
        sig = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises(PreconditionError, match="non-empty"):
            ExponentialFitter().fit_decay(sig)

    def test_decay_residuals_sum_near_zero_for_clean_data(self) -> None:
        t = np.linspace(0.0, 4.0, 400)
        y = 1.5 * np.exp(-0.7 * t) + 0.0
        sig = _make_signal(t, y)
        result = ExponentialFitter().fit_decay(sig)
        # residuals are essentially numerical noise.
        assert np.max(np.abs(result.residuals)) < 1e-3
        assert result.fitted_signal.values.shape == sig.values.shape


# ---------------------------------------------------------------------------
# SinusoidFitter
# ---------------------------------------------------------------------------


class TestSinusoidFitter:
    def test_estimate_initial_params_basic(self) -> None:
        # FFT-based estimator should pick a frequency near the truth.
        fs = 200.0
        t = np.arange(0, 2.0, 1 / fs)
        true_freq = 5.0
        y = 1.7 * np.sin(2 * np.pi * true_freq * t + 0.3) + 0.25
        amp, freq, _phase, offset = SinusoidFitter.estimate_initial_params(t, y)
        assert freq == pytest.approx(true_freq, abs=0.5)
        assert offset == pytest.approx(0.25, abs=0.05)
        assert amp > 0  # always positive by construction

    def test_recovers_sinusoid_parameters(self) -> None:
        fs = 1000.0
        t = np.arange(0, 1.0, 1 / fs)
        amplitude, frequency, phase, offset = 1.5, 7.0, 0.4, -0.2
        y = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
        sig = _make_signal(t, y)
        result = SinusoidFitter().fit(sig)
        assert result.success is True
        assert result.parameters["amplitude"] == pytest.approx(amplitude, rel=1e-3)
        assert result.parameters["frequency"] == pytest.approx(frequency, rel=1e-3)
        assert result.parameters["offset"] == pytest.approx(offset, abs=1e-3)
        # Phase comparison wraps modulo 2*pi.
        phase_err = (result.parameters["phase"] - phase + np.pi) % (2 * np.pi) - np.pi
        assert abs(phase_err) < 1e-2
        assert result.r_squared > 0.9999

    def test_empty_signal_raises(self) -> None:
        sig = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises(PreconditionError, match="non-empty"):
            SinusoidFitter().fit(sig)


# ---------------------------------------------------------------------------
# FitResult dataclass
# ---------------------------------------------------------------------------


class TestFitResult:
    def test_function_string_includes_r_squared(self) -> None:
        sig = _make_signal(np.linspace(0, 1, 5), np.zeros(5))
        result = FitResult(
            parameters={"a": 1.0},
            covariance=None,
            r_squared=0.9876,
            rmse=0.1,
            fitted_signal=sig,
            residuals=np.zeros(5),
        )
        s = result.get_function_string()
        assert "0.9876" in s
        assert "R^2" in s

    def test_default_success_message(self) -> None:
        sig = _make_signal(np.linspace(0, 1, 3), np.zeros(3))
        result = FitResult(
            parameters={},
            covariance=None,
            r_squared=1.0,
            rmse=0.0,
            fitted_signal=sig,
            residuals=np.zeros(3),
        )
        assert result.success is True
        assert result.message == ""
