"""Unit tests for signal_toolkit/fitting.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.core import Signal, SignalGenerator
from src.shared.python.signal_toolkit.fitting import (
    CosineFitter,
    CustomFunctionFitter,
    ExponentialFitter,
    FitResult,
    FunctionFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)


@pytest.fixture
def t200() -> np.ndarray:
    return np.linspace(0, 2.0, 200)


@pytest.fixture
def sine_2hz(t200: np.ndarray) -> Signal:
    vals = 3.0 * np.sin(2 * np.pi * 2.0 * t200)
    return Signal(time=t200, values=vals, name="sine_2hz", units="mV")


@pytest.fixture
def linear_signal(t200: np.ndarray) -> Signal:
    return SignalGenerator.linear(t200, slope=4.0, intercept=2.0, name="linear")


@pytest.fixture
def decay_signal(t200: np.ndarray) -> Signal:
    return SignalGenerator.exponential(t200, amplitude=5.0, decay_rate=1.5, name="decay")


class TestSinusoidFitter:
    """Tests for SinusoidFitter."""

    def test_fit_recovers_params(self, sine_2hz: Signal) -> None:
        fitter = SinusoidFitter()
        result = fitter.fit(sine_2hz)
        assert isinstance(result, FitResult)
        assert result.r_squared > 0.95
        assert result.success

    def test_estimate_initial_params(self, t200: np.ndarray, sine_2hz: Signal) -> None:
        amp, freq, phase, offset = SinusoidFitter.estimate_initial_params(t200, sine_2hz.values)
        assert amp > 0
        assert freq > 0

    def test_fit_string(self, sine_2hz: Signal) -> None:
        fitter = SinusoidFitter()
        result = fitter.fit(sine_2hz)
        s = fitter.get_function_string(result.parameters)
        assert "sin" in s

    def test_fit_result_string(self, sine_2hz: Signal) -> None:
        fitter = SinusoidFitter()
        result = fitter.fit(sine_2hz)
        s = result.get_function_string()
        assert "R^2" in s

    def test_fit_with_initial_guess(self, sine_2hz: Signal) -> None:
        fitter = SinusoidFitter()
        result = fitter.fit(sine_2hz, initial_guess=(3.0, 2.0, 0.0, 0.0))
        assert result.r_squared > 0.9

    def test_fit_empty_signal_raises(self) -> None:
        fitter = SinusoidFitter()
        empty = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises((ValueError, AssertionError)):
            fitter.fit(empty)


class TestCosineFitter:
    """Tests for CosineFitter."""

    def test_fit_cosine(self, t200: np.ndarray) -> None:
        vals = 2.0 * np.cos(2 * np.pi * 1.5 * t200)
        sig = Signal(time=t200, values=vals, name="cos")
        fitter = CosineFitter()
        result = fitter.fit(sig)
        assert result.r_squared > 0.9

    def test_get_function_string(self, t200: np.ndarray) -> None:
        fitter = CosineFitter()
        s = fitter.get_function_string(
            {"amplitude": 1.0, "frequency": 2.0, "phase": 0.0, "offset": 0.0}
        )
        assert "cos" in s


class TestExponentialFitter:
    """Tests for ExponentialFitter."""

    def test_fit_decay(self, decay_signal: Signal) -> None:
        fitter = ExponentialFitter()
        result = fitter.fit_decay(decay_signal)
        assert result.success
        assert result.r_squared > 0.95

    def test_fit_decay_with_guess(self, decay_signal: Signal) -> None:
        fitter = ExponentialFitter()
        result = fitter.fit_decay(decay_signal, initial_guess=(5.0, 1.5, 0.0))
        assert result.r_squared > 0.95

    def test_fit_decay_empty_raises(self) -> None:
        fitter = ExponentialFitter()
        empty = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises((ValueError, AssertionError)):
            fitter.fit_decay(empty)

    def test_fit_growth(self, t200: np.ndarray) -> None:
        vals = 3.0 * (1 - np.exp(-2.0 * (t200 - t200[0])))
        sig = Signal(time=t200, values=vals, name="growth")
        fitter = ExponentialFitter()
        result = fitter.fit_growth(sig)
        assert isinstance(result, FitResult)

    def test_fit_growth_with_guess(self, t200: np.ndarray) -> None:
        vals = 3.0 * (1 - np.exp(-2.0 * (t200 - t200[0])))
        sig = Signal(time=t200, values=vals, name="growth")
        fitter = ExponentialFitter()
        result = fitter.fit_growth(sig, initial_guess=(3.0, 2.0, 0.0))
        assert result.r_squared > 0.9


class TestLinearFitter:
    """Tests for LinearFitter."""

    def test_fit_linear(self, linear_signal: Signal) -> None:
        fitter = LinearFitter()
        result = fitter.fit(linear_signal)
        assert np.isclose(result.parameters["slope"], 4.0, atol=0.1)
        assert result.r_squared > 0.99

    def test_get_function_string(self) -> None:
        fitter = LinearFitter()
        s = fitter.get_function_string({"slope": 2.0, "intercept": 1.0})
        assert "slope" not in s  # Should be formatted as equation
        assert "2.0" in s

    def test_fit_empty_raises(self) -> None:
        fitter = LinearFitter()
        empty = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises((ValueError, AssertionError)):
            fitter.fit(empty)


class TestPolynomialFitter:
    """Tests for PolynomialFitter."""

    def test_fit_quadratic(self, t200: np.ndarray) -> None:
        vals = 2.0 * (t200 - t200[0]) ** 2 + 1.0
        sig = Signal(time=t200, values=vals, name="quad")
        fitter = PolynomialFitter(order=2)
        result = fitter.fit(sig)
        assert result.r_squared > 0.99

    def test_fit_with_override_order(self, t200: np.ndarray) -> None:
        vals = t200.copy()
        sig = Signal(time=t200, values=vals, name="linear")
        fitter = PolynomialFitter(order=6)
        result = fitter.fit(sig, order=1)
        assert result.r_squared > 0.99

    def test_fit_empty_raises(self) -> None:
        fitter = PolynomialFitter()
        empty = Signal(time=np.array([]), values=np.array([]))
        with pytest.raises((ValueError, AssertionError)):
            fitter.fit(empty)

    def test_fit_negative_order_raises(self, t200: np.ndarray) -> None:
        sig = Signal(time=t200, values=t200)
        fitter = PolynomialFitter()
        with pytest.raises((ValueError, AssertionError)):
            fitter.fit(sig, order=-1)

    def test_get_coefficients_array(self) -> None:
        fitter = PolynomialFitter()
        params = {"c0": 1.0, "c1": 2.0, "c2": 3.0}
        coeffs = fitter.get_coefficients_array(params)
        assert np.allclose(coeffs, [1.0, 2.0, 3.0])

    def test_fit_few_points_reduces_order(self) -> None:
        """When fewer points than order+1, order is clamped."""
        t = np.linspace(0, 1, 3)
        sig = Signal(time=t, values=t)
        fitter = PolynomialFitter(order=10)
        result = fitter.fit(sig)
        assert result.success


class TestCustomFunctionFitter:
    """Tests for CustomFunctionFitter."""

    def test_fit_custom_func(self, t200: np.ndarray) -> None:
        vals = 2.5 * np.sin(2 * np.pi * 3.0 * (t200 - t200[0]))
        sig = Signal(time=t200, values=vals, name="custom")

        def my_model(t: np.ndarray, a: float, f: float) -> np.ndarray:
            return a * np.sin(2 * np.pi * f * t)

        fitter = CustomFunctionFitter(my_model, param_names=["a", "f"])
        result = fitter.fit(sig, initial_guess=[2.5, 3.0])
        assert result.r_squared > 0.9

    def test_from_expression(self, t200: np.ndarray) -> None:
        vals = 2.0 * (t200 - t200[0]) + 1.0
        sig = Signal(time=t200, values=vals, name="linear")
        fitter = CustomFunctionFitter.from_expression("a * t + b", ["a", "b"])
        result = fitter.fit(sig, initial_guess=[2.0, 1.0])
        assert result.success

    def test_from_expression_forbidden_pattern_raises(self) -> None:
        with pytest.raises(ValueError):
            CustomFunctionFitter.from_expression("__import__('os')", ["a"])

    def test_fit_with_bounds(self, t200: np.ndarray) -> None:
        vals = np.sin(2 * np.pi * 1.0 * (t200 - t200[0]))
        sig = Signal(time=t200, values=vals, name="sin_1hz")

        def model(t: np.ndarray, a: float) -> np.ndarray:
            return a * np.sin(2 * np.pi * 1.0 * t)

        fitter = CustomFunctionFitter(model, param_names=["a"])
        result = fitter.fit(sig, initial_guess=[1.0], bounds=([0.1], [5.0]))
        assert result.success


class TestFunctionFitter:
    """Tests for unified FunctionFitter interface."""

    def test_fit_sinusoid(self, sine_2hz: Signal) -> None:
        ff = FunctionFitter()
        result = ff.fit_sinusoid(sine_2hz)
        assert result.r_squared > 0.9

    def test_fit_cosine(self, t200: np.ndarray) -> None:
        vals = 2.0 * np.cos(2 * np.pi * 1.0 * t200)
        sig = Signal(time=t200, values=vals, name="cos")
        ff = FunctionFitter()
        result = ff.fit_cosine(sig)
        assert isinstance(result, FitResult)

    def test_fit_linear(self, linear_signal: Signal) -> None:
        ff = FunctionFitter()
        result = ff.fit_linear(linear_signal)
        assert result.r_squared > 0.99

    def test_fit_exponential_decay(self, decay_signal: Signal) -> None:
        ff = FunctionFitter()
        result = ff.fit_exponential_decay(decay_signal)
        assert isinstance(result, FitResult)

    def test_fit_exponential_growth(self, t200: np.ndarray) -> None:
        vals = 3.0 * (1 - np.exp(-2.0 * (t200 - t200[0])))
        sig = Signal(time=t200, values=vals, name="growth")
        ff = FunctionFitter()
        result = ff.fit_exponential_growth(sig)
        assert isinstance(result, FitResult)

    def test_fit_polynomial(self, t200: np.ndarray) -> None:
        ff = FunctionFitter()
        sig = Signal(time=t200, values=(t200 - t200[0]) ** 2)
        result = ff.fit_polynomial(sig, order=2)
        assert result.r_squared > 0.99
