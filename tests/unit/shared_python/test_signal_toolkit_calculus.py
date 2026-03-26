"""Unit tests for signal_toolkit/calculus.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegrationMethod,
    Integrator,
    TangentLine,
    compute_all_tangent_lines,
    compute_arc_length,
    compute_curvature,
    compute_derivative,
    compute_integral,
    compute_tangent_line,
    find_extrema,
    find_inflection_points,
)
from src.shared.python.signal_toolkit.core import Signal, SignalGenerator


@pytest.fixture
def time_array() -> np.ndarray:
    return np.linspace(0, 2 * np.pi, 500)


@pytest.fixture
def sine_signal(time_array: np.ndarray) -> Signal:
    return SignalGenerator.sinusoid(time_array, amplitude=1.0, frequency=1.0, name="sin")


@pytest.fixture
def linear_signal() -> Signal:
    t = np.linspace(0, 5, 200)
    return SignalGenerator.linear(t, slope=2.0, intercept=0.5, name="linear")


@pytest.fixture
def quadratic_signal() -> Signal:
    t = np.linspace(0, 3, 300)
    # y = t^2  => dy/dt = 2t at t=0..3
    return Signal(time=t, values=t**2, name="quad")


class TestDifferentiator:
    """Tests for the Differentiator class."""

    def test_forward_difference(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.FORWARD)
        result = d.differentiate(linear_signal)
        # Linear signal slope=2 → derivative≈2
        assert np.isclose(np.mean(result.values[:-1]), 2.0, atol=0.05)

    def test_backward_difference(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.BACKWARD)
        result = d.differentiate(linear_signal)
        assert np.isclose(np.mean(result.values[1:]), 2.0, atol=0.05)

    def test_central_difference(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.CENTRAL)
        result = d.differentiate(linear_signal)
        assert np.isclose(np.mean(result.values[1:-1]), 2.0, atol=0.02)

    def test_gradient_method(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(linear_signal)
        assert np.isclose(np.mean(result.values), 2.0, atol=0.02)

    def test_savgol_method(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.SAVGOL)
        result = d.differentiate(linear_signal)
        assert np.isclose(np.mean(result.values), 2.0, atol=0.1)

    def test_spline_method(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.SPLINE)
        result = d.differentiate(linear_signal)
        assert np.isclose(np.mean(result.values), 2.0, atol=0.1)

    def test_second_order_derivative(self, quadratic_signal: Signal) -> None:
        """Second derivative of t^2 ≈ 2."""
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(quadratic_signal, order=2)
        assert np.isclose(np.mean(result.values[5:-5]), 2.0, atol=0.2)

    def test_invalid_order_raises(self, linear_signal: Signal) -> None:
        d = Differentiator()
        with pytest.raises((ValueError, AssertionError)):
            d.differentiate(linear_signal, order=0)

    def test_name_and_units_updated(self, linear_signal: Signal) -> None:
        linear_signal.units = "m"
        d = Differentiator()
        result = d.differentiate(linear_signal, order=1)
        assert "m/s" in result.units
        assert "d1(" in result.name

    def test_compute_at_point(self, linear_signal: Signal) -> None:
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        val = d.compute_at_point(linear_signal, t_point=2.5)
        assert np.isclose(val, 2.0, atol=0.1)

    def test_savgol_short_signal(self) -> None:
        """Savgol falls back gracefully for short signals."""
        t = np.linspace(0, 1, 5)
        sig = Signal(time=t, values=t * 3.0, name="short")
        d = Differentiator(method=DifferentiationMethod.SAVGOL)
        result = d.differentiate(sig)
        assert result.values.shape == t.shape


class TestIntegrator:
    """Tests for the Integrator class."""

    def test_trapezoid_constant(self) -> None:
        """Integral of constant=1 over [0,1] = 1."""
        t = np.linspace(0, 1, 1000)
        sig = Signal(time=t, values=np.ones_like(t), name="const")
        integ = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integ.integrate(sig)
        assert np.isclose(result.value, 1.0, atol=1e-3)

    def test_simpson(self) -> None:
        t = np.linspace(0, 1, 101)
        sig = Signal(time=t, values=np.ones_like(t), name="const")
        integ = Integrator(method=IntegrationMethod.SIMPSON)
        result = integ.integrate(sig)
        assert np.isclose(result.value, 1.0, atol=1e-4)

    def test_cumulative(self) -> None:
        t = np.linspace(0, 1, 1000)
        sig = Signal(time=t, values=np.ones_like(t), name="const")
        integ = Integrator(method=IntegrationMethod.CUMULATIVE)
        result = integ.integrate(sig)
        assert np.isclose(result.value, 1.0, atol=1e-3)

    def test_bounds(self) -> None:
        t = np.linspace(0, 2, 1000)
        sig = Signal(time=t, values=np.ones_like(t), name="const")
        integ = Integrator()
        result = integ.integrate(sig, lower_bound=0.5, upper_bound=1.5)
        assert np.isclose(result.value, 1.0, atol=0.01)

    def test_cumulative_integral(self) -> None:
        t = np.linspace(0, 1, 1000)
        sig = Signal(time=t, values=2 * np.ones_like(t), name="two")
        integ = Integrator()
        cum_sig = integ.cumulative_integral(sig, initial_value=0.0)
        # Integral of 2 from 0 to t is 2t
        assert np.isclose(cum_sig.values[-1], 2.0, atol=0.01)

    def test_positive_negative_area(self) -> None:
        t = np.linspace(0, 2 * np.pi, 1000)
        sig = Signal(time=t, values=np.sin(t), name="sin")
        integ = Integrator()
        result = integ.integrate(sig)
        assert result.area_positive > 0
        assert result.area_negative >= 0


class TestStandaloneFunctions:
    """Tests for module-level calculus functions."""

    def test_compute_derivative(self, linear_signal: Signal) -> None:
        result = compute_derivative(linear_signal, order=1, method=DifferentiationMethod.GRADIENT)
        assert np.isclose(np.mean(result.values), 2.0, atol=0.05)

    def test_compute_integral(self) -> None:
        t = np.linspace(0, 1, 500)
        sig = Signal(time=t, values=np.ones_like(t), name="one")
        result = compute_integral(sig)
        assert np.isclose(result.value, 1.0, atol=1e-3)

    def test_compute_tangent_line(self, sine_signal: Signal) -> None:
        tangent = compute_tangent_line(sine_signal, t_point=np.pi / 2)
        assert isinstance(tangent, TangentLine)
        assert len(tangent.t_range) == 100
        eq = tangent.get_equation_string()
        assert "y =" in eq

    def test_compute_tangent_line_with_width(self, sine_signal: Signal) -> None:
        tangent = compute_tangent_line(sine_signal, t_point=np.pi / 2, line_width=0.5)
        assert tangent.t_range[-1] - tangent.t_range[0] <= 0.5 + 1e-6

    def test_compute_all_tangent_lines(self, sine_signal: Signal) -> None:
        tangents = compute_all_tangent_lines(sine_signal, num_points=5)
        assert len(tangents) == 5

    def test_compute_curvature(self) -> None:
        t = np.linspace(0, 5, 300)
        # Circle = constant curvature => roughly constant
        sig = Signal(time=t, values=np.sin(t), name="sin")
        curv = compute_curvature(sig)
        assert curv.values.shape == t.shape
        assert np.all(np.isfinite(curv.values))

    def test_compute_arc_length_linear(self) -> None:
        t = np.linspace(0, 1, 500)
        sig = Signal(time=t, values=t, name="identity")
        # Arc length of y=x from 0 to 1 = sqrt(2)
        length = compute_arc_length(sig)
        assert np.isclose(length, np.sqrt(2), atol=0.01)

    def test_find_extrema_sine(self, sine_signal: Signal) -> None:
        maxima, minima = find_extrema(sine_signal)
        assert len(maxima) >= 1
        assert len(minima) >= 1

    def test_find_inflection_points_sine(self, sine_signal: Signal) -> None:
        inflections = find_inflection_points(sine_signal)
        # sin(t) has inflection at t=0, pi, 2*pi — but sine_signal starts at 0
        assert len(inflections) >= 1
