"""Tests for src.shared.python.signal_toolkit.calculus (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegrationMethod,
    Integrator,
    compute_derivative,
    compute_integral,
)
from src.shared.python.signal_toolkit.core import Signal


def _make_signal(n: int = 100, amp: float = 1.0) -> Signal:
    """Create a sinusoidal test signal: y = amp * sin(2*pi*t), t in [0, 1]."""
    t = np.linspace(0.0, 1.0, n)
    y = amp * np.sin(2 * np.pi * t)
    return Signal(time=t, values=y, name="test_sin", units="m")


def _make_linear_signal(n: int = 100) -> Signal:
    """Create a linear signal y = t."""
    t = np.linspace(0.0, 1.0, n)
    y = t.copy()
    return Signal(time=t, values=y, name="linear", units="m")


class TestDifferentiator:
    def test_gradient_returns_signal(self) -> None:
        sig = _make_signal()
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(sig)
        assert isinstance(result, Signal)

    def test_output_same_length_as_input(self) -> None:
        sig = _make_signal(n=100)
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(sig)
        assert len(result.values) == len(sig.values)

    def test_linear_first_derivative_approx_one(self) -> None:
        sig = _make_linear_signal(n=200)
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(sig)
        # dy/dt of t should be ~1 everywhere except edges
        assert result.values[10:-10] == pytest.approx(np.ones(180), abs=0.01)

    def test_forward_method_runs(self) -> None:
        sig = _make_signal()
        d = Differentiator(method=DifferentiationMethod.FORWARD)
        result = d.differentiate(sig)
        assert len(result.values) == len(sig.values)

    def test_backward_method_runs(self) -> None:
        sig = _make_signal()
        d = Differentiator(method=DifferentiationMethod.BACKWARD)
        result = d.differentiate(sig)
        assert len(result.values) == len(sig.values)

    def test_central_method_runs(self) -> None:
        sig = _make_signal()
        d = Differentiator(method=DifferentiationMethod.CENTRAL)
        result = d.differentiate(sig)
        assert len(result.values) == len(sig.values)

    def test_savgol_method_runs(self) -> None:
        sig = _make_signal(n=50)
        d = Differentiator(method=DifferentiationMethod.SAVGOL)
        result = d.differentiate(sig)
        assert len(result.values) == len(sig.values)

    def test_second_order_derivative(self) -> None:
        sig = _make_signal(n=200)
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(sig, order=2)
        assert result.values is not None

    def test_order_zero_raises(self) -> None:
        sig = _make_signal()
        d = Differentiator()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            d.differentiate(sig, order=0)

    def test_order_negative_raises(self) -> None:
        sig = _make_signal()
        d = Differentiator()
        with pytest.raises((ValueError, TypeError, AssertionError)):
            d.differentiate(sig, order=-1)

    def test_result_name_updated(self) -> None:
        sig = _make_signal()
        d = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = d.differentiate(sig)
        assert "test_sin" in result.name


class TestIntegrator:
    def test_trapezoid_returns_integral_result(self) -> None:
        sig = _make_signal()
        integrator = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integrator.integrate(sig)
        assert result is not None

    def test_integral_value_finite(self) -> None:
        sig = _make_signal()
        integrator = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integrator.integrate(sig)
        assert np.isfinite(result.value)

    def test_integral_of_zeros_is_zero(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        sig = Signal(time=t, values=np.zeros(100))
        integrator = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integrator.integrate(sig)
        assert result.value == pytest.approx(0.0, abs=1e-10)

    def test_integral_of_ones_is_one(self) -> None:
        t = np.linspace(0.0, 1.0, 1000)
        sig = Signal(time=t, values=np.ones(1000))
        integrator = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integrator.integrate(sig)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_simpson_method_runs(self) -> None:
        sig = _make_signal(n=101)  # odd n needed for Simpson
        integrator = Integrator(method=IntegrationMethod.SIMPSON)
        result = integrator.integrate(sig)
        assert np.isfinite(result.value)

    def test_cumulative_method_runs(self) -> None:
        sig = _make_signal()
        integrator = Integrator(method=IntegrationMethod.CUMULATIVE)
        result = integrator.integrate(sig)
        assert result is not None

    def test_bounds_stored(self) -> None:
        sig = _make_signal()
        integrator = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integrator.integrate(sig, lower_bound=0.2, upper_bound=0.8)
        assert result.lower_bound == pytest.approx(0.2)
        assert result.upper_bound == pytest.approx(0.8)


class TestConvenienceFunctions:
    def test_compute_derivative_returns_signal(self) -> None:
        sig = _make_signal()
        result = compute_derivative(sig, method=DifferentiationMethod.GRADIENT)
        assert isinstance(result, Signal)

    def test_compute_integral_returns_result(self) -> None:
        sig = _make_signal()
        result = compute_integral(sig)
        assert result is not None

    def test_compute_integral_finite(self) -> None:
        sig = _make_signal()
        result = compute_integral(sig)
        assert np.isfinite(result.value)
