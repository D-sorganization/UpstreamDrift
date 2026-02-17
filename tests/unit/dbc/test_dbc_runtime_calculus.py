"""DbC runtime contract tests for signal_toolkit.calculus module.

Verifies require() and ensure() contracts on Differentiator, Integrator,
and convenience functions.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegrationMethod,
    Integrator,
    compute_arc_length,
    compute_derivative,
    compute_integral,
)
from src.shared.python.signal_toolkit.core import Signal


def _make_signal(
    freq: float = 1.0,
    duration: float = 1.0,
    fs: float = 1000.0,
) -> Signal:
    """Create a simple sinusoidal test signal."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    values = np.sin(2 * np.pi * freq * t)
    return Signal(time=t, values=values, name="test_sin", units="m")


# ── Differentiator contracts ─────────────────────────────────────


class TestDifferentiatorPreconditions(unittest.TestCase):
    """Test require() contracts on Differentiator."""

    def test_order_must_be_at_least_one(self) -> None:
        """order < 1 must trip the precondition."""
        sig = _make_signal()
        diff = Differentiator()
        with self.assertRaises(PreconditionError):
            diff.differentiate(sig, order=0)

    def test_negative_order_rejected(self) -> None:
        sig = _make_signal()
        diff = Differentiator()
        with self.assertRaises(PreconditionError):
            diff.differentiate(sig, order=-1)


class TestDifferentiatorPostconditions(unittest.TestCase):
    """Test ensure() contracts on Differentiator."""

    def test_output_length_matches_input(self) -> None:
        """Derivative output must have same number of samples."""
        sig = _make_signal()
        diff = Differentiator()
        result = diff.differentiate(sig, order=1)
        self.assertEqual(len(result.values), len(sig.values))

    def test_second_derivative_output_length(self) -> None:
        sig = _make_signal()
        diff = Differentiator(method=DifferentiationMethod.CENTRAL)
        result = diff.differentiate(sig, order=2)
        self.assertEqual(len(result.values), len(sig.values))

    def test_gradient_method_output_length(self) -> None:
        sig = _make_signal()
        diff = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = diff.differentiate(sig, order=1)
        self.assertEqual(len(result.values), len(sig.values))

    def test_forward_method_output_length(self) -> None:
        sig = _make_signal()
        diff = Differentiator(method=DifferentiationMethod.FORWARD)
        result = diff.differentiate(sig, order=1)
        self.assertEqual(len(result.values), len(sig.values))


# ── Integrator contracts ──────────────────────────────────────────


class TestIntegratorPostconditions(unittest.TestCase):
    """Test ensure() contracts on Integrator.integrate."""

    def test_area_positive_non_negative(self) -> None:
        """area_positive must be >= 0 for a purely positive signal."""
        t = np.linspace(0, 1, 500)
        values = np.abs(np.sin(2 * np.pi * t))  # all positive
        sig = Signal(time=t, values=values, name="abs_sin")
        integrator = Integrator()
        result = integrator.integrate(sig)
        self.assertGreaterEqual(result.area_positive, 0)

    def test_area_negative_non_negative(self) -> None:
        """area_negative must be >= 0 (it's stored as absolute value)."""
        t = np.linspace(0, 1, 500)
        values = -np.abs(np.sin(2 * np.pi * t))  # all negative
        sig = Signal(time=t, values=values, name="neg_sin")
        integrator = Integrator()
        result = integrator.integrate(sig)
        self.assertGreaterEqual(result.area_negative, 0)

    def test_integral_value_finite(self) -> None:
        """Integral value must be finite for well-behaved signals."""
        sig = _make_signal()
        integrator = Integrator()
        result = integrator.integrate(sig)
        self.assertTrue(np.isfinite(result.value))

    def test_simpson_method_postconditions(self) -> None:
        sig = _make_signal()
        integrator = Integrator(method=IntegrationMethod.SIMPSON)
        result = integrator.integrate(sig)
        self.assertGreaterEqual(result.area_positive, 0)
        self.assertGreaterEqual(result.area_negative, 0)
        self.assertTrue(np.isfinite(result.value))

    def test_integral_positive_signal_positive_area(self) -> None:
        """For a purely positive signal, area_positive > 0."""
        t = np.linspace(0, 1, 500)
        sig = Signal(time=t, values=np.ones(500), name="const")
        integrator = Integrator()
        result = integrator.integrate(sig)
        self.assertGreater(result.area_positive, 0)
        self.assertAlmostEqual(result.area_negative, 0, places=10)


# ── Convenience function contracts ────────────────────────────────


class TestComputeDerivative(unittest.TestCase):
    """Test convenience wrapper compute_derivative."""

    def test_valid_first_derivative(self) -> None:
        sig = _make_signal()
        result = compute_derivative(sig, order=1)
        self.assertEqual(len(result.values), len(sig.values))

    def test_order_zero_rejected(self) -> None:
        sig = _make_signal()
        with self.assertRaises(PreconditionError):
            compute_derivative(sig, order=0)


class TestComputeIntegral(unittest.TestCase):
    """Test convenience wrapper compute_integral."""

    def test_finite_integral(self) -> None:
        sig = _make_signal()
        result = compute_integral(sig)
        self.assertTrue(np.isfinite(result.value))


# ── Arc length contracts ──────────────────────────────────────────


class TestArcLength(unittest.TestCase):
    """Test compute_arc_length postcondition."""

    def test_arc_length_non_negative(self) -> None:
        sig = _make_signal()
        length = compute_arc_length(sig)
        self.assertGreaterEqual(length, 0)

    def test_flat_signal_arc_length(self) -> None:
        """A constant signal should have arc length ~= duration."""
        t = np.linspace(0, 1, 1000)
        sig = Signal(time=t, values=5.0 * np.ones(1000), name="const")
        length = compute_arc_length(sig)
        # sqrt(1 + 0^2) integrated from 0 to 1 = 1.0
        self.assertAlmostEqual(length, 1.0, places=2)

    def test_sine_arc_length_greater_than_duration(self) -> None:
        """A sinusoidal signal's arc length should exceed the time span."""
        sig = _make_signal(freq=5.0)
        length = compute_arc_length(sig)
        self.assertGreater(length, sig.time[-1] - sig.time[0])


if __name__ == "__main__":
    unittest.main()
