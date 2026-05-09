"""Tests for the Taylor and Maclaurin series module.

This module contains comprehensive tests for series expansion functionality:
- Taylor series computation at any point
- Maclaurin series (Taylor at x=0)
- Common function series (sin, cos, exp, ln, etc.)
- Convergence analysis
- Error bounds

Following TDD and Design by Contract principles.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# =============================================================================
# SeriesExpansion Class Contract Tests
# =============================================================================


# =============================================================================
# Taylor Series Method Contract Tests
# =============================================================================


# =============================================================================
# Maclaurin Series Contract Tests
# =============================================================================


# =============================================================================
# Get Coefficients Contract Tests
# =============================================================================


# =============================================================================
# Common Series Functions Contract Tests
# =============================================================================


# =============================================================================
# Taylor Series Functional Tests
# =============================================================================


class TestTaylorSeriesFunctional:
    """Functional tests for Taylor series computation."""

    def test_polynomial_exact(self) -> None:
        """Test that Taylor series of a polynomial is exact."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        # f(x) = 1 + 2x + 3x^2
        def f(x) -> float | np.ndarray:
            return 1 + 2 * x + 3 * x**2

        taylor_func = expansion.taylor_series(f, center=0, n_terms=5)

        x_test = np.linspace(-2, 2, 50)
        assert np.allclose(taylor_func(x_test), f(x_test), rtol=1e-6)

    def test_exponential_convergence(self) -> None:
        """Test exponential Taylor series convergence."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        taylor_func = expansion.taylor_series(np.exp, center=0, n_terms=15)

        # Near center, should be very accurate
        x_test = np.linspace(-1, 1, 20)
        expected = np.exp(x_test)
        actual = taylor_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-6)

    def test_sine_convergence(self) -> None:
        """Test sine Taylor series convergence."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        taylor_func = expansion.taylor_series(np.sin, center=0, n_terms=15)

        x_test = np.linspace(-np.pi / 2, np.pi / 2, 20)
        expected = np.sin(x_test)
        actual = taylor_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-5)

    def test_cosine_convergence(self) -> None:
        """Test cosine Taylor series convergence."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        taylor_func = expansion.taylor_series(np.cos, center=0, n_terms=15)

        x_test = np.linspace(-np.pi / 2, np.pi / 2, 20)
        expected = np.cos(x_test)
        actual = taylor_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-5)

    def test_taylor_at_nonzero_center(self) -> None:
        """Test Taylor series expansion at non-zero center."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        # Expand exp(x) around x=1
        taylor_func = expansion.taylor_series(np.exp, center=1, n_terms=15)

        # Should be accurate near x=1
        x_test = np.linspace(0.5, 1.5, 20)
        expected = np.exp(x_test)
        actual = taylor_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-5)

    def test_more_terms_improves_accuracy(self) -> None:
        """Test that more terms generally improves accuracy."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        f = np.sin
        x_test = 1.5  # Away from center

        taylor_5 = expansion.taylor_series(f, center=0, n_terms=5)
        taylor_15 = expansion.taylor_series(f, center=0, n_terms=15)

        error_5 = abs(taylor_5(x_test) - f(x_test))
        error_15 = abs(taylor_15(x_test) - f(x_test))

        assert error_15 < error_5


# =============================================================================
# Common Series Functions Tests
# =============================================================================


# =============================================================================
# Coefficients Tests
# =============================================================================


# =============================================================================
# Convergence Analysis Tests
# =============================================================================


# =============================================================================
# Error Bounds Tests
# =============================================================================


# =============================================================================
# SeriesResult Dataclass Tests
# =============================================================================


# =============================================================================
# Integration with Signal Toolkit Tests
# =============================================================================


# =============================================================================
# Utility Functions Tests
# =============================================================================
