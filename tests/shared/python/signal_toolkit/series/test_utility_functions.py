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


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_factorial_computation(self) -> None:
        """Test that internal factorial helper is correct."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        # Test factorial helper directly
        assert expansion._factorial(0) == 1
        assert expansion._factorial(1) == 1
        assert expansion._factorial(5) == 120
        assert expansion._factorial(10) == 3628800

    def test_numerical_derivative_accuracy(self) -> None:
        """Test numerical derivative used in coefficient computation."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        # For polynomial f(x) = x^3, f'(0) = 0, f''(0) = 0, f'''(0) = 6
        def f(x) -> float | np.ndarray:
            return x**3

        coeffs = expansion.get_coefficients(f, center=0, n_terms=5)

        # Coefficients: c0=0, c1=0, c2=0, c3=1 (since 6/3! = 1), c4=0
        expected = [0, 0, 0, 1, 0]
        assert np.allclose(coeffs, expected, atol=1e-4)
