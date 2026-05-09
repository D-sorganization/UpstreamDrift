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


class TestCommonSeriesFunctional:
    """Functional tests for pre-defined common series."""

    def test_exp_series_accuracy(self) -> None:
        """Test exponential series accuracy."""
        from signal_toolkit.series import exp_series

        exp_func = exp_series(n_terms=15)
        x_test = np.linspace(-2, 2, 50)

        expected = np.exp(x_test)
        actual = exp_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-5)

    def test_sin_series_accuracy(self) -> None:
        """Test sine series accuracy."""
        from signal_toolkit.series import sin_series

        sin_func = sin_series(n_terms=15)
        x_test = np.linspace(-np.pi, np.pi, 50)

        expected = np.sin(x_test)
        actual = sin_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-4)

    def test_cos_series_accuracy(self) -> None:
        """Test cosine series accuracy."""
        from signal_toolkit.series import cos_series

        cos_func = cos_series(n_terms=15)
        x_test = np.linspace(-np.pi, np.pi, 50)

        expected = np.cos(x_test)
        actual = cos_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-4)

    def test_ln_series_convergence_region(self) -> None:
        """Test ln(1+x) series in convergence region |x| < 1."""
        from signal_toolkit.series import ln_series

        ln_func = ln_series(n_terms=50)
        # Stay away from the boundary at x=-1 where convergence is slow
        x_test = np.linspace(-0.8, 0.9, 50)

        expected = np.log(1 + x_test)
        actual = ln_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-3)

    def test_geometric_series_convergence_region(self) -> None:
        """Test geometric series 1/(1-x) in convergence region |x| < 1."""
        from signal_toolkit.series import geometric_series

        geo_func = geometric_series(n_terms=50)
        # Stay away from the boundaries where convergence is slow
        x_test = np.linspace(-0.8, 0.8, 50)

        expected = 1 / (1 - x_test)
        actual = geo_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-2)

    def test_arctan_series_accuracy(self) -> None:
        """Test arctan series accuracy in convergence region."""
        from signal_toolkit.series import arctan_series

        arctan_func = arctan_series(n_terms=30)
        x_test = np.linspace(-0.9, 0.9, 50)

        expected = np.arctan(x_test)
        actual = arctan_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-3)

    def test_sinh_series_accuracy(self) -> None:
        """Test sinh series accuracy."""
        from signal_toolkit.series import sinh_series

        sinh_func = sinh_series(n_terms=15)
        x_test = np.linspace(-2, 2, 50)

        expected = np.sinh(x_test)
        actual = sinh_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-5)

    def test_cosh_series_accuracy(self) -> None:
        """Test cosh series accuracy."""
        from signal_toolkit.series import cosh_series

        cosh_func = cosh_series(n_terms=15)
        x_test = np.linspace(-2, 2, 50)

        expected = np.cosh(x_test)
        actual = cosh_func(x_test)

        assert np.allclose(actual, expected, rtol=1e-5)


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
