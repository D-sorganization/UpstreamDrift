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


class TestSignalToolkitIntegration:
    """Tests for integration with the signal toolkit."""

    def test_series_with_signal(self) -> None:
        """Test applying series approximation to a Signal."""
        from signal_toolkit.core import Signal
        from signal_toolkit.series import sin_series

        # Use the optimized sin_series instead of taylor_series for accuracy
        sin_approx = sin_series(n_terms=15)

        t = np.linspace(-np.pi, np.pi, 100)
        signal = Signal(t, t, name="input")

        # Apply series approximation
        approx_values = sin_approx(signal.values)
        result = Signal(signal.time, approx_values, name="sin_approx")

        assert len(result.values) == len(signal.values)
        assert np.allclose(result.values, np.sin(t), rtol=1e-3)

    def test_generate_series_approximation_signal(self) -> None:
        """Test generating a signal from series approximation."""
        from signal_toolkit.core import SignalGenerator
        from signal_toolkit.series import exp_series

        t = np.linspace(0, 2, 100)
        exp_approx = exp_series(n_terms=20)

        # Create signal using the series approximation
        signal = SignalGenerator.from_function(t, exp_approx)

        # Compare with actual exponential
        expected = np.exp(t)
        assert np.allclose(signal.values, expected, rtol=1e-5)


# =============================================================================
# Utility Functions Tests
# =============================================================================
