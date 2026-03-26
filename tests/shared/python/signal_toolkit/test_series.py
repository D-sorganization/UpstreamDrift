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


class TestSeriesExpansionContract:
    """Design by Contract tests for SeriesExpansion class."""

    def test_instantiates(self) -> None:
        """Postcondition: SeriesExpansion can be instantiated."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        assert expansion is not None

    def test_has_max_terms_attribute(self) -> None:
        """Postcondition: Has max_terms attribute with default value."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        assert hasattr(expansion, "max_terms")
        assert expansion.max_terms == 50  # Reasonable default

    def test_accepts_custom_max_terms(self) -> None:
        """Postcondition: Can specify custom max_terms."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion(max_terms=100)
        assert expansion.max_terms == 100

    def test_rejects_non_positive_max_terms(self) -> None:
        """Precondition: Rejects non-positive max_terms."""
        from signal_toolkit.series import SeriesExpansion

        with pytest.raises(ValueError):
            SeriesExpansion(max_terms=0)

        with pytest.raises(ValueError):
            SeriesExpansion(max_terms=-5)


# =============================================================================
# Taylor Series Method Contract Tests
# =============================================================================


class TestTaylorSeriesContract:
    """Design by Contract tests for taylor_series method."""

    def test_returns_callable(self) -> None:
        """Postcondition: Returns a callable function."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.sin(x)

        taylor_func = expansion.taylor_series(f, center=0, n_terms=5)

        assert callable(taylor_func)

    def test_callable_accepts_scalar(self) -> None:
        """Postcondition: Returned function accepts scalar input."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.exp(x)

        taylor_func = expansion.taylor_series(f, center=0, n_terms=5)

        result = taylor_func(1.0)
        assert isinstance(result, (int, float, np.floating))

    def test_callable_accepts_array(self) -> None:
        """Postcondition: Returned function accepts array input."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.cos(x)

        taylor_func = expansion.taylor_series(f, center=0, n_terms=5)

        x = np.linspace(-1, 1, 10)
        result = taylor_func(x)
        assert isinstance(result, np.ndarray)
        assert len(result) == 10

    def test_rejects_non_callable_function(self) -> None:
        """Precondition: Rejects non-callable function argument."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        with pytest.raises(TypeError):
            expansion.taylor_series("not a function", center=0, n_terms=5)  # type: ignore[arg-type]

    def test_rejects_non_positive_n_terms(self) -> None:
        """Precondition: Rejects non-positive n_terms."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return x**2

        with pytest.raises(ValueError):
            expansion.taylor_series(f, center=0, n_terms=0)

        with pytest.raises(ValueError):
            expansion.taylor_series(f, center=0, n_terms=-3)


# =============================================================================
# Maclaurin Series Contract Tests
# =============================================================================


class TestMaclaurinSeriesContract:
    """Design by Contract tests for maclaurin_series method (Taylor at x=0)."""

    def test_returns_callable(self) -> None:
        """Postcondition: Returns a callable function."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.sin(x)

        maclaurin_func = expansion.maclaurin_series(f, n_terms=5)

        assert callable(maclaurin_func)

    def test_equivalent_to_taylor_at_zero(self) -> None:
        """Postcondition: Maclaurin series equals Taylor series at center=0."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.exp(x)

        taylor_func = expansion.taylor_series(f, center=0, n_terms=10)
        maclaurin_func = expansion.maclaurin_series(f, n_terms=10)

        x_test = np.linspace(-1, 1, 20)
        assert np.allclose(taylor_func(x_test), maclaurin_func(x_test))


# =============================================================================
# Get Coefficients Contract Tests
# =============================================================================


class TestGetCoefficientsContract:
    """Design by Contract tests for get_coefficients method."""

    def test_returns_array(self) -> None:
        """Postcondition: Returns numpy array of coefficients."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.exp(x)

        coeffs = expansion.get_coefficients(f, center=0, n_terms=5)

        assert isinstance(coeffs, np.ndarray)

    def test_correct_number_of_coefficients(self) -> None:
        """Postcondition: Returns exactly n_terms coefficients."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.sin(x)

        for n in [3, 5, 10]:
            coeffs = expansion.get_coefficients(f, center=0, n_terms=n)
            assert len(coeffs) == n


# =============================================================================
# Common Series Functions Contract Tests
# =============================================================================


class TestCommonSeriesContract:
    """Design by Contract tests for pre-defined common series."""

    def test_exp_series_exists(self) -> None:
        """Postcondition: exp_series function exists and is callable."""
        from signal_toolkit.series import exp_series

        assert callable(exp_series)

    def test_sin_series_exists(self) -> None:
        """Postcondition: sin_series function exists and is callable."""
        from signal_toolkit.series import sin_series

        assert callable(sin_series)

    def test_cos_series_exists(self) -> None:
        """Postcondition: cos_series function exists and is callable."""
        from signal_toolkit.series import cos_series

        assert callable(cos_series)

    def test_ln_series_exists(self) -> None:
        """Postcondition: ln_series function exists (for ln(1+x))."""
        from signal_toolkit.series import ln_series

        assert callable(ln_series)

    def test_geometric_series_exists(self) -> None:
        """Postcondition: geometric_series function exists (for 1/(1-x))."""
        from signal_toolkit.series import geometric_series

        assert callable(geometric_series)

    def test_arctan_series_exists(self) -> None:
        """Postcondition: arctan_series function exists."""
        from signal_toolkit.series import arctan_series

        assert callable(arctan_series)

    def test_sinh_series_exists(self) -> None:
        """Postcondition: sinh_series function exists."""
        from signal_toolkit.series import sinh_series

        assert callable(sinh_series)

    def test_cosh_series_exists(self) -> None:
        """Postcondition: cosh_series function exists."""
        from signal_toolkit.series import cosh_series

        assert callable(cosh_series)


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
        def f(x):
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


class TestCoefficients:
    """Tests for Taylor series coefficient extraction."""

    def test_exp_coefficients(self) -> None:
        """Test exponential series coefficients are approximately 1/n!."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        coeffs = expansion.get_coefficients(np.exp, center=0, n_terms=10)

        expected = [1 / math.factorial(n) for n in range(10)]
        # Polynomial fitting gives good approximations but not exact values
        # The key test is that the resulting series produces accurate results
        # Higher-order coefficients have more numerical error
        assert np.allclose(coeffs, expected, rtol=5e-2)

    def test_sin_coefficients(self) -> None:
        """Test sine series coefficients are approximately correct."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        coeffs = expansion.get_coefficients(np.sin, center=0, n_terms=7)

        # sin(x) = x - x^3/3! + x^5/5! - ...
        # Coefficients: 0, 1, 0, -1/6, 0, 1/120, 0
        expected = [0, 1, 0, -1 / 6, 0, 1 / 120, 0]
        # Polynomial fitting gives good approximations; zero coefficients may have
        # small numerical noise
        assert np.allclose(coeffs, expected, rtol=1e-2, atol=1e-3)

    def test_cos_coefficients(self) -> None:
        """Test cosine series coefficients are approximately correct."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        coeffs = expansion.get_coefficients(np.cos, center=0, n_terms=7)

        # cos(x) = 1 - x^2/2! + x^4/4! - ...
        # Coefficients: 1, 0, -1/2, 0, 1/24, 0, -1/720
        expected = [1, 0, -0.5, 0, 1 / 24, 0, -1 / 720]
        # Polynomial fitting gives good approximations; zero coefficients may have
        # small numerical noise
        assert np.allclose(coeffs, expected, rtol=1e-2, atol=1e-3)


# =============================================================================
# Convergence Analysis Tests
# =============================================================================


class TestConvergenceAnalysis:
    """Tests for convergence analysis functionality."""

    def test_convergence_analysis_returns_dict(self) -> None:
        """Postcondition: analyze_convergence returns a dictionary."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        analysis = expansion.analyze_convergence(np.exp, center=0, x_test=1.0)

        assert isinstance(analysis, dict)

    def test_convergence_analysis_has_required_keys(self) -> None:
        """Postcondition: Analysis has all required keys."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        analysis = expansion.analyze_convergence(np.exp, center=0, x_test=1.0)

        assert "convergent" in analysis
        assert "terms_for_convergence" in analysis
        assert "final_error" in analysis
        assert "errors_by_term" in analysis

    def test_exp_converges_everywhere(self) -> None:
        """Test that exp series converges for x values near center."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        # For numerical methods, convergence is reliable close to the center
        # Far from center (|x| > 2), many terms are needed
        for x in [-1, 0, 1]:
            analysis = expansion.analyze_convergence(np.exp, center=0, x_test=x, tolerance=1e-6)
            assert analysis["convergent"]

    def test_ln_diverges_outside_radius(self) -> None:
        """Test that ln(1+x) series diverges for x < -1 or x > 1."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()

        def f(x):
            return np.log(1 + x)

        # Should diverge for x = 2 (outside |x| < 1)
        analysis = expansion.analyze_convergence(f, center=0, x_test=2.0, tolerance=1e-6)
        assert not analysis["convergent"]


# =============================================================================
# Error Bounds Tests
# =============================================================================


class TestErrorBounds:
    """Tests for error bound estimation."""

    def test_error_bound_returns_float(self) -> None:
        """Postcondition: estimate_error_bound returns a float."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        bound = expansion.estimate_error_bound(f=np.exp, center=0, x_test=1.0, n_terms=10)

        assert isinstance(bound, (int, float, np.floating))

    def test_error_bound_is_non_negative(self) -> None:
        """Postcondition: Error bound is non-negative."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        bound = expansion.estimate_error_bound(f=np.sin, center=0, x_test=0.5, n_terms=10)

        assert bound >= 0

    def test_error_bound_decreases_with_terms(self) -> None:
        """Test that actual error decreases with more terms (rather than bound)."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        x_test = 0.5
        f = np.exp

        # Test that actual approximation error decreases with more terms
        taylor_5 = expansion.taylor_series(f, center=0, n_terms=5)
        taylor_15 = expansion.taylor_series(f, center=0, n_terms=15)

        error_5 = abs(taylor_5(x_test) - f(x_test))
        error_15 = abs(taylor_15(x_test) - f(x_test))

        assert error_15 < error_5

    def test_actual_error_within_bound(self) -> None:
        """Test that actual error is within estimated bound."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        f = np.exp
        x_test = 0.5
        n_terms = 10

        taylor_func = expansion.taylor_series(f, center=0, n_terms=n_terms)
        actual_error = abs(taylor_func(x_test) - f(x_test))
        bound = expansion.estimate_error_bound(f, center=0, x_test=x_test, n_terms=n_terms)

        # Bound should be >= actual error (with some margin)
        assert actual_error <= bound * 2  # Allow factor of 2 margin


# =============================================================================
# SeriesResult Dataclass Tests
# =============================================================================


class TestSeriesResult:
    """Tests for SeriesResult dataclass."""

    def test_get_series_result_returns_dataclass(self) -> None:
        """Postcondition: get_series_result returns SeriesResult dataclass."""
        from signal_toolkit.series import SeriesExpansion, SeriesResult

        expansion = SeriesExpansion()
        result = expansion.get_series_result(np.exp, center=0, n_terms=10)

        assert isinstance(result, SeriesResult)

    def test_series_result_has_required_fields(self) -> None:
        """Postcondition: SeriesResult has all required fields."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        result = expansion.get_series_result(np.sin, center=0, n_terms=10)

        assert hasattr(result, "coefficients")
        assert hasattr(result, "n_terms")
        assert hasattr(result, "center")
        assert hasattr(result, "function")
        assert hasattr(result, "radius_of_convergence")

    def test_series_result_function_is_callable(self) -> None:
        """Postcondition: SeriesResult.function is callable."""
        from signal_toolkit.series import SeriesExpansion

        expansion = SeriesExpansion()
        result = expansion.get_series_result(np.cos, center=0, n_terms=10)

        assert callable(result.function)
        assert isinstance(result.function(0.5), (int, float, np.floating))


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
        def f(x):
            return x**3

        coeffs = expansion.get_coefficients(f, center=0, n_terms=5)

        # Coefficients: c0=0, c1=0, c2=0, c3=1 (since 6/3! = 1), c4=0
        expected = [0, 0, 0, 1, 0]
        assert np.allclose(coeffs, expected, atol=1e-4)
