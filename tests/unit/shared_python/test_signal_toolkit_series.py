"""Unit tests for signal_toolkit/series.py (Taylor/Maclaurin series)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.signal_toolkit.series import (
    SeriesExpansion,
    SeriesResult,
    arctan_series,
    cos_series,
    cosh_series,
    exp_series,
    geometric_series,
    ln_series,
    sin_series,
    sinh_series,
)


@pytest.fixture
def se() -> SeriesExpansion:
    return SeriesExpansion(max_terms=20, h=1e-5)


class TestSeriesExpansion:
    """Tests for SeriesExpansion class."""

    def test_invalid_max_terms_raises(self) -> None:
        with pytest.raises(ValueError):
            SeriesExpansion(max_terms=0)

    def test_taylor_sin_at_zero(self, se: SeriesExpansion) -> None:
        """Taylor series of sin(x) at center=0 should approximate well near 0."""
        taylor_fn = se.taylor_series(np.sin, center=0.0, n_terms=10)
        assert np.isclose(taylor_fn(0.0), 0.0, atol=1e-5)
        assert np.isclose(taylor_fn(np.pi / 4), np.sin(np.pi / 4), atol=0.01)

    def test_taylor_cos_at_zero(self, se: SeriesExpansion) -> None:
        taylor_fn = se.taylor_series(np.cos, center=0.0, n_terms=10)
        assert np.isclose(taylor_fn(0.0), 1.0, atol=1e-5)

    def test_taylor_exp(self, se: SeriesExpansion) -> None:
        taylor_fn = se.taylor_series(np.exp, center=0.0, n_terms=15)
        assert np.isclose(taylor_fn(1.0), math.e, atol=0.01)

    def test_taylor_non_callable_raises(self, se: SeriesExpansion) -> None:
        with pytest.raises(TypeError):
            se.taylor_series(42, center=0.0, n_terms=5)  # type: ignore[arg-type]

    def test_taylor_zero_terms_raises(self, se: SeriesExpansion) -> None:
        with pytest.raises(ValueError):
            se.taylor_series(np.sin, center=0.0, n_terms=0)

    def test_maclaurin_is_taylor_at_zero(self, se: SeriesExpansion) -> None:
        """Maclaurin == Taylor at center=0."""
        mac_fn = se.maclaurin_series(np.sin, n_terms=10)
        taylor_fn = se.taylor_series(np.sin, center=0.0, n_terms=10)
        x_test = np.array([0.0, 0.5, 1.0])
        assert np.allclose(mac_fn(x_test), taylor_fn(x_test), atol=1e-8)

    def test_get_coefficients_shape(self, se: SeriesExpansion) -> None:
        coeffs = se.get_coefficients(np.sin, center=0.0, n_terms=8)
        assert len(coeffs) == 8

    def test_get_series_result(self, se: SeriesExpansion) -> None:
        result = se.get_series_result(np.exp, center=0.0, n_terms=10)
        assert isinstance(result, SeriesResult)
        assert result.n_terms == 10
        assert result.center == 0.0
        assert callable(result.function)

    def test_get_series_result_roc(self, se: SeriesExpansion) -> None:
        """Radius of convergence should be computed (a positive float or None)."""
        result = se.get_series_result(np.exp, center=0.0, n_terms=15)
        if result.radius_of_convergence is not None:
            assert result.radius_of_convergence > 0

    def test_analyze_convergence_sin(self, se: SeriesExpansion) -> None:
        analysis = se.analyze_convergence(np.sin, center=0.0, x_test=1.0, tolerance=1e-6)
        assert "convergent" in analysis
        assert "errors_by_term" in analysis
        assert len(analysis["errors_by_term"]) > 0

    def test_analyze_convergence_diverges(self, se: SeriesExpansion) -> None:
        """1/x has a singularity at 0 — non-convergent."""

        def bad_func(x: float) -> float:
            if abs(x) < 1e-10:
                raise ValueError("Singularity")
            return 1.0 / x

        analysis = se.analyze_convergence(bad_func, center=0.0, x_test=0.0)
        # Should gracefully handle and return non-convergent result
        assert not analysis["convergent"]

    def test_estimate_error_bound(self, se: SeriesExpansion) -> None:
        bound = se.estimate_error_bound(np.sin, center=0.0, x_test=0.5, n_terms=5)
        # Error bound should be a finite positive number
        assert np.isfinite(bound) and bound >= 0

    def test_estimate_error_zero_nterms(self, se: SeriesExpansion) -> None:
        bound = se.estimate_error_bound(np.sin, center=0.0, x_test=1.0, n_terms=0)
        assert bound == float("inf")

    def test_taylor_with_array_input(self, se: SeriesExpansion) -> None:
        taylor_fn = se.taylor_series(np.cos, center=0.0, n_terms=10)
        x_arr = np.linspace(-1, 1, 50)
        result = taylor_fn(x_arr)
        assert result.shape == x_arr.shape

    def test_taylor_scalar_returns_float(self, se: SeriesExpansion) -> None:
        taylor_fn = se.taylor_series(np.sin, center=0.0, n_terms=5)
        result = taylor_fn(0.5)
        assert isinstance(result, float)

    def test_factorial_zero(self) -> None:
        assert SeriesExpansion._factorial(0) == 1

    def test_factorial_positive(self) -> None:
        assert SeriesExpansion._factorial(5) == 120

    def test_factorial_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            SeriesExpansion._factorial(-1)

    def test_binomial_basic(self) -> None:
        assert SeriesExpansion._binomial(5, 2) == 10

    def test_binomial_zero(self) -> None:
        assert SeriesExpansion._binomial(5, 0) == 1

    def test_binomial_out_of_range(self) -> None:
        assert SeriesExpansion._binomial(5, 6) == 0


class TestSeriesFunctions:
    """Tests for pre-defined series factory functions."""

    def test_exp_series_at_zero(self) -> None:
        fn = exp_series(n_terms=20)
        assert np.isclose(fn(0.0), 1.0, atol=1e-10)

    def test_exp_series_at_one(self) -> None:
        fn = exp_series(n_terms=20)
        assert np.isclose(fn(1.0), math.e, atol=0.01)

    def test_exp_series_array(self) -> None:
        fn = exp_series(n_terms=15)
        x = np.array([0.0, 0.5, 1.0])
        result = fn(x)
        assert np.allclose(result, np.exp(x), atol=0.01)

    def test_sin_series_at_zero(self) -> None:
        fn = sin_series(n_terms=10)
        assert np.isclose(fn(0.0), 0.0, atol=1e-10)

    def test_sin_series_at_pi_half(self) -> None:
        fn = sin_series(n_terms=15)
        assert np.isclose(fn(np.pi / 2), 1.0, atol=0.01)

    def test_sin_series_array(self) -> None:
        fn = sin_series(n_terms=15)
        x = np.linspace(-np.pi, np.pi, 50)
        assert np.allclose(fn(x), np.sin(x), atol=0.01)

    def test_cos_series_at_zero(self) -> None:
        fn = cos_series(n_terms=10)
        assert np.isclose(fn(0.0), 1.0, atol=1e-10)

    def test_cos_series_at_pi(self) -> None:
        fn = cos_series(n_terms=15)
        assert np.isclose(fn(np.pi), -1.0, atol=0.05)

    def test_cos_series_array(self) -> None:
        fn = cos_series(n_terms=15)
        x = np.linspace(-np.pi, np.pi, 50)
        assert np.allclose(fn(x), np.cos(x), atol=0.05)

    def test_ln_series_near_zero(self) -> None:
        """ln(1 + x) series for |x| < 1."""
        fn = ln_series(n_terms=50)
        # ln(1 + 0) = 0
        assert np.isclose(fn(0.0), 0.0, atol=1e-8)
        # ln(1 + 0.5) ≈ 0.405
        assert np.isclose(fn(0.5), np.log(1.5), atol=0.01)

    def test_geometric_series_near_zero(self) -> None:
        """1/(1-x) series for |x| < 1."""
        fn = geometric_series(n_terms=50)
        # At x=0: 1/(1-0) = 1
        assert np.isclose(fn(0.0), 1.0, atol=1e-8)
        # At x=0.5: 1/(1-0.5) = 2
        assert np.isclose(fn(0.5), 2.0, atol=0.01)

    def test_arctan_series_at_zero(self) -> None:
        fn = arctan_series(n_terms=50)
        assert np.isclose(fn(0.0), 0.0, atol=1e-10)

    def test_arctan_series_at_one(self) -> None:
        """arctan(1) = pi/4."""
        fn = arctan_series(n_terms=100)
        assert np.isclose(fn(1.0), np.pi / 4, atol=0.05)

    def test_sinh_series_at_zero(self) -> None:
        fn = sinh_series(n_terms=10)
        assert np.isclose(fn(0.0), 0.0, atol=1e-10)

    def test_sinh_series_at_one(self) -> None:
        fn = sinh_series(n_terms=15)
        assert np.isclose(fn(1.0), np.sinh(1.0), atol=0.01)

    def test_sinh_series_array(self) -> None:
        fn = sinh_series(n_terms=10)
        x = np.array([0.0, 0.5, 1.0])
        assert np.allclose(fn(x), np.sinh(x), atol=0.01)

    def test_cosh_series_at_zero(self) -> None:
        fn = cosh_series(n_terms=10)
        assert np.isclose(fn(0.0), 1.0, atol=1e-10)

    def test_cosh_series_at_one(self) -> None:
        fn = cosh_series(n_terms=10)
        assert np.isclose(fn(1.0), np.cosh(1.0), atol=0.01)

    def test_cosh_series_array(self) -> None:
        fn = cosh_series(n_terms=10)
        x = np.array([0.0, 0.5, 1.0])
        assert np.allclose(fn(x), np.cosh(x), atol=0.01)

    def test_exp_series_scalar_returns_float(self) -> None:
        fn = exp_series(n_terms=10)
        result = fn(1.0)
        assert isinstance(result, float)
