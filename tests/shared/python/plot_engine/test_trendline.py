"""TDD tests for shared trendline computation module.

Tests accuracy for all 4 trendline types, edge cases,
and TrendlineResult structure.
"""

from __future__ import annotations

import numpy as np
import pytest
from plot_engine.trendline import TrendlineResult, compute_trendline

# ── Linear trendline ─────────────────────────────────────────────────────────


class TestLinearTrendline:
    def test_perfect_line(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 3.0
        result = compute_trendline(x, y, "linear")

        assert isinstance(result, TrendlineResult)
        assert result.trend_type == "linear"
        assert result.r_squared == pytest.approx(1.0, abs=1e-10)
        assert len(result.coefficients) == 2
        assert result.coefficients[0] == pytest.approx(2.0, abs=1e-6)
        assert result.coefficients[1] == pytest.approx(3.0, abs=1e-6)

    def test_equation_format(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 2.5 * x + 1.0
        result = compute_trendline(x, y, "linear")
        assert "y =" in result.equation
        assert "x" in result.equation

    def test_noisy_data(self):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y = 3.0 * x + 5.0 + rng.normal(0, 0.5, 100)
        result = compute_trendline(x, y, "linear")

        assert result.r_squared > 0.95
        assert result.coefficients[0] == pytest.approx(3.0, abs=0.3)
        assert result.coefficients[1] == pytest.approx(5.0, abs=1.0)

    def test_prediction_arrays(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        result = compute_trendline(x, y, "linear", n_points=50)

        assert len(result.x_pred) == 50
        assert len(result.y_pred) == 50
        assert result.x_pred[0] == pytest.approx(0.0)
        assert result.x_pred[-1] == pytest.approx(2.0)

    def test_negative_intercept(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([-5.0, -3.0, -1.0])
        result = compute_trendline(x, y, "linear")
        assert "-" in result.equation


# ── Polynomial trendline ─────────────────────────────────────────────────────


class TestPolynomialTrendline:
    def test_perfect_quadratic(self):
        x = np.linspace(-2, 2, 50)
        y = 3.0 * x**2 - 2.0 * x + 1.0
        result = compute_trendline(x, y, "polynomial", degree=2)

        assert result.trend_type == "polynomial"
        assert result.r_squared == pytest.approx(1.0, abs=1e-8)
        assert result.coefficients[0] == pytest.approx(3.0, abs=1e-4)
        assert result.coefficients[1] == pytest.approx(-2.0, abs=1e-4)
        assert result.coefficients[2] == pytest.approx(1.0, abs=1e-4)

    def test_cubic(self):
        x = np.linspace(0, 5, 100)
        y = x**3 - 2 * x**2 + x
        result = compute_trendline(x, y, "polynomial", degree=3)
        assert result.r_squared > 0.999

    def test_equation_contains_powers(self):
        x = np.linspace(0, 3, 20)
        y = x**2
        result = compute_trendline(x, y, "polynomial", degree=2)
        assert "x^2" in result.equation

    def test_degree_capped_at_data_size(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        # degree=5 but only 3 points — should cap at degree 2
        result = compute_trendline(x, y, "polynomial", degree=5)
        assert len(result.coefficients) == 3  # degree 2 + 1


# ── Exponential trendline ────────────────────────────────────────────────────


class TestExponentialTrendline:
    def test_perfect_exponential(self):
        x = np.linspace(0, 3, 50)
        y = 2.0 * np.exp(0.5 * x)
        result = compute_trendline(x, y, "exponential")

        assert result.trend_type == "exponential"
        assert result.r_squared > 0.999
        assert result.coefficients[0] == pytest.approx(2.0, abs=0.1)
        assert result.coefficients[1] == pytest.approx(0.5, abs=0.05)

    def test_equation_format(self):
        x = np.linspace(0, 2, 20)
        y = 1.5 * np.exp(0.3 * x)
        result = compute_trendline(x, y, "exponential")
        assert "exp(" in result.equation

    def test_requires_positive_y(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([-1.0, -2.0, -3.0])
        with pytest.raises(ValueError, match="positive y"):
            compute_trendline(x, y, "exponential")


# ── Power trendline ──────────────────────────────────────────────────────────


class TestPowerTrendline:
    def test_perfect_power(self):
        x = np.linspace(0.5, 5, 50)
        y = 3.0 * x**2.0
        result = compute_trendline(x, y, "power")

        assert result.trend_type == "power"
        assert result.r_squared > 0.999
        assert result.coefficients[0] == pytest.approx(3.0, abs=0.1)
        assert result.coefficients[1] == pytest.approx(2.0, abs=0.05)

    def test_equation_format(self):
        x = np.linspace(1, 5, 20)
        y = 2.0 * x**1.5
        result = compute_trendline(x, y, "power")
        assert "x^" in result.equation

    def test_requires_positive_x_and_y(self):
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            compute_trendline(x, y, "power")


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_two_points_linear(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 5.0])
        result = compute_trendline(x, y, "linear")
        assert result.coefficients[0] == pytest.approx(5.0)

    def test_insufficient_data(self):
        x = np.array([1.0])
        y = np.array([2.0])
        with pytest.raises(ValueError, match="At least 2"):
            compute_trendline(x, y, "linear")

    def test_empty_arrays(self):
        with pytest.raises(ValueError):
            compute_trendline(np.array([]), np.array([]), "linear")

    def test_nan_handling(self):
        x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, np.nan, 8.0])
        result = compute_trendline(x, y, "linear")
        # Should have filtered to 3 valid points: (0,0), (1,2), (4,8)
        assert result.r_squared > 0.9

    def test_all_nan(self):
        x = np.array([np.nan, np.nan])
        y = np.array([np.nan, np.nan])
        with pytest.raises(ValueError, match="At least 2"):
            compute_trendline(x, y, "linear")

    def test_unknown_trend_type(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown"):
            compute_trendline(x, y, "logarithmic")

    def test_default_n_points(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        result = compute_trendline(x, y, "linear")
        assert len(result.x_pred) == 200  # default

    def test_custom_n_points(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        result = compute_trendline(x, y, "linear", n_points=10)
        assert len(result.x_pred) == 10


# ── TrendlineResult structure ────────────────────────────────────────────────


class TestTrendlineResult:
    def test_fields(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 3.0, 5.0, 7.0])
        result = compute_trendline(x, y, "linear")

        assert hasattr(result, "trend_type")
        assert hasattr(result, "coefficients")
        assert hasattr(result, "equation")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "x_pred")
        assert hasattr(result, "y_pred")

    def test_repr_excludes_arrays(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        result = compute_trendline(x, y, "linear")
        r = repr(result)
        # x_pred and y_pred have repr=False
        assert "x_pred" not in r
        assert "y_pred" not in r
