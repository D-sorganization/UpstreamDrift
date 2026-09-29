"""Tests for plot_engine.specs, plot_engine.trendline, and plot_engine.protocols (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.plot_engine.specs import (
    HistogramSpec,
    PlotSpec,
    SeriesData,
    SeriesStyle,
    SurfacePlotSpec,
    TrendlineSpec,
)
from src.shared.python.plot_engine.trendline import TrendlineResult, compute_trendline


class TestSeriesStyle:
    def test_plot_engine_specs_default_construction(self) -> None:
        s = SeriesStyle()
        assert s is not None

    def test_color_field(self) -> None:
        s = SeriesStyle(color="#ff0000")
        assert s.color == "#ff0000"


class TestTrendlineSpec:
    def test_construction_with_type(self) -> None:
        t = TrendlineSpec(type="linear")
        assert t.type == "linear"

    def test_polynomial_type(self) -> None:
        t = TrendlineSpec(type="polynomial", degree=3)
        assert t.degree == 3


class TestPlotSpec:
    def test_plot_engine_specs_construction(self) -> None:
        spec = PlotSpec(title="Test Plot")
        assert spec.title == "Test Plot"

    def test_plot_engine_specs_defaults(self) -> None:
        spec = PlotSpec()
        assert spec is not None


class TestSeriesData:
    def test_plot_engine_specs_construction(self) -> None:
        s = SeriesData(name="series1", x=[1.0, 2.0], y=[3.0, 4.0])
        assert s.name == "series1"


class TestSurfacePlotSpec:
    def test_plot_engine_specs_construction(self) -> None:
        spec = SurfacePlotSpec(
            title="Surface",
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
        )
        assert spec.title == "Surface"


class TestHistogramSpec:
    def test_plot_engine_specs_construction(self) -> None:
        spec = HistogramSpec(title="Histogram")
        assert spec is not None


class TestTrendline:
    def _make_linear_data(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 10.0, 20)
        y = 2.0 * x + 1.0 + np.random.default_rng(42).normal(0, 0.1, 20)
        return x, y

    def test_returns_trendline_result(self) -> None:
        x, y = self._make_linear_data()
        result = compute_trendline(x, y, "linear")
        assert isinstance(result, TrendlineResult)

    def test_linear_r_squared_near_one(self) -> None:
        x, y = self._make_linear_data()
        result = compute_trendline(x, y, "linear")
        assert result.r_squared > 0.99

    def test_polynomial_trendline(self) -> None:
        x = np.linspace(0.0, 5.0, 20)
        y = x**2 + 1.0
        result = compute_trendline(x, y, "polynomial")
        assert isinstance(result, TrendlineResult)

    def test_exponential_trendline(self) -> None:
        x = np.linspace(0.5, 5.0, 20)
        y = 2.0 * np.exp(0.5 * x)
        result = compute_trendline(x, y, "exponential")
        assert isinstance(result, TrendlineResult)
        assert result.r_squared > 0.99

    def test_trendline_result_has_y_pred(self) -> None:
        x, y = self._make_linear_data()
        result = compute_trendline(x, y, "linear", n_points=20)
        assert len(result.y_pred) == 20

    def test_trendline_result_has_equation(self) -> None:
        x, y = self._make_linear_data()
        result = compute_trendline(x, y, "linear")
        assert isinstance(result.equation, str)
