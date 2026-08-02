"""TDD tests for matplotlib renderer.

Tests that each plot type produces valid Figure objects with
correct structure, series rendering, trendline annotations,
and theme integration.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pytest
from plot_engine.matplotlib_renderer import MatplotlibRenderer
from plot_engine.specs import (
    AxisSpec,
    ContourPlotSpec,
    FilterComparisonSpec,
    HeatmapSpec,
    HistogramSpec,
    LegendSpec,
    PlotSpec,
    SeriesData,
    SeriesStyle,
    SurfacePlotSpec,
    TrendlineSpec,
)


@pytest.fixture()
def renderer():
    return MatplotlibRenderer()


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ── Line/Scatter plot ────────────────────────────────────────────────────────


class TestLinePlot:
    def test_empty_spec(self, renderer):
        fig = renderer.render(PlotSpec())
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_single_series(self, renderer):
        spec = PlotSpec(
            title="Test",
            series=[SeriesData(name="s1", x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 4.0])],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert ax.get_title() == "Test"
        assert len(ax.lines) >= 1

    def test_multiple_series(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(name="a", x=[0.0, 1.0], y=[0.0, 1.0]),
                SeriesData(name="b", x=[0.0, 1.0], y=[1.0, 0.0]),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert len(ax.lines) >= 2

    def test_scatter_mode(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="scatter",
                    x=[0.0, 1.0, 2.0],
                    y=[3.0, 1.0, 2.0],
                    style=SeriesStyle(display_mode="scatter", marker="circle"),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # Scatter creates PathCollections, not lines
        assert len(ax.collections) >= 1

    def test_line_plus_scatter(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="both",
                    x=[0.0, 1.0, 2.0],
                    y=[0.0, 1.0, 2.0],
                    style=SeriesStyle(display_mode="line+scatter", marker="square"),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1

    def test_custom_colors(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="red",
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    style=SeriesStyle(color="#ff0000"),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert len(ax.lines) >= 1

    def test_axis_labels(self, renderer):
        spec = PlotSpec(
            x_axis=AxisSpec(label="Time (s)"),
            y_axis=AxisSpec(label="Voltage (V)"),
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Voltage (V)"

    def test_axis_limits(self, renderer):
        spec = PlotSpec(
            series=[SeriesData(name="s", x=[0.0, 10.0], y=[0.0, 10.0])],
            x_axis=AxisSpec(min=2.0, max=8.0),
            y_axis=AxisSpec(min=1.0, max=9.0),
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] == pytest.approx(2.0)
        assert xlim[1] == pytest.approx(8.0)
        assert ylim[0] == pytest.approx(1.0)
        assert ylim[1] == pytest.approx(9.0)

    def test_log_scale(self, renderer):
        spec = PlotSpec(
            series=[SeriesData(name="s", x=[1.0, 10.0, 100.0], y=[1.0, 2.0, 3.0])],
            x_axis=AxisSpec(log_scale=True),
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"


# ── Trendlines ───────────────────────────────────────────────────────────────


class TestTrendlines:
    def test_linear_trendline(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="data",
                    x=[0.0, 1.0, 2.0, 3.0, 4.0],
                    y=[1.0, 3.0, 5.0, 7.0, 9.0],
                    trendline=TrendlineSpec(type="linear"),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # Original data + trendline = at least 2 lines
        assert len(ax.lines) >= 2

    def test_trendline_annotation(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="data",
                    x=[0.0, 1.0, 2.0, 3.0],
                    y=[0.0, 2.0, 4.0, 6.0],
                    trendline=TrendlineSpec(
                        type="linear",
                        show_equation=True,
                        show_r_squared=True,
                    ),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # Should have at least one annotation (equation + R²)
        annotation_texts = [t.get_text() for t in ax.texts]
        has_equation = any("y =" in t for t in annotation_texts)
        has_r2 = any("R\u00b2" in t for t in annotation_texts)
        assert has_equation or has_r2

    def test_trendline_no_equation(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="data",
                    x=[0.0, 1.0, 2.0],
                    y=[0.0, 1.0, 2.0],
                    trendline=TrendlineSpec(
                        type="linear",
                        show_equation=False,
                        show_r_squared=False,
                    ),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # Trendline drawn but no text annotations
        equation_texts = [t.get_text() for t in ax.texts if "y =" in t.get_text()]
        assert len(equation_texts) == 0

    def test_polynomial_trendline(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="data",
                    x=[0.0, 1.0, 2.0, 3.0, 4.0],
                    y=[0.0, 1.0, 4.0, 9.0, 16.0],
                    trendline=TrendlineSpec(type="polynomial", degree=2),
                ),
            ],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert len(ax.lines) >= 2


# ── Surface plot ─────────────────────────────────────────────────────────────


class TestSurfacePlot:
    def test_basic_surface(self, renderer):
        x = [0.0, 1.0, 2.0]
        y = [0.0, 1.0, 2.0]
        z = [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        spec = SurfacePlotSpec(
            title="Surface",
            z_data=z,
            x_grid=x,
            y_grid=y,
        )
        fig = renderer.render(spec)
        assert fig is not None
        # Should have a 3D axes
        ax = fig.axes[0]
        assert ax.name == "3d"

    def test_surface_title(self, renderer):
        spec = SurfacePlotSpec(
            title="My Surface",
            z_data=[[1.0]],
            x_grid=[0.0],
            y_grid=[0.0],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert ax.get_title() == "My Surface"


# ── Contour plot ─────────────────────────────────────────────────────────────


class TestContourPlot:
    def test_basic_contour(self, renderer):
        x = np.linspace(0, 1, 10).tolist()
        y = np.linspace(0, 1, 10).tolist()
        xm, ym = np.meshgrid(x, y)
        z = (np.sin(xm * np.pi) * np.cos(ym * np.pi)).tolist()
        spec = ContourPlotSpec(
            title="Contour",
            z_data=z,
            x_grid=x,
            y_grid=y,
            levels=10,
        )
        fig = renderer.render(spec)
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_filled_contour_has_colorbar(self, renderer):
        x = np.linspace(0, 1, 5).tolist()
        y = np.linspace(0, 1, 5).tolist()
        xm, ym = np.meshgrid(x, y)
        z = (xm + ym).tolist()
        spec = ContourPlotSpec(
            z_data=z,
            x_grid=x,
            y_grid=y,
            filled=True,
            show_colorbar=True,
        )
        fig = renderer.render(spec)
        # Colorbar adds an extra axes
        assert len(fig.axes) >= 2

    def test_unfilled_contour(self, renderer):
        x = np.linspace(0, 1, 5).tolist()
        y = np.linspace(0, 1, 5).tolist()
        xm, ym = np.meshgrid(x, y)
        z = (xm * ym).tolist()
        spec = ContourPlotSpec(
            z_data=z,
            x_grid=x,
            y_grid=y,
            filled=False,
            show_colorbar=False,
        )
        fig = renderer.render(spec)
        assert len(fig.axes) == 1


# ── Heatmap ──────────────────────────────────────────────────────────────────


class TestHeatmap:
    def test_basic_heatmap(self, renderer):
        spec = HeatmapSpec(
            title="Heatmap",
            z_data=[[1.0, 2.0], [3.0, 4.0]],
        )
        fig = renderer.render(spec)
        assert fig is not None
        ax = fig.axes[0]
        assert len(ax.images) >= 1

    def test_heatmap_with_labels(self, renderer):
        spec = HeatmapSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_labels=["A", "B"],
            y_labels=["C", "D"],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert len(ax.images) >= 1

    def test_heatmap_annotated(self, renderer):
        spec = HeatmapSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            annotate=True,
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # 4 text annotations (one per cell)
        assert len(ax.texts) == 4

    def test_heatmap_with_colorbar(self, renderer):
        spec = HeatmapSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            show_colorbar=True,
        )
        fig = renderer.render(spec)
        assert len(fig.axes) >= 2


# ── Histogram ────────────────────────────────────────────────────────────────


class TestHistogram:
    def test_basic_histogram(self, renderer):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100).tolist()
        spec = HistogramSpec(
            title="Distribution",
            series=[SeriesData(name="data", x=list(range(100)), y=data)],
            bins=20,
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert ax.get_title() == "Distribution"
        # Histogram creates patches (rectangles)
        assert len(ax.patches) > 0

    def test_density_histogram(self, renderer):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 50).tolist()
        spec = HistogramSpec(
            series=[SeriesData(name="d", x=list(range(50)), y=data)],
            density=True,
        )
        fig = renderer.render(spec)
        assert fig is not None


# ── Filter comparison ────────────────────────────────────────────────────────


class TestFilterComparison:
    def test_basic_comparison(self, renderer):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        orig_y = [1.0, 3.0, 2.0, 4.0, 3.0]
        filt_y = [1.2, 2.8, 2.1, 3.8, 3.1]
        spec = FilterComparisonSpec(
            title="Filter Comparison",
            original_series=[SeriesData(name="raw", x=x, y=orig_y)],
            filtered_series=[SeriesData(name="LP filter", x=x, y=filt_y)],
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        assert ax is not None
        assert len(ax.lines) >= 2

    def test_with_difference(self, renderer):
        x = [0.0, 1.0, 2.0]
        spec = FilterComparisonSpec(
            original_series=[SeriesData(name="o", x=x, y=[1.0, 2.0, 3.0])],
            filtered_series=[SeriesData(name="f", x=x, y=[0.9, 2.1, 2.9])],
            show_difference=True,
        )
        fig = renderer.render(spec)
        # Should have 2 subplots
        assert len(fig.axes) >= 2


# ── Legend ────────────────────────────────────────────────────────────────────


class TestLegend:
    def test_legend_visible(self, renderer):
        spec = PlotSpec(
            series=[
                SeriesData(name="a", x=[0.0, 1.0], y=[0.0, 1.0]),
                SeriesData(name="b", x=[0.0, 1.0], y=[1.0, 0.0]),
            ],
            legend=LegendSpec(visible=True, position="right"),
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    def test_legend_hidden(self, renderer):
        spec = PlotSpec(
            series=[SeriesData(name="a", x=[0.0], y=[0.0])],
            legend=LegendSpec(visible=False),
        )
        fig = renderer.render(spec)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is None


# ── to_image ─────────────────────────────────────────────────────────────────


class TestToImage:
    def test_png_output(self, renderer):
        spec = PlotSpec(
            series=[SeriesData(name="s", x=[0.0, 1.0], y=[0.0, 1.0])],
        )
        img_bytes = renderer.to_image(spec, fmt="png")
        assert isinstance(img_bytes, bytes)
        assert len(img_bytes) > 0
        # PNG magic number
        assert img_bytes[:4] == b"\x89PNG"

    def test_svg_output(self, renderer):
        spec = PlotSpec(
            series=[SeriesData(name="s", x=[0.0, 1.0], y=[0.0, 1.0])],
        )
        img_bytes = renderer.to_image(spec, fmt="svg")
        assert b"<svg" in img_bytes


# ── Figure dimensions ────────────────────────────────────────────────────────


class TestDimensions:
    def test_custom_dimensions(self, renderer):
        spec = PlotSpec(width=1200, height=400)
        fig = renderer.render(spec)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12.0, abs=0.5)
        assert h == pytest.approx(4.0, abs=0.5)
