"""TDD tests for Plotly.js converter.

Tests that each PlotSpec type produces valid Plotly trace structures
with correct type fields, data arrays, and layout settings.
"""

from __future__ import annotations

import pytest
from plot_engine.plotly_converter import PlotlyConverter
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
def converter():
    return PlotlyConverter()


# ── Line/Scatter traces ─────────────────────────────────────────────────────


class TestLineScatter:
    def test_empty_spec(self, converter):
        result = converter.convert(PlotSpec())
        assert "data" in result
        assert "layout" in result
        assert result["data"] == []

    def test_single_line_trace(self, converter):
        spec = PlotSpec(
            series=[SeriesData(name="s1", x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 4.0])],
        )
        result = converter.convert(spec)
        assert len(result["data"]) == 1
        trace = result["data"][0]
        assert trace["type"] == "scatter"
        assert trace["mode"] == "lines"
        assert trace["x"] == [0.0, 1.0, 2.0]
        assert trace["y"] == [0.0, 1.0, 4.0]
        assert trace["name"] == "s1"

    def test_scatter_mode(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="pts",
                    x=[1.0, 2.0],
                    y=[3.0, 4.0],
                    style=SeriesStyle(display_mode="scatter", marker="circle"),
                ),
            ],
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["mode"] == "markers"
        assert trace["marker"]["symbol"] == "circle"

    def test_line_plus_scatter(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="both",
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    style=SeriesStyle(display_mode="line+scatter", marker="square"),
                ),
            ],
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["mode"] == "lines+markers"
        assert trace["marker"]["symbol"] == "square"

    def test_custom_color(self, converter):
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
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["line"]["color"] == "#ff0000"

    def test_line_style_dashed(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="dashed",
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    style=SeriesStyle(line_style="dashed"),
                ),
            ],
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["line"]["dash"] == "dash"

    def test_opacity(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="transparent",
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    style=SeriesStyle(opacity=0.5),
                ),
            ],
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["opacity"] == 0.5

    def test_multiple_series(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(name="a", x=[0.0, 1.0], y=[0.0, 1.0]),
                SeriesData(name="b", x=[0.0, 1.0], y=[1.0, 0.0]),
            ],
        )
        result = converter.convert(spec)
        assert len(result["data"]) == 2
        assert result["data"][0]["name"] == "a"
        assert result["data"][1]["name"] == "b"


# ── Trendlines ───────────────────────────────────────────────────────────────


class TestTrendlines:
    def test_linear_trendline_trace(self, converter):
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
        result = converter.convert(spec)
        # Data trace + trendline trace
        assert len(result["data"]) == 2
        trend = result["data"][1]
        assert trend["type"] == "scatter"
        assert trend["mode"] == "lines"
        assert "y =" in trend["name"]

    def test_trendline_with_r_squared(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="d",
                    x=[0.0, 1.0, 2.0],
                    y=[0.0, 1.0, 2.0],
                    trendline=TrendlineSpec(
                        type="linear", show_equation=True, show_r_squared=True
                    ),
                ),
            ],
        )
        result = converter.convert(spec)
        trend = result["data"][1]
        assert "R\u00b2" in trend["name"]

    def test_trendline_custom_color(self, converter):
        spec = PlotSpec(
            series=[
                SeriesData(
                    name="d",
                    x=[0.0, 1.0, 2.0],
                    y=[0.0, 1.0, 2.0],
                    trendline=TrendlineSpec(type="linear", color="#00ff00"),
                ),
            ],
        )
        result = converter.convert(spec)
        trend = result["data"][1]
        assert trend["line"]["color"] == "#00ff00"


# ── Surface ──────────────────────────────────────────────────────────────────


class TestSurface:
    def test_basic_surface(self, converter):
        spec = SurfacePlotSpec(
            title="Surface",
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
        )
        result = converter.convert(spec)
        assert len(result["data"]) >= 1
        surface = result["data"][0]
        assert surface["type"] == "surface"
        assert surface["z"] == [[1.0, 2.0], [3.0, 4.0]]
        assert surface["colorscale"] == "viridis"

    def test_surface_with_scatter(self, converter):
        spec = SurfacePlotSpec(
            z_data=[[1.0]],
            x_grid=[0.0],
            y_grid=[0.0],
            show_scatter=True,
        )
        result = converter.convert(spec)
        assert len(result["data"]) == 2
        assert result["data"][1]["type"] == "scatter3d"

    def test_surface_no_scatter(self, converter):
        spec = SurfacePlotSpec(
            z_data=[[1.0]],
            x_grid=[0.0],
            y_grid=[0.0],
            show_scatter=False,
        )
        result = converter.convert(spec)
        assert len(result["data"]) == 1

    def test_surface_has_scene(self, converter):
        spec = SurfacePlotSpec(
            z_data=[[1.0]],
            x_grid=[0.0],
            y_grid=[0.0],
        )
        result = converter.convert(spec)
        assert "scene" in result["layout"]


# ── Contour ──────────────────────────────────────────────────────────────────


class TestContour:
    def test_basic_contour(self, converter):
        spec = ContourPlotSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
        )
        result = converter.convert(spec)
        assert len(result["data"]) == 1
        trace = result["data"][0]
        assert trace["type"] == "contour"
        assert trace["ncontours"] == 20

    def test_filled_contour(self, converter):
        spec = ContourPlotSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
            filled=True,
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["contours"]["coloring"] == "heatmap"

    def test_contour_no_colorbar(self, converter):
        spec = ContourPlotSpec(
            z_data=[[1.0]],
            x_grid=[0.0],
            y_grid=[0.0],
            show_colorbar=False,
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["showscale"] is False


# ── Heatmap ──────────────────────────────────────────────────────────────────


class TestHeatmap:
    def test_basic_heatmap(self, converter):
        spec = HeatmapSpec(z_data=[[1.0, 2.0], [3.0, 4.0]])
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["type"] == "heatmap"
        assert trace["z"] == [[1.0, 2.0], [3.0, 4.0]]

    def test_heatmap_with_labels(self, converter):
        spec = HeatmapSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_labels=["A", "B"],
            y_labels=["C", "D"],
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["x"] == ["A", "B"]
        assert trace["y"] == ["C", "D"]

    def test_heatmap_annotated(self, converter):
        spec = HeatmapSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            annotate=True,
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert "text" in trace
        assert "texttemplate" in trace

    def test_heatmap_colorscale(self, converter):
        spec = HeatmapSpec(
            z_data=[[1.0]],
            colormap="plasma",
        )
        result = converter.convert(spec)
        assert result["data"][0]["colorscale"] == "plasma"


# ── Histogram ────────────────────────────────────────────────────────────────


class TestHistogram:
    def test_basic_histogram(self, converter):
        spec = HistogramSpec(
            series=[
                SeriesData(
                    name="d", x=list(range(100)), y=[float(i) for i in range(100)]
                )
            ],
            bins=20,
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["type"] == "histogram"
        assert trace["nbinsx"] == 20

    def test_density_histogram(self, converter):
        spec = HistogramSpec(
            series=[SeriesData(name="d", x=[0.0], y=[1.0])],
            density=True,
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["histnorm"] == "probability density"

    def test_cumulative_histogram(self, converter):
        spec = HistogramSpec(
            series=[SeriesData(name="d", x=[0.0], y=[1.0])],
            cumulative=True,
        )
        result = converter.convert(spec)
        trace = result["data"][0]
        assert trace["cumulative"]["enabled"] is True

    def test_stacked_layout(self, converter):
        spec = HistogramSpec(
            series=[
                SeriesData(name="a", x=[0.0], y=[1.0]),
                SeriesData(name="b", x=[0.0], y=[2.0]),
            ],
            stacked=True,
        )
        result = converter.convert(spec)
        assert result["layout"]["barmode"] == "stack"


# ── Filter comparison ────────────────────────────────────────────────────────


class TestFilterComparison:
    def test_basic_comparison(self, converter):
        spec = FilterComparisonSpec(
            original_series=[SeriesData(name="raw", x=[0.0, 1.0], y=[1.0, 2.0])],
            filtered_series=[SeriesData(name="filt", x=[0.0, 1.0], y=[0.9, 2.1])],
        )
        result = converter.convert(spec)
        assert len(result["data"]) == 2
        assert "Original:" in result["data"][0]["name"]
        assert "Filtered:" in result["data"][1]["name"]

    def test_filtered_default_dashed(self, converter):
        spec = FilterComparisonSpec(
            original_series=[SeriesData(name="o", x=[0.0], y=[1.0])],
            filtered_series=[SeriesData(name="f", x=[0.0], y=[0.9])],
        )
        result = converter.convert(spec)
        filt_trace = result["data"][1]
        assert filt_trace["line"]["dash"] == "dash"

    def test_with_difference(self, converter):
        spec = FilterComparisonSpec(
            original_series=[SeriesData(name="o", x=[0.0, 1.0], y=[1.0, 2.0])],
            filtered_series=[SeriesData(name="f", x=[0.0, 1.0], y=[0.9, 2.1])],
            show_difference=True,
        )
        result = converter.convert(spec)
        # 2 main traces + 1 difference trace
        assert len(result["data"]) == 3
        diff_trace = result["data"][2]
        assert diff_trace["yaxis"] == "y2"
        assert "Diff:" in diff_trace["name"]
        assert diff_trace["line"]["color"] == "#ff6b6b"

    def test_difference_secondary_axis(self, converter):
        spec = FilterComparisonSpec(
            original_series=[SeriesData(name="o", x=[0.0], y=[1.0])],
            filtered_series=[SeriesData(name="f", x=[0.0], y=[0.9])],
            show_difference=True,
        )
        result = converter.convert(spec)
        assert "yaxis2" in result["layout"]


# ── Layout ───────────────────────────────────────────────────────────────────


class TestLayout:
    def test_title(self, converter):
        result = converter.convert(PlotSpec(title="My Title"))
        assert result["layout"]["title"]["text"] == "My Title"

    def test_no_title(self, converter):
        result = converter.convert(PlotSpec())
        assert "title" not in result["layout"]

    def test_dimensions(self, converter):
        result = converter.convert(PlotSpec(width=1200, height=400))
        assert result["layout"]["width"] == 1200
        assert result["layout"]["height"] == 400

    def test_axis_labels(self, converter):
        result = converter.convert(
            PlotSpec(
                x_axis=AxisSpec(label="Time"),
                y_axis=AxisSpec(label="Value"),
            )
        )
        assert result["layout"]["xaxis"]["title"]["text"] == "Time"
        assert result["layout"]["yaxis"]["title"]["text"] == "Value"

    def test_axis_log_scale(self, converter):
        result = converter.convert(PlotSpec(x_axis=AxisSpec(log_scale=True)))
        assert result["layout"]["xaxis"]["type"] == "log"

    def test_grid_setting(self, converter):
        result = converter.convert(PlotSpec(x_axis=AxisSpec(grid=False)))
        assert result["layout"]["xaxis"]["showgrid"] is False

    def test_legend_hidden(self, converter):
        result = converter.convert(PlotSpec(legend=LegendSpec(visible=False)))
        assert result["layout"]["showlegend"] is False

    def test_legend_position_right(self, converter):
        result = converter.convert(
            PlotSpec(
                series=[SeriesData(name="a", x=[0.0], y=[0.0])],
                legend=LegendSpec(visible=True, position="right"),
            )
        )
        assert result["layout"]["showlegend"] is True
        assert result["layout"]["legend"]["x"] == 1.02

    def test_legend_position_bottom(self, converter):
        result = converter.convert(
            PlotSpec(
                series=[SeriesData(name="a", x=[0.0], y=[0.0])],
                legend=LegendSpec(position="bottom"),
            )
        )
        assert result["layout"]["legend"]["orientation"] == "h"
