"""TDD tests for PlotSpec Pydantic contract hierarchy.

Tests serialization roundtrips, field validation, defaults,
and JSON schema generation for all spec types.
"""

from __future__ import annotations

import json

import pytest
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
from pydantic import ValidationError

# ── SeriesStyle ──────────────────────────────────────────────────────────────


class TestSeriesStyle:
    def test_defaults(self):
        s = SeriesStyle()
        assert s.color is None
        assert s.line_style == "solid"
        assert s.line_width == 1.5
        assert s.marker == "none"
        assert s.marker_size == 6.0
        assert s.opacity == 1.0
        assert s.display_mode == "line"

    def test_roundtrip(self):
        s = SeriesStyle(
            color="#ff0000",
            line_style="dashed",
            marker="circle",
            display_mode="scatter",
        )
        d = s.model_dump()
        s2 = SeriesStyle(**d)
        assert s == s2

    def test_json_roundtrip(self):
        s = SeriesStyle(color="#00ff00", opacity=0.5)
        j = s.model_dump_json()
        s2 = SeriesStyle.model_validate_json(j)
        assert s == s2

    def test_opacity_bounds(self):
        with pytest.raises(ValidationError):
            SeriesStyle(opacity=1.5)
        with pytest.raises(ValidationError):
            SeriesStyle(opacity=-0.1)

    def test_invalid_line_style(self):
        with pytest.raises(ValidationError):
            SeriesStyle(line_style="wavy")

    def test_invalid_marker(self):
        with pytest.raises(ValidationError):
            SeriesStyle(marker="hexagon")

    def test_invalid_display_mode(self):
        with pytest.raises(ValidationError):
            SeriesStyle(display_mode="area")


# ── TrendlineSpec ────────────────────────────────────────────────────────────


class TestTrendlineSpec:
    def test_defaults(self):
        t = TrendlineSpec(type="linear")
        assert t.degree == 2
        assert t.show_equation is True
        assert t.show_r_squared is True
        assert t.line_style == "dashed"

    def test_roundtrip(self):
        t = TrendlineSpec(type="polynomial", degree=4, show_equation=False)
        t2 = TrendlineSpec(**t.model_dump())
        assert t == t2

    def test_degree_bounds(self):
        with pytest.raises(ValidationError):
            TrendlineSpec(type="polynomial", degree=0)
        with pytest.raises(ValidationError):
            TrendlineSpec(type="polynomial", degree=11)

    def test_valid_types(self):
        for tt in ["linear", "polynomial", "exponential", "power"]:
            t = TrendlineSpec(type=tt)
            assert t.type == tt


# ── AxisSpec ─────────────────────────────────────────────────────────────────


class TestAxisSpec:
    def test_defaults(self):
        a = AxisSpec()
        assert a.label == ""
        assert a.min is None
        assert a.max is None
        assert a.log_scale is False
        assert a.grid is True

    def test_roundtrip(self):
        a = AxisSpec(label="Time (s)", min=0.0, max=10.0, log_scale=True)
        a2 = AxisSpec(**a.model_dump())
        assert a == a2


# ── LegendSpec ───────────────────────────────────────────────────────────────


class TestLegendSpec:
    def test_defaults(self):
        lg = LegendSpec()
        assert lg.visible is True
        assert lg.position == "right"
        assert lg.labels == {}

    def test_custom_labels(self):
        lg = LegendSpec(labels={"signal_a": "Temperature", "signal_b": "Pressure"})
        assert lg.labels["signal_a"] == "Temperature"

    def test_roundtrip(self):
        lg = LegendSpec(visible=False, position="bottom", labels={"a": "A"})
        lg2 = LegendSpec(**lg.model_dump())
        assert lg == lg2


# ── SeriesData ───────────────────────────────────────────────────────────────


class TestSeriesData:
    def test_basic(self):
        sd = SeriesData(name="test", x=[1.0, 2.0, 3.0], y=[4.0, 5.0, 6.0])
        assert sd.name == "test"
        assert len(sd.x) == 3
        assert sd.trendline is None

    def test_with_style_and_trendline(self):
        sd = SeriesData(
            name="signal",
            x=[1.0, 2.0],
            y=[3.0, 4.0],
            style=SeriesStyle(color="#abcdef", marker="square"),
            trendline=TrendlineSpec(type="linear"),
        )
        assert sd.style.color == "#abcdef"
        assert sd.trendline.type == "linear"

    def test_roundtrip(self):
        sd = SeriesData(
            name="data",
            x=[0.0, 1.0, 2.0],
            y=[0.0, 1.0, 4.0],
            trendline=TrendlineSpec(type="polynomial", degree=2),
        )
        sd2 = SeriesData(**sd.model_dump())
        assert sd == sd2


# ── PlotSpec (base) ──────────────────────────────────────────────────────────


class TestPlotSpec:
    def test_defaults(self):
        p = PlotSpec()
        assert p.title == ""
        assert p.series == []
        assert p.width == 800
        assert p.height == 600

    def test_with_series(self):
        p = PlotSpec(
            title="My Plot",
            series=[
                SeriesData(name="a", x=[1.0], y=[2.0]),
                SeriesData(name="b", x=[3.0], y=[4.0]),
            ],
        )
        assert len(p.series) == 2

    def test_json_roundtrip(self):
        p = PlotSpec(
            title="Test",
            series=[SeriesData(name="s", x=[1.0, 2.0], y=[3.0, 4.0])],
            x_axis=AxisSpec(label="X"),
            y_axis=AxisSpec(label="Y"),
        )
        j = p.model_dump_json()
        p2 = PlotSpec.model_validate_json(j)
        assert p == p2

    def test_dimension_bounds(self):
        with pytest.raises(ValidationError):
            PlotSpec(width=50)
        with pytest.raises(ValidationError):
            PlotSpec(height=5000)

    def test_json_schema_generation(self):
        schema = PlotSpec.model_json_schema()
        assert "title" in schema["properties"]
        assert "series" in schema["properties"]
        assert schema["properties"]["width"]["default"] == 800


# ── SurfacePlotSpec ──────────────────────────────────────────────────────────


class TestSurfacePlotSpec:
    def test_required_fields(self):
        sp = SurfacePlotSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
        )
        assert sp.colormap == "viridis"
        assert sp.opacity == 0.8
        assert sp.show_wireframe is False

    def test_roundtrip(self):
        sp = SurfacePlotSpec(
            title="Surface",
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
            colormap="plasma",
            show_wireframe=True,
        )
        sp2 = SurfacePlotSpec(**sp.model_dump())
        assert sp == sp2

    def test_json_roundtrip(self):
        sp = SurfacePlotSpec(z_data=[[0.0]], x_grid=[0.0], y_grid=[0.0])
        j = sp.model_dump_json()
        sp2 = SurfacePlotSpec.model_validate_json(j)
        assert sp == sp2


# ── ContourPlotSpec ──────────────────────────────────────────────────────────


class TestContourPlotSpec:
    def test_defaults(self):
        cp = ContourPlotSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
        )
        assert cp.levels == 20
        assert cp.filled is True
        assert cp.show_colorbar is True
        assert cp.show_labels is False

    def test_levels_bounds(self):
        with pytest.raises(ValidationError):
            ContourPlotSpec(z_data=[[1.0]], x_grid=[0.0], y_grid=[0.0], levels=1)

    def test_roundtrip(self):
        cp = ContourPlotSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_grid=[0.0, 1.0],
            y_grid=[0.0, 1.0],
            levels=50,
            filled=False,
            show_labels=True,
        )
        cp2 = ContourPlotSpec(**cp.model_dump())
        assert cp == cp2


# ── HeatmapSpec ──────────────────────────────────────────────────────────────


class TestHeatmapSpec:
    def test_defaults(self):
        hm = HeatmapSpec(z_data=[[1.0, 2.0], [3.0, 4.0]])
        assert hm.colormap == "YlGnBu"
        assert hm.annotate is False
        assert hm.x_labels == []
        assert hm.y_labels == []

    def test_with_labels(self):
        hm = HeatmapSpec(
            z_data=[[1.0, 2.0], [3.0, 4.0]],
            x_labels=["A", "B"],
            y_labels=["C", "D"],
            annotate=True,
        )
        assert hm.x_labels == ["A", "B"]

    def test_roundtrip(self):
        hm = HeatmapSpec(z_data=[[0.5]])
        hm2 = HeatmapSpec(**hm.model_dump())
        assert hm == hm2


# ── HistogramSpec ────────────────────────────────────────────────────────────


class TestHistogramSpec:
    def test_defaults(self):
        h = HistogramSpec()
        assert h.bins == 30
        assert h.density is False
        assert h.cumulative is False
        assert h.stacked is False

    def test_bins_bounds(self):
        with pytest.raises(ValidationError):
            HistogramSpec(bins=0)

    def test_roundtrip(self):
        h = HistogramSpec(bins=50, density=True, cumulative=True)
        h2 = HistogramSpec(**h.model_dump())
        assert h == h2


# ── FilterComparisonSpec ─────────────────────────────────────────────────────


class TestFilterComparisonSpec:
    def test_defaults(self):
        fc = FilterComparisonSpec()
        assert fc.original_series == []
        assert fc.filtered_series == []
        assert fc.show_difference is False
        assert fc.difference_color == "#ff6b6b"

    def test_with_series(self):
        orig = SeriesData(name="raw", x=[1.0, 2.0], y=[3.0, 4.0])
        filt = SeriesData(
            name="filtered",
            x=[1.0, 2.0],
            y=[3.1, 3.9],
            style=SeriesStyle(line_style="dashed"),
        )
        fc = FilterComparisonSpec(
            original_series=[orig],
            filtered_series=[filt],
            show_difference=True,
        )
        assert len(fc.original_series) == 1
        assert fc.show_difference is True

    def test_json_roundtrip(self):
        fc = FilterComparisonSpec(
            original_series=[SeriesData(name="o", x=[0.0], y=[1.0])],
            filtered_series=[SeriesData(name="f", x=[0.0], y=[0.9])],
            show_difference=True,
        )
        j = fc.model_dump_json()
        fc2 = FilterComparisonSpec.model_validate_json(j)
        assert fc == fc2


# ── Cross-spec JSON schema ───────────────────────────────────────────────────


class TestSchemaGeneration:
    @pytest.mark.parametrize(
        "spec_cls",
        [
            PlotSpec,
            SurfacePlotSpec,
            ContourPlotSpec,
            HeatmapSpec,
            HistogramSpec,
            FilterComparisonSpec,
        ],
    )
    def test_json_schema_is_valid_json(self, spec_cls):
        schema = spec_cls.model_json_schema()
        # Should serialize to valid JSON
        j = json.dumps(schema)
        parsed = json.loads(j)
        assert "properties" in parsed
        assert "title" in parsed

    @pytest.mark.parametrize(
        "spec_cls",
        [
            PlotSpec,
            SurfacePlotSpec,
            ContourPlotSpec,
            HeatmapSpec,
            HistogramSpec,
            FilterComparisonSpec,
        ],
    )
    def test_inherits_base_fields(self, spec_cls):
        schema = spec_cls.model_json_schema()
        # All specs should have title, width, height
        props = schema.get("properties", {})
        assert "title" in props
        assert "width" in props
        assert "height" in props
