"""Tests for plot_engine.matplotlib_renderer and plot_engine.plotly_converter (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.plot_engine.matplotlib_renderer import MatplotlibRenderer
from src.shared.python.plot_engine.plotly_converter import PlotlyConverter
from src.shared.python.plot_engine.specs import PlotSpec, SeriesData


class TestMatplotlibRenderer:
    def test_plot_engine_renderers_construction(self) -> None:
        renderer = MatplotlibRenderer()
        assert renderer is not None

    def test_render_empty_spec(self) -> None:
        renderer = MatplotlibRenderer()
        spec = PlotSpec(title="Empty")
        fig = renderer.render(spec)
        assert fig is not None

    def test_render_returns_figure(self) -> None:
        import matplotlib.figure

        renderer = MatplotlibRenderer()
        spec = PlotSpec(title="Test")
        fig = renderer.render(spec)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_render_with_series(self) -> None:
        renderer = MatplotlibRenderer()
        series = SeriesData(name="data", x=[1.0, 2.0, 3.0], y=[1.0, 4.0, 9.0])
        spec = PlotSpec(title="With Series", series=[series])
        fig = renderer.render(spec)
        assert fig is not None


class TestPlotlyConverter:
    def test_plot_engine_renderers_construction(self) -> None:
        converter = PlotlyConverter()
        assert converter is not None

    def test_convert_returns_something(self) -> None:
        converter = PlotlyConverter()
        spec = PlotSpec(title="Converted")
        result = converter.convert(spec)
        assert result is not None
