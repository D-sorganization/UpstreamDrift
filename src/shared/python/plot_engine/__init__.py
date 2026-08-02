"""Shared plot engine for cross-platform visualization.

This module provides a DRY plotting system that renders the same PlotSpec
contracts to both matplotlib (PyQt6) and Plotly.js JSON (React/Tauri).

Usage:
    from plot_engine.specs import PlotSpec, SeriesData
    from plot_engine.matplotlib_renderer import MatplotlibRenderer
    from plot_engine.plotly_converter import PlotlyConverter

    spec = PlotSpec(title="My Plot", series=[...])

    # PyQt6 path
    fig = MatplotlibRenderer().render(spec)

    # React/Tauri path
    plotly_json = PlotlyConverter().convert(spec)
"""

from .protocols import PlotConverter, PlotRenderer, ThemeColorProvider
from .specs import (
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

__all__ = [
    # Protocols
    "PlotConverter",
    "PlotRenderer",
    "ThemeColorProvider",
    # Specs
    "AxisSpec",
    "ContourPlotSpec",
    "FilterComparisonSpec",
    "HeatmapSpec",
    "HistogramSpec",
    "LegendSpec",
    "PlotSpec",
    "SeriesData",
    "SeriesStyle",
    "SurfacePlotSpec",
    "TrendlineSpec",
]
