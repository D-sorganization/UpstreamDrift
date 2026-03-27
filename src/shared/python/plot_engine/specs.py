"""PlotSpec Pydantic contract hierarchy.

Cross-platform plot specifications used by both matplotlib (PyQt6)
and Plotly.js (React/Tauri) renderers. This is the single source
of truth for plot configuration across all GUI implementations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SeriesStyle(BaseModel):
    """Visual style for a data series."""

    color: str | None = None
    line_style: Literal["solid", "dashed", "dotted", "dashdot"] = "solid"
    line_width: float = 1.5
    marker: Literal[
        "none", "circle", "square", "triangle", "diamond", "cross", "plus", "star"
    ] = "none"
    marker_size: float = 6.0
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    display_mode: Literal["line", "scatter", "line+scatter"] = "line"


class TrendlineSpec(BaseModel):
    """Trendline configuration for a data series."""

    type: Literal["linear", "polynomial", "exponential", "power"]
    degree: int = Field(default=2, ge=1, le=10)
    show_equation: bool = True
    show_r_squared: bool = True
    color: str | None = None
    line_style: Literal["solid", "dashed", "dotted", "dashdot"] = "dashed"


class AxisSpec(BaseModel):
    """Axis configuration."""

    label: str = ""
    min: float | None = None
    max: float | None = None
    log_scale: bool = False
    grid: bool = True


class LegendSpec(BaseModel):
    """Legend configuration."""

    visible: bool = True
    position: Literal["right", "left", "top", "bottom", "none"] = "right"
    labels: dict[str, str] = Field(default_factory=dict)


class SeriesData(BaseModel):
    """A single data series with optional styling and trendline."""

    name: str
    x: list[float]
    y: list[float]
    style: SeriesStyle = Field(default_factory=SeriesStyle)
    trendline: TrendlineSpec | None = None


class PlotSpec(BaseModel):
    """Base plot specification — the cross-platform contract."""

    title: str = ""
    series: list[SeriesData] = Field(default_factory=list)
    x_axis: AxisSpec = Field(default_factory=AxisSpec)
    y_axis: AxisSpec = Field(default_factory=AxisSpec)
    legend: LegendSpec = Field(default_factory=LegendSpec)
    width: int = Field(default=800, ge=100, le=4000)
    height: int = Field(default=600, ge=100, le=4000)


class SurfacePlotSpec(PlotSpec):
    """3D surface plot specification."""

    z_data: list[list[float]]
    x_grid: list[float]
    y_grid: list[float]
    z_axis: AxisSpec = Field(default_factory=AxisSpec)
    colormap: str = "viridis"
    opacity: float = Field(default=0.8, ge=0.0, le=1.0)
    show_wireframe: bool = False
    show_scatter: bool = True
    interpolation: Literal[
        "linear", "cubic", "nearest", "multiquadric", "inverse", "gaussian"
    ] = "linear"


class ContourPlotSpec(PlotSpec):
    """2D contour plot specification."""

    z_data: list[list[float]]
    x_grid: list[float]
    y_grid: list[float]
    levels: int = Field(default=20, ge=2, le=200)
    filled: bool = True
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_labels: bool = False


class HeatmapSpec(PlotSpec):
    """Heatmap specification."""

    z_data: list[list[float]]
    x_labels: list[str] = Field(default_factory=list)
    y_labels: list[str] = Field(default_factory=list)
    colormap: str = "YlGnBu"
    annotate: bool = False
    show_colorbar: bool = True


class HistogramSpec(PlotSpec):
    """Histogram specification."""

    bins: int = Field(default=30, ge=1, le=1000)
    density: bool = False
    cumulative: bool = False
    stacked: bool = False


class FilterComparisonSpec(PlotSpec):
    """Filter comparison overlay specification."""

    original_series: list[SeriesData] = Field(default_factory=list)
    filtered_series: list[SeriesData] = Field(default_factory=list)
    show_difference: bool = False
    difference_color: str = "#ff6b6b"
