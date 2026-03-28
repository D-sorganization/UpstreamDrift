"""Plotly.js JSON converter for PlotSpec contracts.

Converts PlotSpec objects into Plotly.js-compatible JSON dicts
that React can pass directly to <Plot data={...} layout={...} />.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .specs import (
    ContourPlotSpec,
    FilterComparisonSpec,
    HeatmapSpec,
    HistogramSpec,
    PlotSpec,
    SeriesData,
    SurfacePlotSpec,
)
from .trendline import compute_trendline

logger = logging.getLogger(__name__)

# Line style mapping to Plotly dash values
_DASH_MAP = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
    "dashdot": "dashdot",
}

# Marker mapping to Plotly symbol names
_SYMBOL_MAP = {
    "none": None,
    "circle": "circle",
    "square": "square",
    "triangle": "triangle-up",
    "diamond": "diamond",
    "cross": "x",
    "plus": "cross",
    "star": "star",
}


class PlotlyConverter:
    """Converts PlotSpec contracts to Plotly.js JSON."""

    def convert(self, spec: PlotSpec) -> dict[str, Any]:
        """Convert a PlotSpec to Plotly.js JSON.

        Returns:
            Dict with "data" (list of traces) and "layout" keys.
        """
        if not (spec is not None):
            raise ValueError("spec must be provided")
        if isinstance(spec, SurfacePlotSpec):
            return self._surface(spec)
        if isinstance(spec, ContourPlotSpec):
            return self._contour(spec)
        if isinstance(spec, HeatmapSpec):
            return self._heatmap(spec)
        if isinstance(spec, HistogramSpec):
            return self._histogram(spec)
        if isinstance(spec, FilterComparisonSpec):
            return self._filter_comparison(spec)
        return self._line_scatter(spec)

    # ── Type-specific converters ─────────────────────────────────────────────

    def _line_scatter(self, spec: PlotSpec) -> dict[str, Any]:
        if not (spec is not None):
            raise ValueError("spec must be provided")
        traces = []
        for series in spec.series:
            traces.append(self._series_trace(series))
            if series.trendline is not None:
                trend_trace = self._trendline_trace(series)
                if trend_trace:
                    traces.append(trend_trace)

        return {"data": traces, "layout": self._build_layout(spec)}

    def _surface(self, spec: SurfacePlotSpec) -> dict[str, Any]:
        if not (spec is not None):
            raise ValueError("spec must be provided")
        trace: dict[str, Any] = {
            "type": "surface",
            "z": spec.z_data,
            "x": spec.x_grid,
            "y": spec.y_grid,
            "colorscale": spec.colormap,
            "opacity": spec.opacity,
        }

        if spec.show_wireframe:
            trace["contours"] = {
                "x": {"show": True, "color": "gray", "width": 1},
                "y": {"show": True, "color": "gray", "width": 1},
            }

        traces = [trace]

        if spec.show_scatter:
            x_mesh, y_mesh = np.meshgrid(spec.x_grid, spec.y_grid)
            scatter_trace: dict[str, Any] = {
                "type": "scatter3d",
                "x": x_mesh.ravel().tolist(),
                "y": y_mesh.ravel().tolist(),
                "z": np.asarray(spec.z_data).ravel().tolist(),
                "mode": "markers",
                "marker": {"size": 2, "color": "black", "opacity": 0.3},
            }
            traces.append(scatter_trace)

        layout = self._build_layout(spec)
        layout["scene"] = {
            "xaxis": self._axis_dict(spec.x_axis),
            "yaxis": self._axis_dict(spec.y_axis),
            "zaxis": self._axis_dict(spec.z_axis),
        }
        return {"data": traces, "layout": layout}

    def _contour(self, spec: ContourPlotSpec) -> dict[str, Any]:
        if not (spec is not None):
            raise ValueError("spec must be provided")
        trace_type = "contour"
        trace: dict[str, Any] = {
            "type": trace_type,
            "z": spec.z_data,
            "x": spec.x_grid,
            "y": spec.y_grid,
            "colorscale": spec.colormap,
            "ncontours": spec.levels,
        }

        if spec.filled:
            trace["contours"] = {"coloring": "heatmap"}

        if spec.show_labels:
            trace["contours"] = trace.get("contours", {})
            trace["contours"]["showlabels"] = True

        trace["showscale"] = spec.show_colorbar

        return {"data": [trace], "layout": self._build_layout(spec)}

    def _heatmap(self, spec: HeatmapSpec) -> dict[str, Any]:
        if not (spec is not None):
            raise ValueError("spec must be provided")
        trace: dict[str, Any] = {
            "type": "heatmap",
            "z": spec.z_data,
            "colorscale": spec.colormap,
            "showscale": spec.show_colorbar,
        }

        if spec.x_labels:
            trace["x"] = spec.x_labels
        if spec.y_labels:
            trace["y"] = spec.y_labels

        if spec.annotate:
            # Plotly uses text + texttemplate for annotations
            z_arr = np.asarray(spec.z_data)
            trace["text"] = [[f"{v:.2f}" for v in row] for row in z_arr]
            trace["texttemplate"] = "%{text}"
            trace["hoverinfo"] = "z"

        return {"data": [trace], "layout": self._build_layout(spec)}

    def _histogram(self, spec: HistogramSpec) -> dict[str, Any]:
        if not (spec is not None):
            raise ValueError("spec must be provided")
        traces = []
        for series in spec.series:
            trace: dict[str, Any] = {
                "type": "histogram",
                "x": series.y,  # Histogram plots distribution of y values
                "name": series.name,
                "nbinsx": spec.bins,
            }
            if series.style.color:
                trace["marker"] = {"color": series.style.color}
            if spec.density:
                trace["histnorm"] = "probability density"
            if spec.cumulative:
                trace["cumulative"] = {"enabled": True}
            traces.append(trace)

        layout = self._build_layout(spec)
        if spec.stacked:
            layout["barmode"] = "stack"

        return {"data": traces, "layout": layout}

    def _filter_comparison(self, spec: FilterComparisonSpec) -> dict[str, Any]:
        if not (spec is not None):
            raise ValueError("spec must be provided")
        traces = []

        for series in spec.original_series:
            trace = self._series_trace(series, name_prefix="Original: ")
            traces.append(trace)

        for series in spec.filtered_series:
            trace = self._series_trace(series, name_prefix="Filtered: ")
            # Default filtered to dashed
            if series.style.line_style == "solid":
                if "line" not in trace:
                    trace["line"] = {}
                trace["line"]["dash"] = "dash"
            traces.append(trace)

        layout = self._build_layout(spec)

        if spec.show_difference:
            # Add difference traces on a secondary y-axis
            for i in range(min(len(spec.original_series), len(spec.filtered_series))):
                orig = spec.original_series[i]
                filt = spec.filtered_series[i]
                y_orig = np.asarray(orig.y)
                y_filt = np.asarray(filt.y)
                min_len = min(len(y_orig), len(y_filt))
                diff = (y_orig[:min_len] - y_filt[:min_len]).tolist()

                diff_trace: dict[str, Any] = {
                    "type": "scatter",
                    "x": orig.x[:min_len],
                    "y": diff,
                    "name": f"Diff: {orig.name}",
                    "mode": "lines",
                    "line": {"color": spec.difference_color, "width": 1},
                    "yaxis": "y2",
                }
                traces.append(diff_trace)

            layout["yaxis2"] = {
                "title": "Difference",
                "overlaying": "y",
                "side": "right",
            }

        return {"data": traces, "layout": layout}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _series_trace(
        self,
        series: SeriesData,
        name_prefix: str = "",
    ) -> dict[str, Any]:
        """Convert a SeriesData to a Plotly trace."""
        if not (series is not None):
            raise ValueError("series must be provided")
        style = series.style
        mode = self._display_mode_to_plotly(style.display_mode)

        trace: dict[str, Any] = {
            "type": "scatter",
            "x": series.x,
            "y": series.y,
            "name": f"{name_prefix}{series.name}",
            "mode": mode,
        }

        # Line styling
        if "lines" in mode:
            trace["line"] = {
                "width": style.line_width,
                "dash": _DASH_MAP.get(style.line_style, "solid"),
            }
            if style.color:
                trace["line"]["color"] = style.color

        # Marker styling
        if "markers" in mode:
            marker: dict[str, Any] = {"size": style.marker_size}
            symbol = _SYMBOL_MAP.get(style.marker)
            if symbol:
                marker["symbol"] = symbol
            if style.color:
                marker["color"] = style.color
            trace["marker"] = marker

        if style.opacity < 1.0:
            trace["opacity"] = style.opacity

        return trace

    def _trendline_trace(self, series: SeriesData) -> dict[str, Any] | None:
        """Compute trendline and return as a trace."""
        if not (series is not None):
            raise ValueError("series must be provided")
        if series.trendline is None:
            return None

        tspec = series.trendline
        try:
            result = compute_trendline(
                np.asarray(series.x),
                np.asarray(series.y),
                trend_type=tspec.type,
                degree=tspec.degree,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Trendline failed for {series.name}: {e}")
            return None

        name_parts = []
        if tspec.show_equation:
            name_parts.append(result.equation)
        if tspec.show_r_squared:
            name_parts.append(f"R\u00b2={result.r_squared:.4f}")

        trace: dict[str, Any] = {
            "type": "scatter",
            "x": result.x_pred.tolist(),
            "y": result.y_pred.tolist(),
            "name": " | ".join(name_parts) if name_parts else f"Trend: {series.name}",
            "mode": "lines",
            "line": {
                "dash": _DASH_MAP.get(tspec.line_style, "dash"),
                "width": 1.5,
            },
        }
        if tspec.color:
            trace["line"]["color"] = tspec.color

        return trace

    def _build_layout(self, spec: PlotSpec) -> dict[str, Any]:
        """Build Plotly layout dict from spec."""
        if not (spec is not None):
            raise ValueError("spec must be provided")
        layout: dict[str, Any] = {
            "title": {"text": spec.title} if spec.title else None,
            "width": spec.width,
            "height": spec.height,
            "xaxis": self._axis_dict(spec.x_axis),
            "yaxis": self._axis_dict(spec.y_axis),
        }

        # Legend
        if not spec.legend.visible or spec.legend.position == "none":
            layout["showlegend"] = False
        else:
            layout["showlegend"] = True
            layout["legend"] = self._legend_dict(spec.legend)

        # Remove None values
        return {k: v for k, v in layout.items() if v is not None}

    @staticmethod
    def _axis_dict(axis: Any) -> dict[str, Any]:
        """Convert AxisSpec to Plotly axis dict."""
        d: dict[str, Any] = {}
        if axis.label:
            d["title"] = {"text": axis.label}
        if axis.min is not None:
            d.setdefault("range", [None, None])[0] = axis.min
        if axis.max is not None:
            d.setdefault("range", [None, None])[1] = axis.max
        if axis.log_scale:
            d["type"] = "log"
        d["showgrid"] = axis.grid
        return d

    @staticmethod
    def _legend_dict(legend: Any) -> dict[str, Any]:
        """Convert LegendSpec to Plotly legend dict."""
        pos_map = {
            "right": {"x": 1.02, "y": 1, "xanchor": "left"},
            "left": {"x": -0.15, "y": 1, "xanchor": "right"},
            "top": {"x": 0.5, "y": 1.1, "xanchor": "center", "orientation": "h"},
            "bottom": {"x": 0.5, "y": -0.15, "xanchor": "center", "orientation": "h"},
        }
        return pos_map.get(legend.position, {})

    @staticmethod
    def _display_mode_to_plotly(mode: str) -> str:
        """Convert display mode to Plotly mode string."""
        return {
            "line": "lines",
            "scatter": "markers",
            "line+scatter": "lines+markers",
        }.get(mode, "lines")
