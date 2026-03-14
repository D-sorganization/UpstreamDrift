"""Matplotlib renderer for PlotSpec contracts.

Renders PlotSpec objects into matplotlib Figure instances for use
in PyQt6 applications. Integrates with PlotThemeManager for
consistent styling across the fleet.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .protocols import ThemeColorProvider
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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Marker mapping from spec names to matplotlib markers
_MARKER_MAP = {
    "none": "",
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "cross": "x",
    "plus": "+",
    "star": "*",
}

# Line style mapping
_LINESTYLE_MAP = {
    "solid": "-",
    "dashed": "--",
    "dotted": ":",
    "dashdot": "-.",
}


class MatplotlibRenderer:
    """Renders PlotSpec contracts into matplotlib Figures."""

    def __init__(self, theme_manager: ThemeColorProvider | None = None) -> None:
        self._theme_manager = theme_manager

    def render(
        self,
        spec: PlotSpec,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Render a PlotSpec to a matplotlib Figure.

        Dispatches to the appropriate type-specific renderer.
        """
        assert spec is not None, "spec must be provided"
        if isinstance(spec, SurfacePlotSpec):
            return self.render_surface(spec, fig)
        if isinstance(spec, ContourPlotSpec):
            return self.render_contour(spec, fig, ax)
        if isinstance(spec, HeatmapSpec):
            return self.render_heatmap(spec, fig, ax)
        if isinstance(spec, HistogramSpec):
            return self.render_histogram(spec, fig, ax)
        if isinstance(spec, FilterComparisonSpec):
            return self.render_filter_comparison(spec, fig)
        return self.render_line_plot(spec, fig, ax)

    def render_line_plot(
        self,
        spec: PlotSpec,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Render a line/scatter plot."""
        assert spec is not None, "spec must be provided"
        fig, ax = self._ensure_fig_ax(fig, ax, spec)

        colors = self._get_theme_colors()
        for i, series in enumerate(spec.series):
            color = series.style.color or self._cycle_color(colors, i)
            self._plot_series(ax, series, color)

            if series.trendline is not None:
                self._render_trendline(ax, series, color)

        self._apply_axis_spec(ax, spec)
        self._apply_legend(ax, spec)

        if self._theme_manager:
            self._theme_manager.apply_to_figure(fig)

        fig.tight_layout()
        return fig

    def render_surface(
        self,
        spec: SurfacePlotSpec,
        fig: Figure | None = None,
    ) -> Figure:
        """Render a 3D surface plot."""
        assert spec is not None, "spec must be provided"
        if fig is None:
            fig = plt.figure(
                figsize=(spec.width / 100, spec.height / 100),
            )
        ax = fig.add_subplot(111, projection="3d")

        x_grid = np.asarray(spec.x_grid)
        y_grid = np.asarray(spec.y_grid)
        z_data = np.asarray(spec.z_data)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        cmap = spec.colormap
        if self._theme_manager:
            theme_colors = self._theme_manager.get_colors()
            cmap = theme_colors.get("contour_cmap", cmap)

        ax.plot_surface(  # type: ignore[attr-defined]
            x_mesh,
            y_mesh,
            z_data,
            cmap=cmap,
            alpha=spec.opacity,
            edgecolor="none" if not spec.show_wireframe else "gray",
            linewidth=0.3 if spec.show_wireframe else 0,
        )

        if spec.show_scatter:
            ax.scatter(  # type: ignore[misc]
                x_mesh.ravel(),
                y_mesh.ravel(),
                z_data.ravel(),
                c="k",
                s=2,
                alpha=0.3,
            )

        ax.set_xlabel(spec.x_axis.label or "X")
        ax.set_ylabel(spec.y_axis.label or "Y")
        ax.set_zlabel(spec.z_axis.label or "Z")  # type: ignore[attr-defined]

        if spec.title:
            ax.set_title(spec.title)

        fig.tight_layout()
        return fig

    def render_contour(
        self,
        spec: ContourPlotSpec,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Render a contour plot."""
        assert spec is not None, "spec must be provided"
        fig, ax = self._ensure_fig_ax(fig, ax, spec)

        x_grid = np.asarray(spec.x_grid)
        y_grid = np.asarray(spec.y_grid)
        z_data = np.asarray(spec.z_data)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        cmap = spec.colormap
        if self._theme_manager:
            theme_colors = self._theme_manager.get_colors()
            cmap = theme_colors.get("contour_cmap", cmap)

        if spec.filled:
            cs = ax.contourf(x_mesh, y_mesh, z_data, levels=spec.levels, cmap=cmap)
        else:
            cs = ax.contour(x_mesh, y_mesh, z_data, levels=spec.levels, cmap=cmap)

        if spec.show_labels and not spec.filled:
            ax.clabel(cs, inline=True, fontsize=8)

        if spec.show_colorbar:
            fig.colorbar(cs, ax=ax)

        self._apply_axis_spec(ax, spec)
        fig.tight_layout()
        return fig

    def render_heatmap(
        self,
        spec: HeatmapSpec,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Render a heatmap."""
        assert spec is not None, "spec must be provided"
        fig, ax = self._ensure_fig_ax(fig, ax, spec)

        z_data = np.asarray(spec.z_data)

        cmap = spec.colormap
        if self._theme_manager:
            theme_colors = self._theme_manager.get_colors()
            cmap = theme_colors.get("heatmap_cmap", cmap)

        im = ax.imshow(z_data, cmap=cmap, aspect="auto", origin="lower")

        if spec.x_labels:
            ax.set_xticks(range(len(spec.x_labels)))
            ax.set_xticklabels(spec.x_labels, rotation=45, ha="right")
        if spec.y_labels:
            ax.set_yticks(range(len(spec.y_labels)))
            ax.set_yticklabels(spec.y_labels)

        if spec.annotate:
            for i in range(z_data.shape[0]):
                for j in range(z_data.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{z_data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        if spec.show_colorbar:
            fig.colorbar(im, ax=ax)

        if spec.title:
            ax.set_title(spec.title)

        fig.tight_layout()
        return fig

    def render_histogram(
        self,
        spec: HistogramSpec,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> Figure:
        """Render a histogram."""
        assert spec is not None, "spec must be provided"
        fig, ax = self._ensure_fig_ax(fig, ax, spec)

        colors = self._get_theme_colors()
        data_arrays = []
        labels = []

        for series in spec.series:
            data_arrays.append(series.y)
            labels.append(series.name)

        if data_arrays:
            hist_colors = [
                spec.series[i].style.color or self._cycle_color(colors, i)
                for i in range(len(data_arrays))
            ]
            ax.hist(
                data_arrays,
                bins=spec.bins,
                density=spec.density,
                cumulative=spec.cumulative,
                stacked=spec.stacked,
                label=labels,
                color=hist_colors,
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
            )

        self._apply_axis_spec(ax, spec)
        self._apply_legend(ax, spec)
        fig.tight_layout()
        return fig

    def render_filter_comparison(
        self,
        spec: FilterComparisonSpec,
        fig: Figure | None = None,
    ) -> Figure:
        """Render a filter comparison with optional difference subplot."""
        assert spec is not None, "spec must be provided"
        n_rows = 2 if spec.show_difference else 1
        height_ratios = [3, 1] if spec.show_difference else [1]

        if fig is None:
            fig = plt.figure(figsize=(spec.width / 100, spec.height / 100))

        gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.3)
        ax_main = fig.add_subplot(gs[0])

        colors = self._get_theme_colors()

        # Plot original series
        for i, series in enumerate(spec.original_series):
            color = series.style.color or self._cycle_color(colors, i)
            self._plot_series(ax_main, series, color, label_prefix="Original: ")

        # Plot filtered series (with dashed default)
        for i, series in enumerate(spec.filtered_series):
            color = series.style.color or self._cycle_color(colors, i)
            effective_style = series.style.model_copy()
            if series.style.line_style == "solid":
                effective_style.line_style = "dashed"
            self._plot_series(
                ax_main,
                series,
                color,
                label_prefix="Filtered: ",
                override_linestyle=_LINESTYLE_MAP.get(effective_style.line_style, "--"),
            )

        self._apply_axis_spec(ax_main, spec)
        self._apply_legend(ax_main, spec)

        # Difference subplot
        if spec.show_difference and n_rows > 1:
            ax_diff = fig.add_subplot(gs[1])
            for i in range(min(len(spec.original_series), len(spec.filtered_series))):
                orig = spec.original_series[i]
                filt = spec.filtered_series[i]
                # Compute difference where x values match
                x_orig = np.asarray(orig.x)
                y_orig = np.asarray(orig.y)
                y_filt = np.asarray(filt.y)
                min_len = min(len(y_orig), len(y_filt))
                diff = y_orig[:min_len] - y_filt[:min_len]
                ax_diff.plot(
                    x_orig[:min_len],
                    diff,
                    color=spec.difference_color,
                    linewidth=1.0,
                    label=f"Diff: {orig.name}",
                )
            ax_diff.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
            ax_diff.set_xlabel(spec.x_axis.label)
            ax_diff.set_ylabel("Difference")
            ax_diff.legend(fontsize=8)
            ax_diff.grid(True, alpha=0.3)

        with contextlib.suppress(ValueError):
            fig.tight_layout()
        return fig

    def to_image(
        self,
        spec: PlotSpec,
        fmt: str = "png",
        dpi: int = 150,
    ) -> bytes:
        """Render a PlotSpec to image bytes."""
        assert spec is not None, "spec must be provided"
        fig = self.render(spec)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _ensure_fig_ax(
        self,
        fig: Figure | None,
        ax: Axes | None,
        spec: PlotSpec,
    ) -> tuple[Figure, Axes]:
        """Create or reuse figure and axes."""
        assert spec is not None, "spec must be provided"
        if fig is None:
            fig, ax = plt.subplots(
                figsize=(spec.width / 100, spec.height / 100),
            )
        elif ax is None:
            ax = fig.add_subplot(111)
        return fig, ax

    def _get_theme_colors(self) -> list[str]:
        """Get color cycle from theme or default matplotlib."""
        if self._theme_manager:
            colors_dict = self._theme_manager.get_colors()
            result: list[str] = colors_dict.get("primary_colors", []) + colors_dict.get(
                "secondary_colors", []
            )
            return result
        # Default color cycle
        return [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]

    @staticmethod
    def _cycle_color(colors: list[str], index: int) -> str:
        """Get a color from the cycle by index."""
        assert colors is not None, "colors must be provided"
        if not colors:
            return "#1f77b4"
        return colors[index % len(colors)]

    def _plot_series(
        self,
        ax: Axes,
        series: SeriesData,
        color: str,
        label_prefix: str = "",
        override_linestyle: str | None = None,
    ) -> None:
        """Plot a single data series on an axes."""
        assert ax is not None, "ax must be provided"
        x = np.asarray(series.x)
        y = np.asarray(series.y)
        style = series.style
        label = f"{label_prefix}{series.name}"
        marker = _MARKER_MAP.get(style.marker, "")
        linestyle = override_linestyle or _LINESTYLE_MAP.get(style.line_style, "-")

        if style.display_mode == "line":
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=style.line_width,
                alpha=style.opacity,
                label=label,
            )
        elif style.display_mode == "scatter":
            ax.scatter(
                x,
                y,
                c=color,
                marker=marker or "o",
                s=style.marker_size**2,
                alpha=style.opacity,
                label=label,
            )
        elif style.display_mode == "line+scatter":
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=style.line_width,
                marker=marker or "o",
                markersize=style.marker_size,
                alpha=style.opacity,
                label=label,
            )

    def _render_trendline(
        self,
        ax: Axes,
        series: SeriesData,
        base_color: str,
    ) -> None:
        """Render a trendline for a series."""
        assert ax is not None, "ax must be provided"
        if series.trendline is None:
            return

        tspec = series.trendline
        try:
            result = compute_trendline(
                np.asarray(series.x),
                np.asarray(series.y),
                trend_type=tspec.type,
                degree=tspec.degree,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Trendline computation failed for {series.name}: {e}")
            return

        trend_color = tspec.color or base_color
        linestyle = _LINESTYLE_MAP.get(tspec.line_style, "--")

        ax.plot(
            result.x_pred,
            result.y_pred,
            color=trend_color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.8,
        )

        # Annotation with equation and/or R²
        parts = []
        if tspec.show_equation:
            parts.append(result.equation)
        if tspec.show_r_squared:
            parts.append(f"R\u00b2 = {result.r_squared:.4f}")

        if parts:
            annotation = "\n".join(parts)
            ax.annotate(
                annotation,
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                verticalalignment="top",
                fontsize=8,
                color=trend_color,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            )

    @staticmethod
    def _apply_axis_spec(ax: Axes, spec: PlotSpec) -> None:
        """Apply axis configuration from spec."""
        assert ax is not None, "ax must be provided"
        if spec.title:
            ax.set_title(spec.title)

        ax.set_xlabel(spec.x_axis.label)
        ax.set_ylabel(spec.y_axis.label)

        if spec.x_axis.min is not None or spec.x_axis.max is not None:
            ax.set_xlim(spec.x_axis.min, spec.x_axis.max)
        if spec.y_axis.min is not None or spec.y_axis.max is not None:
            ax.set_ylim(spec.y_axis.min, spec.y_axis.max)

        if spec.x_axis.log_scale:
            ax.set_xscale("log")
        if spec.y_axis.log_scale:
            ax.set_yscale("log")

        ax.grid(spec.x_axis.grid or spec.y_axis.grid, alpha=0.3)

    @staticmethod
    def _apply_legend(ax: Axes, spec: PlotSpec) -> None:
        """Apply legend configuration from spec."""
        assert ax is not None, "ax must be provided"
        if not spec.legend.visible or spec.legend.position == "none":
            return

        loc_map = {
            "right": "center right",
            "left": "center left",
            "top": "upper center",
            "bottom": "lower center",
        }
        loc = loc_map.get(spec.legend.position, "best")

        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        # Apply custom labels
        if spec.legend.labels:
            labels = [spec.legend.labels.get(lbl, lbl) for lbl in labels]

        ax.legend(handles, labels, loc=loc, fontsize=8)
