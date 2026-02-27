"""Protocol interfaces for the plot engine.

These protocols define the structural typing contracts for plot renderers
and converters. Any class that implements the required methods satisfies
the protocol, enabling duck-typed plugin architectures without inheritance.

Usage:
    def render_plot(renderer: PlotRenderer, spec: PlotSpec) -> Figure:
        return renderer.render(spec)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .specs import PlotSpec


@runtime_checkable
class PlotRenderer(Protocol):
    """Protocol for rendering PlotSpec contracts to matplotlib Figures.

    Implementations: MatplotlibRenderer
    """

    def render(self, spec: PlotSpec, **kwargs: Any) -> Figure:
        """Render a PlotSpec to a matplotlib Figure."""
        ...

    def to_image(self, spec: PlotSpec, fmt: str = "png", dpi: int = 150) -> bytes:
        """Render a PlotSpec to image bytes."""
        ...


@runtime_checkable
class PlotConverter(Protocol):
    """Protocol for converting PlotSpec contracts to serialized formats.

    Implementations: PlotlyConverter
    """

    def convert(self, spec: PlotSpec) -> dict[str, Any]:
        """Convert a PlotSpec to a serialized dictionary (e.g. Plotly JSON)."""
        ...


@runtime_checkable
class ThemeColorProvider(Protocol):
    """Protocol for providing theme colors to renderers.

    Any theme manager that exposes ``get_colors()`` and
    ``apply_to_figure()`` can be used as a color provider.
    """

    def get_colors(self) -> dict[str, Any]:
        """Return a dict of color values (primary_colors, secondary_colors, etc.)."""
        ...

    def apply_to_figure(self, fig: Figure) -> None:
        """Apply theme styling to an existing matplotlib Figure."""
        ...
