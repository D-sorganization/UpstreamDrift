"""Base plotting components for golf swing analysis.

This module contains foundational classes and protocols for the plotting system:
- MplCanvas: Matplotlib canvas for PyQt6 embedding
- RecorderInterface: Protocol defining data source interface

These are extracted from plotting.py for modularity.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Qt backend - optional for headless environments
try:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    class MplCanvas(FigureCanvasQTAgg):
        """Matplotlib canvas for embedding in PyQt6."""

        def __init__(self, width: float = 8, height: float = 6, dpi: int = 100) -> None:
            """Initialize canvas with figure.

            Args:
                width: Figure width in inches
                height: Figure height in inches
                dpi: Dots per inch for rendering
            """
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            super().__init__(self.fig)

except ImportError:
    # Qt not available (e.g., in headless CI environments)
    class MplCanvas:  # type: ignore[no-redef]
        """Matplotlib canvas for embedding in PyQt6 (not available in headless mode)."""

        def __init__(self, width: float = 8, height: float = 6, dpi: int = 100) -> None:  # noqa: ARG002
            """Initialize canvas with figure (implementation for headless environments)."""
            msg = (
                "MplCanvas requires Qt backend which is not available in headless envs"
            )
            raise RuntimeError(msg)


class RecorderInterface(Protocol):
    """Protocol for a recorder that provides time series data.

    This protocol defines the interface that data sources must implement
    to work with the golf swing plotting system.
    """

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field.

        Args:
            field_name: Name of the field

        Returns:
            Tuple of (times, values)
        """
        ...  # pragma: no cover

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific induced acceleration source.

        Args:
            source_name: Name or index of the force source (e.g. 'gravity', 0)

        Returns:
            Tuple of (times, acceleration_array)
        """
        ...  # pragma: no cover

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract time series for a specific counterfactual component.

        Args:
            cf_name: Name of the counterfactual (e.g. 'ztcf', 'zvcf')

        Returns:
            Tuple of (times, data_array)
        """
        ...  # pragma: no cover


# Import color scheme from unified theme system
# This ensures consistent colors across all visualizations
try:
    from shared.python.theme.colors import CHART_COLORS, DEFAULT_COLORS
except ImportError:
    # Fallback for when theme module is not available
    DEFAULT_COLORS = {
        "primary": "#0A84FF",
        "secondary": "#FF9F0A",
        "tertiary": "#30D158",
        "quaternary": "#FF375F",
        "quinary": "#BF5AF2",
        "senary": "#AC8E68",
        "accent": "#64D2FF",
        "dark": "#A0A0A0",
        "grid": "#333333",
    }
    CHART_COLORS = [
        "#0A84FF",
        "#30D158",
        "#FF9F0A",
        "#FF375F",
        "#BF5AF2",
        "#64D2FF",
        "#FFD60A",
        "#AC8E68",
    ]

__all__ = [
    "CHART_COLORS",
    "DEFAULT_COLORS",
    "MplCanvas",
    "RecorderInterface",
    "logger",
]
