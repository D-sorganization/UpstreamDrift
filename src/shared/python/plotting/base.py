"""Base plotting components for golf swing analysis.

This module contains foundational classes and protocols for the plotting system:
- MplCanvas: Matplotlib canvas for PyQt6 embedding
- RecorderInterface: Protocol defining data source interface

These are extracted from plotting.py for modularity.
"""

from __future__ import annotations

from src.shared.python.engine_core.interfaces import RecorderInterface
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Qt backend - optional for headless environments
try:
    from src.shared.python.ui.qt.plotting import MplCanvas
except ImportError:
    # Qt not available (e.g., in headless CI environments)
    class MplCanvas:  # type: ignore[no-redef]
        """Matplotlib canvas for embedding in PyQt6 (not available in headless mode)."""

        def __init__(
            self, width: float = 8, height: float = 6, dpi: int = 100
        ) -> None:  # noqa: ARG002
            """Initialize canvas with figure (implementation for headless environments)."""
            msg = (
                "MplCanvas requires Qt backend which is not available in headless envs"
            )
            raise RuntimeError(msg)


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
