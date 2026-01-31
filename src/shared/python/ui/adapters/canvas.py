"""Canvas Adapter for Framework-Agnostic Rendering.

This module provides an abstraction layer over matplotlib canvases,
allowing code to work with or without Qt installed.

Classes:
    CanvasAdapter: Abstract base for canvas implementations
    QtCanvas: PyQt6/PySide6 implementation
    HeadlessCanvas: Non-GUI implementation for testing/CI

Usage:
    canvas = get_canvas_adapter(width=800, height=600)
    fig = canvas.get_figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 4, 9])
    canvas.refresh()
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
from matplotlib.figure import Figure

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class CanvasProtocol(Protocol):
    """Protocol defining required canvas interface."""

    def get_figure(self) -> Figure:
        """Get the matplotlib Figure."""
        ...

    def refresh(self) -> None:
        """Refresh the canvas display."""
        ...

    def save(self, path: str, **kwargs: Any) -> None:
        """Save canvas to file."""
        ...


class CanvasAdapter(ABC):
    """Abstract base class for canvas adapters.

    Provides a common interface for matplotlib canvases that can be
    implemented by Qt widgets, headless backends, or web components.
    """

    def __init__(self, width: float = 8.0, height: float = 6.0, dpi: int = 100) -> None:
        """Initialize canvas with dimensions.

        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self._figure: Figure | None = None

    @abstractmethod
    def get_figure(self) -> Figure:
        """Get the matplotlib Figure.

        Returns:
            The Figure object for plotting
        """

    @abstractmethod
    def refresh(self) -> None:
        """Refresh the canvas to display updates."""

    def save(self, path: str, **kwargs: Any) -> None:
        """Save the figure to a file.

        Args:
            path: Output file path
            **kwargs: Additional arguments for savefig
        """
        fig = self.get_figure()
        fig.savefig(path, dpi=kwargs.pop("dpi", self.dpi), **kwargs)

    def clear(self) -> None:
        """Clear the figure."""
        fig = self.get_figure()
        fig.clear()

    def add_subplot(self, *args: Any, **kwargs: Any) -> Any:
        """Add a subplot to the figure.

        Args:
            *args: Positional arguments for add_subplot
            **kwargs: Keyword arguments for add_subplot

        Returns:
            The created Axes object
        """
        return self.get_figure().add_subplot(*args, **kwargs)


class HeadlessCanvas(CanvasAdapter):
    """Headless canvas implementation for non-GUI environments.

    Uses matplotlib's Agg backend for rendering without display.
    Suitable for:
    - CI/CD environments
    - Server-side rendering
    - Unit testing
    """

    def __init__(self, width: float = 8.0, height: float = 6.0, dpi: int = 100) -> None:
        """Initialize headless canvas.

        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        super().__init__(width, height, dpi)

        # Use Agg backend for headless rendering
        import matplotlib

        matplotlib.use("Agg")

        self._figure = Figure(figsize=(width, height), dpi=dpi)

    def get_figure(self) -> Figure:
        """Get the matplotlib Figure."""
        if self._figure is None:
            self._figure = Figure(figsize=(self.width, self.height), dpi=self.dpi)
        return self._figure

    def refresh(self) -> None:
        """Refresh is a no-op for headless canvas."""
        # No display to refresh

    def to_array(self) -> np.ndarray:
        """Render figure to numpy array.

        Useful for testing and image processing.

        Returns:
            RGB array of shape (height, width, 3)
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(self._figure)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3]


class QtCanvas(CanvasAdapter):
    """PyQt6/PySide6 canvas implementation.

    Wraps FigureCanvasQTAgg for Qt integration.
    """

    def __init__(self, width: float = 8.0, height: float = 6.0, dpi: int = 100) -> None:
        """Initialize Qt canvas.

        Args:
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch

        Raises:
            RuntimeError: If Qt is not available
        """
        super().__init__(width, height, dpi)

        try:
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            except ImportError:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

            self._figure = Figure(figsize=(width, height), dpi=dpi)
            self._qt_canvas = FigureCanvasQTAgg(self._figure)

        except ImportError as e:
            raise RuntimeError(
                "Qt backend not available. Install PyQt6 or use HeadlessCanvas."
            ) from e

    def get_figure(self) -> Figure:
        """Get the matplotlib Figure."""
        return self._figure

    def refresh(self) -> None:
        """Refresh the Qt canvas display."""
        self._qt_canvas.draw()
        self._qt_canvas.flush_events()

    def get_widget(self) -> Any:
        """Get the underlying Qt widget.

        Returns:
            FigureCanvasQTAgg widget for embedding in Qt layouts
        """
        return self._qt_canvas


def is_headless() -> bool:
    """Check if running in headless environment.

    Returns:
        True if no display is available
    """
    # Check environment variable
    if os.environ.get("HEADLESS", "").lower() == "true":
        return True

    # Check for display on Linux
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        return True

    return False


def is_qt_available() -> bool:
    """Check if Qt is available.

    Returns:
        True if PyQt6 or PySide6 can be imported
    """
    try:
        try:
            from matplotlib.backends.backend_qtagg import (  # noqa: F401
                FigureCanvasQTAgg,
            )

            return True
        except ImportError:
            from matplotlib.backends.backend_qt5agg import (  # noqa: F401
                FigureCanvasQTAgg,
            )

            return True
    except ImportError:
        return False


def get_canvas_adapter(
    width: float = 8.0,
    height: float = 6.0,
    dpi: int = 100,
    force_headless: bool = False,
) -> CanvasAdapter:
    """Get appropriate canvas adapter for the current environment.

    Automatically selects between Qt and headless implementations
    based on environment detection.

    Args:
        width: Figure width in inches
        height: Figure height in inches
        dpi: Dots per inch
        force_headless: Force headless mode regardless of environment

    Returns:
        Appropriate CanvasAdapter implementation
    """
    if force_headless or is_headless() or not is_qt_available():
        logger.debug("Using HeadlessCanvas")
        return HeadlessCanvas(width, height, dpi)
    else:
        logger.debug("Using QtCanvas")
        return QtCanvas(width, height, dpi)


__all__ = [
    "CanvasAdapter",
    "CanvasProtocol",
    "HeadlessCanvas",
    "QtCanvas",
    "get_canvas_adapter",
    "is_headless",
    "is_qt_available",
]
