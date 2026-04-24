# mypy: disable-error-code="no-redef"
"""Thin facade for the legacy golf swing visualizer."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

try:
    from .golf_visualizer_app import ModernGolfVisualizerApp
    from .golf_visualizer_data import DataProcessor
    from .golf_visualizer_models import FrameData, RenderConfig
    from .golf_visualizer_renderer import OpenGLRenderer
    from .golf_visualizer_widget import ModernGolfVisualizerWidget
except ImportError:
    from golf_visualizer_app import ModernGolfVisualizerApp
    from golf_visualizer_data import DataProcessor
    from golf_visualizer_models import FrameData, RenderConfig
    from golf_visualizer_renderer import OpenGLRenderer
    from golf_visualizer_widget import ModernGolfVisualizerWidget

logger = logging.getLogger(__name__)


__all__ = [
    "FrameData",
    "RenderConfig",
    "DataProcessor",
    "OpenGLRenderer",
    "ModernGolfVisualizerWidget",
    "ModernGolfVisualizerApp",
    "main",
]


def main() -> None:
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Modern Golf Swing Visualizer")
    app.setApplicationVersion("2.0")
    window = ModernGolfVisualizerApp()
    window.show()
    try:
        window.gl_widget.load_data("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")
    except (RuntimeError, ValueError, OSError) as exc:
        logger.info("Note: Sample data not found - %s", exc)
        logger.info("Please load data using File -> Load Data")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
