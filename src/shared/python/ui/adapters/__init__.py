"""UI Adapters Package.

This package provides abstraction layers for UI frameworks, allowing
core simulation code to remain framework-agnostic.

The adapters implement common interfaces that can be backed by:
- PyQt6/PySide6 for desktop applications
- Mock implementations for headless/testing environments
- Web backends for browser-based interfaces

Usage:
    from src.shared.python.ui.adapters import get_canvas_adapter

    # Get appropriate canvas based on environment
    canvas = get_canvas_adapter(width=800, height=600)
    canvas.draw_line(...)

This pattern isolates Qt dependencies from core simulation code,
enabling headless operation and easier testing.
"""

from src.shared.python.ui.adapters.canvas import (
    CanvasAdapter,
    HeadlessCanvas,
    get_canvas_adapter,
)
from src.shared.python.ui.adapters.thread import (
    BackgroundWorker,
    get_worker_adapter,
)

__all__ = [
    "CanvasAdapter",
    "get_canvas_adapter",
    "HeadlessCanvas",
    "BackgroundWorker",
    "get_worker_adapter",
]
