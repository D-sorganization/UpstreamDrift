"""
Model Explorer GUI for browsing, loading, and previewing URDF/MJCF models.

Provides a visual interface for the model library with display controls.
"""

from model_generation.explorer.display_config import DISPLAY_OPTIONS


def get_explorer_window():
    """Lazy import of ModelExplorerWindow to avoid PyQt6 dependency at import time."""
    from model_generation.explorer.model_explorer import ModelExplorerWindow

    return ModelExplorerWindow


__all__ = ["DISPLAY_OPTIONS", "get_explorer_window"]
