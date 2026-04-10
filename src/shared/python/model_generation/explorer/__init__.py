"""
Model Explorer GUI for browsing, loading, and previewing URDF/MJCF models.

Provides a visual interface for the model library with display controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_generation.explorer.display_config import DISPLAY_OPTIONS

if TYPE_CHECKING:
    from model_generation.explorer.model_explorer import ModelExplorerWindow


def get_explorer_window() -> type[ModelExplorerWindow]:
    """Lazy import of ModelExplorerWindow to avoid PyQt6 dependency at import time."""
    from model_generation.explorer.model_explorer import ModelExplorerWindow

    return ModelExplorerWindow


__all__ = ["DISPLAY_OPTIONS", "get_explorer_window"]
