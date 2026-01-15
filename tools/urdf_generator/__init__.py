"""Interactive URDF Generator for Golf Modeling Suite.

This module provides an interactive GUI tool for creating URDF files
with support for parallel kinematic configurations commonly used in
golf swing modeling.
"""

__version__ = "1.0.0"
__author__ = "Golf Modeling Suite Team"

from .segment_manager import SegmentManager
from .urdf_builder import Handedness, URDFBuilder

__all__ = ["URDFGeneratorWindow", "URDFBuilder", "SegmentManager", "Handedness"]


def __getattr__(name: str):
    """Lazy import for GUI components that require PyQt6."""
    if name == "URDFGeneratorWindow":
        from .main_window import URDFGeneratorWindow

        return URDFGeneratorWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
