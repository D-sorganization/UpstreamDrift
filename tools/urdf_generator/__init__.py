"""Interactive URDF Generator for Golf Modeling Suite.

This module provides an interactive GUI tool for creating URDF files
with support for parallel kinematic configurations commonly used in
golf swing modeling.
"""

__version__ = "1.0.0"
__author__ = "Golf Modeling Suite Team"

from .main_window import URDFGeneratorWindow
from .segment_manager import SegmentManager
from .urdf_builder import URDFBuilder

__all__ = ["URDFGeneratorWindow", "URDFBuilder", "SegmentManager"]
