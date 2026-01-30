"""
Public API for model generation.

This module provides the high-level user-facing API for model generation.
"""

# Re-export main API components from package root
from model_generation import (
    BuildResult,
    ManualBuilder,
    ParametricBuilder,
    quick_build,
    quick_urdf,
)

__all__ = [
    "ManualBuilder",
    "ParametricBuilder",
    "BuildResult",
    "quick_urdf",
    "quick_build",
]
