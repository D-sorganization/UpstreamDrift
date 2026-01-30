"""
Model builders for URDF generation.

This module provides builder classes for constructing URDF models:
- ManualBuilder: Segment-by-segment construction with validation
- ParametricBuilder: Parameter-driven generation (humanoid, etc.)
- CompositeBuilder: Combine multiple sources/models
"""

from model_generation.builders.base_builder import (
    BaseURDFBuilder,
    BuildResult,
)
from model_generation.builders.manual_builder import ManualBuilder
from model_generation.builders.parametric_builder import ParametricBuilder
from model_generation.builders.urdf_writer import URDFWriter

__all__ = [
    "BaseURDFBuilder",
    "BuildResult",
    "ManualBuilder",
    "ParametricBuilder",
    "URDFWriter",
]
