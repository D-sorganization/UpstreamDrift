"""
Format converters for model generation.

This module provides bidirectional conversion between:
- URDF (Universal Robot Description Format)
- MJCF (MuJoCo XML format)
- SDF (Simulation Description Format)
- SimScape MDL (MATLAB Simscape, import only)
"""

from model_generation.converters.urdf_parser import URDFParser, ParsedModel
from model_generation.converters.mjcf_converter import MJCFConverter
from model_generation.converters.format_utils import (
    convert_urdf_to_mjcf,
    convert_mjcf_to_urdf,
    detect_format,
    ModelFormat,
)

__all__ = [
    "URDFParser",
    "ParsedModel",
    "MJCFConverter",
    "convert_urdf_to_mjcf",
    "convert_mjcf_to_urdf",
    "detect_format",
    "ModelFormat",
]
