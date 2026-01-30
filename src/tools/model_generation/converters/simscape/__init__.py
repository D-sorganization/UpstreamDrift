"""
SimScape Multibody conversion tools.

This module provides conversion between MATLAB SimScape Multibody
models and URDF format.
"""

from model_generation.converters.simscape.mdl_parser import (
    MDLParser,
    SimscapeBlock,
    SimscapeBlockType,
    SimscapeConnection,
    SimscapeModel,
    SimscapeParameter,
)
from model_generation.converters.simscape.simscape_converter import (
    ConversionConfig,
    ConversionResult,
    SimscapeToURDFConverter,
    convert_simscape_to_urdf,
)

__all__ = [
    # Parser
    "MDLParser",
    "SimscapeModel",
    "SimscapeBlock",
    "SimscapeBlockType",
    "SimscapeParameter",
    "SimscapeConnection",
    # Converter
    "SimscapeToURDFConverter",
    "ConversionConfig",
    "ConversionResult",
    "convert_simscape_to_urdf",
]
