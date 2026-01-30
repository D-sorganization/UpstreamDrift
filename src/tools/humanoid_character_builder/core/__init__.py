"""
Core module for humanoid character builder.

Contains fundamental data structures and definitions for body parameters,
segment hierarchies, and anthropometric data.
"""

from humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    SegmentParameters,
    AppearanceParameters,
)
from humanoid_character_builder.core.segment_definitions import (
    SegmentDefinition,
    JointDefinition,
    JointType,
    GeometryType,
    HUMANOID_SEGMENTS,
    HUMANOID_JOINTS,
)
from humanoid_character_builder.core.anthropometry import (
    AnthropometryData,
    get_segment_mass_ratio,
    get_segment_length_ratio,
    estimate_segment_masses,
    estimate_segment_dimensions,
)

__all__ = [
    "BodyParameters",
    "SegmentParameters",
    "AppearanceParameters",
    "SegmentDefinition",
    "JointDefinition",
    "JointType",
    "GeometryType",
    "HUMANOID_SEGMENTS",
    "HUMANOID_JOINTS",
    "AnthropometryData",
    "get_segment_mass_ratio",
    "get_segment_length_ratio",
    "estimate_segment_masses",
    "estimate_segment_dimensions",
]
