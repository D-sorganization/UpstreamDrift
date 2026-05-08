"""
Public interfaces module for humanoid character builder.

Provides the clean, user-facing API for character building.
"""

from humanoid_character_builder.interfaces.api import CharacterBuilder
from humanoid_character_builder.interfaces.options import ExportOptions
from humanoid_character_builder.interfaces.quick import quick_build, quick_urdf
from humanoid_character_builder.interfaces.results import (
    BuildErrorCategory,
    CharacterBuildResult,
    SegmentMeshInfo,
)

__all__ = [
    "CharacterBuilder",
    "CharacterBuildResult",
    "SegmentMeshInfo",
    "ExportOptions",
    "BuildErrorCategory",
    "quick_build",
    "quick_urdf",
]
