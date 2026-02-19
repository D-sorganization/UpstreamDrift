"""
URDF Editing tools.

This module provides editors for URDF manipulation:
- FrankensteinEditor: Compose models from multiple sources
- URDFTextEditor: Text-based editing with diff support
"""

from model_generation.editor.frankenstein_editor import (
    ComponentReference,
    ComponentType,
    FrankensteinEditor,
)
from model_generation.editor.text_editor import (
    DiffHunk,
    DiffResult,
    URDFTextEditor,
    ValidationMessage,
    ValidationSeverity,
)

__all__ = [
    # Frankenstein Editor
    "FrankensteinEditor",
    "ComponentType",
    "ComponentReference",
    # Text Editor
    "URDFTextEditor",
    "ValidationMessage",
    "ValidationSeverity",
    "DiffResult",
    "DiffHunk",
]
