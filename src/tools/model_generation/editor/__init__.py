"""
URDF Editing tools.

This module provides editors for URDF manipulation:
- FrankensteinEditor: Compose models from multiple sources
- URDFTextEditor: Text-based editing with diff support
"""

from model_generation.editor.frankenstein_editor import (
    FrankensteinEditor,
)
from model_generation.editor.frankenstein_types import (
    ComponentReference,
    ComponentType,
)
from model_generation.editor.text_editor import (
    URDFTextEditor,
)
from model_generation.editor.text_editor_types import (
    DiffHunk,
    DiffResult,
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
