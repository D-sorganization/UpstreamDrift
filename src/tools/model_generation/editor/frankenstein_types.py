"""
Shared types for the Frankenstein Editor.

Dataclasses and enums used across clipboard, history, and transform modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from model_generation.converters.urdf_parser import ParsedModel
from model_generation.core.types import Joint, Link, Material


class ComponentType(Enum):
    """Types of components that can be copied."""

    LINK = "link"
    SUBTREE = "subtree"
    JOINT = "joint"
    MATERIAL = "material"


@dataclass
class ComponentReference:
    """Reference to a component in a model."""

    model_id: str
    component_type: ComponentType
    component_name: str
    # For subtree: the root link name
    subtree_root: str | None = None

    def __str__(self) -> str:
        if self.component_type == ComponentType.SUBTREE:
            return f"{self.model_id}:{self.subtree_root}/*"
        return f"{self.model_id}:{self.component_name}"


@dataclass
class PendingOperation:
    """A pending copy/paste operation."""

    operation_type: str  # 'copy_link', 'copy_subtree', 'attach', 'rename', 'delete'
    source_ref: ComponentReference | None
    target_model_id: str | None
    parameters: dict[str, Any] = field(default_factory=dict)
    preview_links: list[Link] = field(default_factory=list)
    preview_joints: list[Joint] = field(default_factory=list)


@dataclass
class EditorState:
    """State of the Frankenstein editor for undo/redo."""

    models: dict[str, ParsedModel]
    clipboard: list[tuple[ComponentType, list[Link], list[Joint], dict[str, Material]]]
    operation_history: list[PendingOperation]
    timestamp: float = 0.0
