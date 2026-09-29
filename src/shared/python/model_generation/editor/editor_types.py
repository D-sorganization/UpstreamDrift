"""Data types for the Frankenstein Editor.

Shared enums, dataclasses, and state representations used across
editor submodules (clipboard, modifications, main editor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.shared.python.model_generation.converters.urdf_parser import ParsedModel
from src.shared.python.model_generation.core.types import (
    Joint,
    JointType,
    Link,
    Origin,
)


class ComponentType(Enum):
    """Types of components that can be copied."""

    LINK = "link"
    SUBTREE = "subtree"
    JOINT = "joint"
    MATERIAL = "material"


@dataclass(frozen=True)
class AttachmentSpec:
    """How a pasted subtree is joined to its target link.

    The three fields always travel together: they are meaningless unless
    ``attach_to`` names a link, and they are consumed at a single point in
    :meth:`FrankensteinEditor._paste_links_and_joints`. Bundling them keeps
    that method inside the 8-parameter architecture budget (#9617) and gives
    the "paste at the root" case a name -- :meth:`detached` -- rather than
    three separate defaults spelled out at the call site.
    """

    attach_to: str | None = None
    origin: Origin | None = None
    joint_type: JointType = JointType.FIXED

    @classmethod
    def detached(cls) -> AttachmentSpec:
        """Paste without joining to an existing link."""
        return cls()

    @property
    def is_attached(self) -> bool:
        """Whether a joint to an existing link should be created."""
        return bool(self.attach_to)


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
    clipboard: list[ComponentReference]
    operation_history: list[PendingOperation]
    timestamp: float = 0.0


__all__ = [
    "AttachmentSpec",
    "ComponentType",
    "ComponentReference",
    "PendingOperation",
    "EditorState",
]
