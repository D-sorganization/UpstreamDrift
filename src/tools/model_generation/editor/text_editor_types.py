"""
Shared types for the URDF Text Editor.

Dataclasses and enums used across validation and diff modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation messages."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationMessage:
    """A validation message for URDF content."""

    severity: ValidationSeverity
    line: int
    column: int
    message: str
    element: str | None = None

    def __str__(self) -> str:
        prefix = self.severity.value.upper()
        loc = f"Line {self.line}"
        if self.column > 0:
            loc += f", Col {self.column}"
        if self.element:
            return f"[{prefix}] {loc} ({self.element}): {self.message}"
        return f"[{prefix}] {loc}: {self.message}"


@dataclass
class DiffHunk:
    """A single hunk in a diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)


@dataclass
class DiffResult:
    """Result of comparing two URDF versions."""

    original_content: str
    modified_content: str
    hunks: list[DiffHunk]
    unified_diff: str
    additions: int
    deletions: int
    has_changes: bool

    def get_summary(self) -> str:
        """Get a summary of changes."""
        if not self.has_changes:
            return "No changes"
        return f"{self.additions} additions, {self.deletions} deletions in {len(self.hunks)} hunks"


@dataclass
class EditorVersion:
    """A version of the document."""

    content: str
    timestamp: datetime
    description: str
    checksum: str
