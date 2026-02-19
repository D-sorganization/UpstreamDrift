"""
URDF Text Editor with diff view support.

Provides text-based editing of URDF files with:
- Syntax validation
- Diff generation between versions
- Undo/redo support
- Real-time validation feedback

Decomposed via SRP into:
- text_editor_types.py: Shared data types
- text_editor_validation.py: ValidationMixin (XML/URDF validation)
- text_editor_diff.py: DiffMixin (diff computation)
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DefusedET

from .text_editor_diff import DiffMixin
from .text_editor_types import (
    DiffHunk,
    DiffResult,
    EditorVersion,
    ValidationMessage,
    ValidationSeverity,
)
from .text_editor_validation import ValidationMixin

logger = logging.getLogger(__name__)

# Re-export types for backwards compatibility
__all__ = [
    "DiffHunk",
    "DiffResult",
    "URDFTextEditor",
    "ValidationMessage",
    "ValidationSeverity",
]


class URDFTextEditor(ValidationMixin, DiffMixin):
    """
    Text editor for URDF files with validation and diff support.

    Features:
    - Real-time XML validation
    - URDF-specific validation (links, joints, references)
    - Diff generation between versions
    - Undo/redo with version history
    - Syntax highlighting hints

    Example:
        editor = URDFTextEditor()
        editor.load_file("/path/to/robot.urdf")

        # Make changes
        editor.set_content(modified_urdf)

        # Validate
        messages = editor.validate()
        for msg in messages:
            print(msg)

        # Get diff
        diff = editor.get_diff_from_original()
        print(diff.unified_diff)

        # Save
        editor.save_file()
    """

    def __init__(self, max_history: int = 100) -> None:
        """
        Initialize the editor.

        Args:
            max_history: Maximum number of undo states to keep
        """
        self._content: str = ""
        self._original_content: str = ""
        self._file_path: Path | None = None
        self._history: list[EditorVersion] = []
        self._history_index: int = -1
        self._max_history = max_history
        self._validation_callbacks: list[Callable[[list[ValidationMessage]], None]] = []
        self._change_callbacks: list[Callable[[str], None]] = []

    # ============================================================
    # File Operations
    # ============================================================

    def load_file(self, path: str | Path) -> str:
        """
        Load a URDF file.

        Args:
            path: Path to URDF file

        Returns:
            File content
        """
        self._file_path = Path(path)
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")

        self._content = self._file_path.read_text()
        self._original_content = self._content

        # Clear history and add initial version
        self._history = []
        self._add_to_history("Loaded file")
        self._history_index = 0

        logger.info(f"Loaded URDF from {self._file_path}")
        return self._content

    def load_string(self, content: str, description: str = "Loaded content") -> None:
        """
        Load URDF from a string.

        Args:
            content: URDF XML content
            description: Description for history
        """
        self._content = content
        self._original_content = content
        self._file_path = None

        self._history = []
        self._add_to_history(description)
        self._history_index = 0

    def save_file(self, path: str | Path | None = None) -> Path:
        """
        Save content to file.

        Args:
            path: Optional path (uses original if not specified)

        Returns:
            Path to saved file
        """
        if path:
            self._file_path = Path(path)
        elif not self._file_path:
            raise ValueError("No file path specified")

        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(self._content)

        # Update original to track new baseline
        self._original_content = self._content

        logger.info(f"Saved URDF to {self._file_path}")
        return self._file_path

    def export_string(self) -> str:
        """Get current content as string."""
        return self._content

    # ============================================================
    # Content Editing
    # ============================================================

    def get_content(self) -> str:
        """Get current content."""
        return self._content

    def set_content(
        self,
        content: str,
        description: str = "Edit",
        validate: bool = True,
    ) -> list[ValidationMessage]:
        """
        Set content (with automatic history tracking).

        Args:
            content: New content
            description: Description for history
            validate: If True, run validation

        Returns:
            List of validation messages
        """
        if content == self._content:
            return []

        self._content = content
        self._add_to_history(description)

        # Notify change callbacks
        for change_callback in self._change_callbacks:
            change_callback(content)

        messages: list[ValidationMessage] = []
        if validate:
            messages = self.validate()
            for validation_callback in self._validation_callbacks:
                validation_callback(messages)

        return messages

    def insert_text(
        self,
        position: int,
        text: str,
        description: str = "Insert",
    ) -> list[ValidationMessage]:
        """
        Insert text at position.

        Args:
            position: Character position
            text: Text to insert
            description: Description for history

        Returns:
            Validation messages
        """
        new_content = self._content[:position] + text + self._content[position:]
        return self.set_content(new_content, description)

    def delete_text(
        self,
        start: int,
        end: int,
        description: str = "Delete",
    ) -> list[ValidationMessage]:
        """
        Delete text range.

        Args:
            start: Start position
            end: End position
            description: Description for history

        Returns:
            Validation messages
        """
        new_content = self._content[:start] + self._content[end:]
        return self.set_content(new_content, description)

    def replace_text(
        self,
        start: int,
        end: int,
        text: str,
        description: str = "Replace",
    ) -> list[ValidationMessage]:
        """
        Replace text range.

        Args:
            start: Start position
            end: End position
            text: Replacement text
            description: Description for history

        Returns:
            Validation messages
        """
        new_content = self._content[:start] + text + self._content[end:]
        return self.set_content(new_content, description)

    def replace_element(
        self,
        element_name: str,
        old_content: str,
        new_content: str,
    ) -> list[ValidationMessage]:
        """
        Replace specific element content.

        Args:
            element_name: Name of element (e.g., link name)
            old_content: Content to find
            new_content: Replacement content

        Returns:
            Validation messages
        """
        if old_content not in self._content:
            logger.warning(f"Content not found: {old_content[:50]}...")
            return []

        content = self._content.replace(old_content, new_content, 1)
        return self.set_content(content, f"Replace {element_name}")

    # ============================================================
    # History / Undo / Redo
    # ============================================================

    def undo(self) -> bool:
        """
        Undo last change.

        Returns:
            True if undone
        """
        if self._history_index <= 0:
            logger.warning("Nothing to undo")
            return False

        self._history_index -= 1
        self._content = self._history[self._history_index].content

        logger.info(f"Undone to: {self._history[self._history_index].description}")
        return True

    def redo(self) -> bool:
        """
        Redo last undone change.

        Returns:
            True if redone
        """
        if self._history_index >= len(self._history) - 1:
            logger.warning("Nothing to redo")
            return False

        self._history_index += 1
        self._content = self._history[self._history_index].content

        logger.info(f"Redone to: {self._history[self._history_index].description}")
        return True

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._history_index > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._history_index < len(self._history) - 1

    def get_history(self) -> list[dict[str, Any]]:
        """
        Get version history.

        Returns:
            List of version info dicts
        """
        return [
            {
                "index": idx,
                "description": v.description,
                "timestamp": v.timestamp.isoformat(),
                "checksum": v.checksum[:8],
                "is_current": idx == self._history_index,
            }
            for idx, v in enumerate(self._history)
        ]

    def go_to_version(self, index: int) -> bool:
        """
        Go to a specific version in history.

        Args:
            index: Version index

        Returns:
            True if successful
        """
        if index < 0 or index >= len(self._history):
            logger.error(f"Invalid version index: {index}")
            return False

        self._history_index = index
        self._content = self._history[index].content
        logger.info(f"Went to version {index}: {self._history[index].description}")
        return True

    def _add_to_history(self, description: str) -> None:
        """Add current content to history."""
        checksum = hashlib.md5(self._content.encode()).hexdigest()

        version = EditorVersion(
            content=self._content,
            timestamp=datetime.now(),
            description=description,
            checksum=checksum,
        )

        # Remove any redo history
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        self._history.append(version)
        self._history_index = len(self._history) - 1

        # Limit history size
        while len(self._history) > self._max_history:
            self._history.pop(0)
            self._history_index -= 1

    # ============================================================
    # Callbacks
    # ============================================================

    def register_validation_callback(
        self, callback: Callable[[list[ValidationMessage]], None]
    ) -> None:
        """Register callback for validation results."""
        self._validation_callbacks.append(callback)

    def register_change_callback(self, callback: Callable[[str], None]) -> None:
        """Register callback for content changes."""
        self._change_callbacks.append(callback)

    # ============================================================
    # Utilities
    # ============================================================

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        return self._content != self._original_content

    def get_line_count(self) -> int:
        """Get number of lines."""
        return len(self._content.splitlines())

    def get_line(self, line_number: int) -> str | None:
        """
        Get a specific line (1-indexed).

        Args:
            line_number: Line number (1-indexed)

        Returns:
            Line content or None
        """
        lines = self._content.splitlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        return None

    def find_text(
        self, pattern: str, regex: bool = False
    ) -> list[tuple[int, int, str]]:
        """
        Find all occurrences of text.

        Args:
            pattern: Search pattern
            regex: If True, treat pattern as regex

        Returns:
            List of (line, column, matched_text) tuples
        """
        results = []
        lines = self._content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            if regex:
                for match in re.finditer(pattern, line):
                    results.append((line_idx, match.start(), match.group()))
            else:
                start = 0
                while True:
                    pos = line.find(pattern, start)
                    if pos == -1:
                        break
                    results.append((line_idx, pos, pattern))
                    start = pos + 1

        return results

    def replace_all(
        self,
        search: str,
        replace: str,
        regex: bool = False,
    ) -> int:
        """
        Replace all occurrences.

        Args:
            search: Search pattern
            replace: Replacement text
            regex: If True, treat as regex

        Returns:
            Number of replacements made
        """
        if regex:
            new_content, count = re.subn(search, replace, self._content)
        else:
            count = self._content.count(search)
            new_content = self._content.replace(search, replace)

        if count > 0:
            self.set_content(new_content, f"Replace all '{search}' -> '{replace}'")

        return count

    def format_xml(self, indent: str = "  ") -> list[ValidationMessage]:
        """
        Format/prettify the XML content.

        Args:
            indent: Indentation string

        Returns:
            Validation messages
        """
        try:
            root = DefusedET.fromstring(self._content)
            self._indent_xml(root, indent=indent)
            formatted = ET.tostring(root, encoding="unicode")

            # Add XML declaration
            formatted = '<?xml version="1.0"?>\n' + formatted

            return self.set_content(formatted, "Format XML")
        except ET.ParseError as e:
            return [
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=1,
                    column=0,
                    message=f"Cannot format: {e}",
                )
            ]

    def _indent_xml(
        self,
        elem: ET.Element,
        level: int = 0,
        indent: str = "  ",
    ) -> None:
        """Recursively add indentation to XML element."""
        i = "\n" + level * indent
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + indent
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1, indent)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def get_element_at_position(self, line: int, column: int) -> dict[str, Any] | None:
        """
        Get information about element at position.

        Args:
            line: Line number (1-indexed)
            column: Column number (0-indexed)

        Returns:
            Dict with element info or None
        """
        lines = self._content.splitlines()
        if line < 1 or line > len(lines):
            return None

        line_content = lines[line - 1]

        # Find element tags on this line
        for match in re.finditer(r"<(\w+)([^>]*)>", line_content):
            start = match.start()
            end = match.end()
            if start <= column <= end:
                tag = match.group(1)
                attrs_str = match.group(2)

                # Parse attributes
                attrs = {}
                for attr_match in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attrs_str):
                    attrs[attr_match.group(1)] = attr_match.group(2)

                return {
                    "tag": tag,
                    "attributes": attrs,
                    "line": line,
                    "start_column": start,
                    "end_column": end,
                }

        return None
