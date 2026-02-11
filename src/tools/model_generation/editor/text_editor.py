"""
URDF Text Editor with diff view support.

Provides text-based editing of URDF files with:
- Syntax validation
- Diff generation between versions
- Undo/redo support
- Real-time validation feedback
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DefusedET

logger = logging.getLogger(__name__)


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


class URDFTextEditor:
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
    # Validation
    # ============================================================

    def validate(self) -> list[ValidationMessage]:
        """
        Validate current URDF content.

        Returns:
            List of validation messages
        """
        messages = []

        # XML validation
        messages.extend(self._validate_xml())

        if not any(m.severity == ValidationSeverity.ERROR for m in messages):
            # URDF-specific validation
            messages.extend(self._validate_urdf())

        return messages

    def _validate_xml(self) -> list[ValidationMessage]:
        """Validate XML syntax."""
        messages = []

        try:
            DefusedET.fromstring(self._content)
        except ET.ParseError as e:
            # Parse error message for line/column
            error_str = str(e)
            line, col = 1, 0

            # Try to extract line number
            match = re.search(r"line (\d+)", error_str)
            if match:
                line = int(match.group(1))

            match = re.search(r"column (\d+)", error_str)
            if match:
                col = int(match.group(1))

            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=line,
                    column=col,
                    message=f"XML syntax error: {error_str}",
                )
            )

        return messages

    def _validate_urdf(self) -> list[ValidationMessage]:
        """Validate URDF-specific rules."""
        messages: list[ValidationMessage] = []

        try:
            root = DefusedET.fromstring(self._content)
        except ET.ParseError:
            return messages  # Already reported in XML validation

        # Check root element
        if root.tag != "robot":
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    line=1,
                    column=0,
                    message=f"Root element should be 'robot', got '{root.tag}'",
                )
            )
            return messages

        # Check robot name
        if not root.get("name"):
            messages.append(
                ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    line=1,
                    column=0,
                    message="Robot element missing 'name' attribute",
                    element="robot",
                )
            )

        # Collect links and joints
        links = {}
        joints = {}

        for _idx, link_elem in enumerate(root.findall("link")):
            name = link_elem.get("name")
            if not name:
                line = self._find_element_line(link_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message="Link element missing 'name' attribute",
                        element="link",
                    )
                )
            elif name in links:
                line = self._find_element_line(link_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Duplicate link name: '{name}'",
                        element=name,
                    )
                )
            else:
                links[name] = link_elem

            # Check inertial
            inertial = link_elem.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    mass = mass_elem.get("value")
                    if mass is not None:
                        try:
                            mass_val = float(mass)
                            if mass_val < 0:
                                line = self._find_element_line(mass_elem)
                                messages.append(
                                    ValidationMessage(
                                        severity=ValidationSeverity.ERROR,
                                        line=line,
                                        column=0,
                                        message=f"Negative mass value: {mass_val}",
                                        element=name,
                                    )
                                )
                            elif mass_val == 0:
                                line = self._find_element_line(mass_elem)
                                messages.append(
                                    ValidationMessage(
                                        severity=ValidationSeverity.WARNING,
                                        line=line,
                                        column=0,
                                        message="Zero mass value",
                                        element=name,
                                    )
                                )
                        except ValueError:
                            line = self._find_element_line(mass_elem)
                            messages.append(
                                ValidationMessage(
                                    severity=ValidationSeverity.ERROR,
                                    line=line,
                                    column=0,
                                    message=f"Invalid mass value: '{mass}'",
                                    element=name,
                                )
                            )

        for _idx, joint_elem in enumerate(root.findall("joint")):
            name = joint_elem.get("name")
            if not name:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message="Joint element missing 'name' attribute",
                        element="joint",
                    )
                )
            elif name in joints:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Duplicate joint name: '{name}'",
                        element=name,
                    )
                )
            else:
                joints[name] = joint_elem

            # Check joint type
            joint_type = joint_elem.get("type")
            valid_types = {
                "revolute",
                "continuous",
                "prismatic",
                "fixed",
                "floating",
                "planar",
            }
            if joint_type not in valid_types:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message=f"Invalid joint type: '{joint_type}'",
                        element=name,
                    )
                )

            # Check parent/child references
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")

            if parent_elem is None:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message="Joint missing parent element",
                        element=name,
                    )
                )
            else:
                parent_link = parent_elem.get("link")
                if parent_link and parent_link not in links:
                    line = self._find_element_line(parent_elem)
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.ERROR,
                            line=line,
                            column=0,
                            message=f"Parent link not found: '{parent_link}'",
                            element=name,
                        )
                    )

            if child_elem is None:
                line = self._find_element_line(joint_elem)
                messages.append(
                    ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        line=line,
                        column=0,
                        message="Joint missing child element",
                        element=name,
                    )
                )
            else:
                child_link = child_elem.get("link")
                if child_link and child_link not in links:
                    line = self._find_element_line(child_elem)
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.ERROR,
                            line=line,
                            column=0,
                            message=f"Child link not found: '{child_link}'",
                            element=name,
                        )
                    )

            # Check limits for revolute/prismatic
            if joint_type in {"revolute", "prismatic"}:
                limit_elem = joint_elem.find("limit")
                if limit_elem is None:
                    line = self._find_element_line(joint_elem)
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            line=line,
                            column=0,
                            message=f"{joint_type} joint missing limit element",
                            element=name,
                        )
                    )

        # Check for orphan links (no joint connection)
        child_links = set()
        for joint_elem in root.findall("joint"):
            child_elem = joint_elem.find("child")
            if child_elem is not None:
                child_links.add(child_elem.get("link"))

        for link_name in links:
            if link_name not in child_links:
                # This might be the root link
                is_parent = any(
                    (parent := j.find("parent")) is not None
                    and parent.get("link") == link_name
                    for j in root.findall("joint")
                )
                if not is_parent and len(links) > 1:
                    messages.append(
                        ValidationMessage(
                            severity=ValidationSeverity.WARNING,
                            line=1,
                            column=0,
                            message=f"Link '{link_name}' is not connected to any joint",
                            element=link_name,
                        )
                    )

        return messages

    def _find_element_line(self, elem: ET.Element) -> int:
        """Find the line number of an element (approximate)."""
        # This is a simple heuristic - search for element in content
        ET.tostring(elem, encoding="unicode")
        tag_start = f"<{elem.tag}"

        # Find in content
        lines = self._content.split("\n")
        for idx, line in enumerate(lines, 1):
            if tag_start in line:
                # Check if attributes match
                name = elem.get("name")
                if name is None or f'name="{name}"' in line or f"name='{name}'" in line:
                    return idx

        return 1

    # ============================================================
    # Diff Operations
    # ============================================================

    def get_diff_from_original(self) -> DiffResult:
        """
        Get diff between current content and original.

        Returns:
            DiffResult with changes
        """
        return self._compute_diff(self._original_content, self._content)

    def get_diff_between_versions(
        self,
        version_a: int,
        version_b: int,
    ) -> DiffResult:
        """
        Get diff between two versions in history.

        Args:
            version_a: First version index
            version_b: Second version index

        Returns:
            DiffResult with changes
        """
        if version_a < 0 or version_a >= len(self._history):
            raise IndexError(f"Invalid version index: {version_a}")
        if version_b < 0 or version_b >= len(self._history):
            raise IndexError(f"Invalid version index: {version_b}")

        content_a = self._history[version_a].content
        content_b = self._history[version_b].content

        return self._compute_diff(content_a, content_b)

    def get_diff_with_string(self, other_content: str) -> DiffResult:
        """
        Get diff between current content and provided string.

        Args:
            other_content: Content to compare with

        Returns:
            DiffResult with changes
        """
        return self._compute_diff(self._content, other_content)

    def _compute_diff(self, original: str, modified: str) -> DiffResult:
        """Compute diff between two strings."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        # Generate unified diff
        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile="original",
                tofile="modified",
                lineterm="",
            )
        )
        unified_diff = "".join(diff_lines)

        # Parse hunks
        hunks: list[DiffHunk] = []
        current_hunk_lines: list[str] = []
        old_start, old_count, new_start, new_count = 0, 0, 0, 0
        additions = 0
        deletions = 0

        for line in diff_lines:
            if line.startswith("@@"):
                # Save previous hunk
                if current_hunk_lines:
                    hunks.append(
                        DiffHunk(
                            old_start=old_start,
                            old_count=old_count,
                            new_start=new_start,
                            new_count=new_count,
                            lines=current_hunk_lines,
                        )
                    )
                    current_hunk_lines = []

                # Parse hunk header
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2) or 1)
                    new_start = int(match.group(3))
                    new_count = int(match.group(4) or 1)
            elif line.startswith(("---", "+++")):
                continue
            elif line.startswith("+"):
                current_hunk_lines.append(line)
                additions += 1
            elif line.startswith("-"):
                current_hunk_lines.append(line)
                deletions += 1
            elif line.startswith(" ") or line == "\n":
                current_hunk_lines.append(line)

        # Save last hunk
        if current_hunk_lines:
            hunks.append(
                DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=current_hunk_lines,
                )
            )

        return DiffResult(
            original_content=original,
            modified_content=modified,
            hunks=hunks,
            unified_diff=unified_diff,
            additions=additions,
            deletions=deletions,
            has_changes=original != modified,
        )

    def get_side_by_side_diff(
        self,
        original: str | None = None,
        modified: str | None = None,
        context_lines: int = 3,
    ) -> list[tuple[str | None, str | None, str]]:
        """
        Get side-by-side diff representation.

        Args:
            original: Original content (default: original file content)
            modified: Modified content (default: current content)
            context_lines: Number of context lines

        Returns:
            List of (left_line, right_line, change_type) tuples.
            change_type is one of: 'equal', 'insert', 'delete', 'replace'
        """
        if original is None:
            original = self._original_content
        if modified is None:
            modified = self._content

        original_lines = original.splitlines()
        modified_lines = modified.splitlines()

        differ = difflib.SequenceMatcher(None, original_lines, modified_lines)

        result: list[tuple[str | None, str | None, str]] = []
        for opcode, i1, i2, j1, j2 in differ.get_opcodes():
            if opcode == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2), strict=False):
                    result.append((original_lines[i], modified_lines[j], "equal"))
            elif opcode == "insert":
                for j in range(j1, j2):
                    result.append((None, modified_lines[j], "insert"))
            elif opcode == "delete":
                for i in range(i1, i2):
                    result.append((original_lines[i], None, "delete"))
            elif opcode == "replace":
                # Match up lines as best we can
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    left = original_lines[i1 + k] if i1 + k < i2 else None
                    right = modified_lines[j1 + k] if j1 + k < j2 else None
                    result.append((left, right, "replace"))

        return result

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
