"""
Diff operations mixin for the URDFTextEditor.

Extracted from URDFTextEditor to respect SRP:
diff computation is independent of editing, history, and validation logic.
"""

from __future__ import annotations

import difflib
import re

from .text_editor_types import DiffHunk, DiffResult


class DiffMixin:
    """Diff operations for the URDFTextEditor.

    Requires host class to provide:
        _content: str
        _original_content: str
        _history: list[EditorVersion]
    """

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

    @staticmethod
    def _compute_diff(original: str, modified: str) -> DiffResult:
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
                match = re.match(
                    r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line
                )
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
