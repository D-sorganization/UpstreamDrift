"""Regression tests ensuring redacted shell code blocks remain runnable."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_TARGET_MARKDOWN_FILES = (
    "docs/plans/club_only_matching/REPRODUCTION_GUIDE.md",
    "docs/plans/qualified_motion_integration/PINK_TURNOVER.md",
    "docs/plans/qualified_motion_integration/TURNOVER.md",
    "docs/research/shot_pattern_analysis/TURNOVER.md",
    "docs/research/shot_pattern_analysis/phase2/RESUME_HANDOFF.md",
    "docs/research/shot_pattern_analysis/phase2/TURNOVER.md",
    "docs/research/shot_pattern_analysis/phase2/review_followups/ARTIFACT_DELIVERY_PLAN.md",
    "docs/research/shot_pattern_analysis/phase2/review_followups/COURSE_SCORING_REFACTOR_REVIEW.md",
)

_SHELL_LANGUAGES = frozenset(
    {"bash", "sh", "shell", "zsh", "console", "powershell", "pwsh", "cmd"}
)
_FENCE_OPEN_RE = re.compile(r"^\s*(?:```|~~~)\s*([A-Za-z0-9_+-]+)?")
_FENCE_CLOSE_RE = re.compile(r"^\s*(?:```|~~~)\s*$")
_PLACEHOLDER_TOKENS = ("<owner-home>", "<host>")


def _iter_shell_fence_violations(rel_path: str) -> list[str]:
    """Return human-readable violations for shell code blocks in a Markdown file."""
    path = _REPO_ROOT / rel_path
    lines = path.read_text(encoding="utf-8").splitlines()
    violations: list[str] = []
    in_shell_fence = False

    for line_no, line in enumerate(lines, start=1):
        if in_shell_fence:
            if _FENCE_CLOSE_RE.match(line):
                in_shell_fence = False
                continue
            for token in _PLACEHOLDER_TOKENS:
                if token in line:
                    violations.append(
                        f"{rel_path}:{line_no}: shell block contains unrunnable placeholder {token}"
                    )
        else:
            match = _FENCE_OPEN_RE.match(line)
            if match:
                lang = (match.group(1) or "").strip().lower()
                if lang in _SHELL_LANGUAGES:
                    in_shell_fence = True

    return violations


@pytest.mark.unit
def test_shell_blocks_use_env_vars_instead_of_angle_placeholders() -> None:
    """Runnable shell blocks must not break with angle-bracket redaction tokens."""
    violations: list[str] = []
    for rel_path in _TARGET_MARKDOWN_FILES:
        violations.extend(_iter_shell_fence_violations(rel_path))
    assert not violations, "\n".join(violations)
