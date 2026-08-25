"""Regression guards for the canonical specification change log."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_PATH = _REPO_ROOT / "SPEC.md"
_CHANGE_LOG_HEADING = "## 12. Change Log"
_CHANGE_LOG_ROW = re.compile(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|\s*\d+\.\d+\.\d+\s*\|")
_REPAIR_ENTRY = (
    "| 2026-08-22 | 1.0.574 | Retained PR #8995's MediaPipe landmark-mean optimization"
)


def _spec_lines() -> list[str]:
    return _SPEC_PATH.read_text(encoding="utf-8").splitlines()


def test_changelog_rows_are_not_duplicated_across_specification() -> None:
    """A generated change-log row must not be injected throughout the document."""
    lines = _spec_lines()
    assert _CHANGE_LOG_HEADING in lines
    counts = Counter(line for line in lines if _CHANGE_LOG_ROW.match(line))
    # Two legacy rows occur twice in the first-parent specification. Preserve
    # that inherited debt here; this guard targets generated fan-out such as
    # #8995, which repeated one row thousands of times.
    duplicates = {line: count for line, count in counts.items() if count > 2}

    assert duplicates == {}, f"Duplicated change-log rows: {duplicates}"


def test_spec_repair_entry_is_recorded_exactly_once() -> None:
    """The #8995 optimization and its repair have one canonical history record."""
    matches = [line for line in _spec_lines() if line.startswith(_REPAIR_ENTRY)]

    assert len(matches) == 1
