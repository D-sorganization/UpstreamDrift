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
# Prettier renders the separator with as many cells as the widest row (three
# historical rows carry unescaped pipes), so accept any all-dash pipe row.
_HEADER_SEPARATOR_ROW = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|){3,}\s*$")
_REPAIR_ENTRY = (
    "| 2026-08-22 | 1.0.574 | Retained PR #8995's MediaPipe landmark-mean optimization"
)


def _spec_lines() -> list[str]:
    return _SPEC_PATH.read_text(encoding="utf-8").splitlines()


def _change_log_bounds(lines: list[str]) -> tuple[int, int]:
    """Return ``(start, end)`` line indices of the Section 12 region.

    ``start`` is the heading line; ``end`` is the next ``## `` heading (or end
    of file), so rows belong to Section 12 iff ``start <= index < end``.
    """
    start = lines.index(_CHANGE_LOG_HEADING)
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return start, end


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


def test_no_changelog_rows_outside_section_12() -> None:
    """Every dated change-log row must live inside the Section 12 table.

    Git's merge machinery repeatedly anchors ``| YYYY-MM-DD | 1.0.N | ... |``
    rows into *other* pipe tables (Section 8's tool table, the post-Section-12
    note sections) because all pipe rows share the same shape, so a textual
    auto-merge cannot tell the tables apart. This gate makes such a bad merge
    fail CI while the merger still has context, instead of the rows silently
    accreting where no reader looks for them.
    """
    lines = _spec_lines()
    start, end = _change_log_bounds(lines)
    offenders = [
        f"  line {i + 1}: {line[:120]}"
        for i, line in enumerate(lines)
        if _CHANGE_LOG_ROW.match(line) and not (start <= i < end)
    ]

    assert offenders == [], (
        "Dated change-log rows found outside the '## 12. Change Log' section, "
        "almost certainly grafted there by a git auto-merge. Move them into "
        "the Section 12 table (in version order), or delete them if they "
        "duplicate an existing Section 12 row:\n" + "\n".join(offenders)
    )


def test_section_12_table_has_a_single_header_separator() -> None:
    """The Section 12 table must not be split by duplicated header separators.

    The same auto-merge failure mode that scatters rows also duplicates the
    ``| --- | --- | --- |`` separator mid-table, silently splitting the change
    log into fragments that render as prose.
    """
    lines = _spec_lines()
    start, end = _change_log_bounds(lines)
    separators = [
        i + 1 for i in range(start, end) if _HEADER_SEPARATOR_ROW.match(lines[i])
    ]

    assert len(separators) == 1, (
        "Expected exactly one header separator row in the Section 12 change "
        f"log, found {len(separators)} at lines {separators}."
    )
