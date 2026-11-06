"""Contracts for the size-aware development-log archiver (#14535)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from scripts.ci import archive_devlog_with_headroom as headroom
from shared_scripts.development_log_schema import (
    ARCHIVE_AFTER_DAYS,
    CANONICAL_RELATIVE_PATH,
    MAX_BYTES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/development-log-archive.yml"
TODAY = date(2026, 11, 5)

pytestmark = pytest.mark.unit


def _entry(issue: int, finished: date, padding: int = 0) -> str:
    return (
        f"### DL-#{issue} · Fixture {issue}\n\n"
        "- **State:** shipped\n"
        "- **Owner:** cursor\n"
        f"- **Issue:** #{issue}\n"
        "- **PR:** #1\n"
        "- **Paths:** `tests/ci/test_archive_devlog_with_headroom.py`\n"
        "- **Started:** 2026-01-01\n"
        f"- **Last verified:** {finished.isoformat()} (fixture).\n"
        f"- **Summary:** {'x' * padding}\n\n"
    )


def _write_log(root: Path, entries: list[str]) -> Path:
    path = root / CANONICAL_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Development Log\n\n## Active\n\n" + "".join(entries), "utf-8")
    return path


def _young_heavy_log(root: Path, *, total: int) -> Path:
    """Entries all younger than the default window, sized to ``total`` bytes."""
    ages = [1, 2, 3, 4, 5, 6]
    per_entry = total // len(ages)
    entries = [
        _entry(1000 + age, TODAY - timedelta(days=age), per_entry - 260) for age in ages
    ]
    return _write_log(root, entries)


def test_trigger_is_below_the_hard_ceiling_and_target_below_trigger() -> None:
    assert headroom.HEADROOM_TARGET_BYTES < headroom.SIZE_TRIGGER_BYTES < MAX_BYTES
    assert headroom.SIZE_TRIGGER_BYTES == 180_000


def test_default_window_is_unchanged_when_log_is_small(tmp_path: Path) -> None:
    path = _young_heavy_log(tmp_path, total=50_000)
    before = path.read_text(encoding="utf-8")

    written = headroom.archive_with_headroom(tmp_path, today=TODAY)

    assert written == []
    assert path.read_text(encoding="utf-8") == before


def test_near_cap_log_archives_young_entries_oldest_first(tmp_path: Path) -> None:
    path = _young_heavy_log(tmp_path, total=190_000)
    assert path.stat().st_size >= headroom.SIZE_TRIGGER_BYTES

    written = headroom.archive_with_headroom(tmp_path, today=TODAY)

    assert written, "size trigger must archive even though no entry exceeds 7 days"
    remaining = path.read_text(encoding="utf-8")
    assert len(remaining.encode("utf-8")) < headroom.HEADROOM_TARGET_BYTES
    # Oldest terminal entry (6 days) goes first; newest (1 day) is kept.
    assert "DL-#1006" not in remaining
    assert "DL-#1001" in remaining
    archive = (path.parent / f"DEVELOPMENT_LOG_ARCHIVE_{TODAY.year}.md").read_text(
        encoding="utf-8"
    )
    assert "DL-#1006" in archive


def test_stops_as_soon_as_target_is_met(tmp_path: Path) -> None:
    path = _young_heavy_log(tmp_path, total=182_000)

    headroom.archive_with_headroom(tmp_path, today=TODAY)

    # Only the slack that is needed leaves; the newest entries stay.
    remaining = path.read_text(encoding="utf-8")
    assert "DL-#1001" in remaining
    assert "DL-#1002" in remaining


def test_non_terminal_entries_are_never_archived(tmp_path: Path) -> None:
    active = _entry(2000, TODAY - timedelta(days=30), 190_000).replace(
        "State:** shipped", "State:** in_progress"
    )
    path = _write_log(tmp_path, [active])

    headroom.archive_with_headroom(tmp_path, today=TODAY)

    assert "DL-#2000" in path.read_text(encoding="utf-8")


def test_age_windows_start_at_the_default_and_tighten() -> None:
    assert headroom.AGE_WINDOWS[0] == ARCHIVE_AFTER_DAYS
    assert list(headroom.AGE_WINDOWS) == sorted(headroom.AGE_WINDOWS, reverse=True)
    assert headroom.AGE_WINDOWS[-1] >= 0


def test_workflow_uses_the_size_aware_entry_point() -> None:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    step = next(
        s
        for s in workflow["jobs"]["archive"]["steps"]
        if s.get("name") == "Archive terminal development-log entries"
    )
    assert "scripts/ci/archive_devlog_with_headroom.py" in step["run"]


def test_archive_after_days_window_is_seven_days() -> None:
    """Issue #14499: archive window reduced to 7 days to keep log under 200k ceiling."""
    assert ARCHIVE_AFTER_DAYS == 7
    assert headroom.AGE_WINDOWS[0] == 7
