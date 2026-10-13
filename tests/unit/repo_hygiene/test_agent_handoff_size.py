"""AGENT_HANDOFF.md stays current-state only; history lives in the archive (#12416)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HANDOFF = REPO_ROOT / "AGENT_HANDOFF.md"
ARCHIVE = REPO_ROOT / "docs" / "development" / "AGENT_HANDOFF_ARCHIVE.md"
MAX_LIVE_LINES = 150  # AGENTS.md policy; never loosen.


def test_live_handoff_is_within_the_agents_md_line_limit() -> None:
    assert len(HANDOFF.read_text(encoding="utf-8").splitlines()) <= MAX_LIVE_LINES


def test_history_archive_exists_and_is_linked_from_the_live_handoff() -> None:
    assert ARCHIVE.is_file(), "history must move to the archive, not be deleted"
    assert "docs/development/AGENT_HANDOFF_ARCHIVE.md" in HANDOFF.read_text(
        encoding="utf-8"
    )
    assert "#12416" in ARCHIVE.read_text(encoding="utf-8")
