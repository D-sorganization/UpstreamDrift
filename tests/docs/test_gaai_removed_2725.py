"""The retired GAAI framework stays removed (Repository_Management #2725)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# Live entry points and tooling that must not point at the removed framework.
# Historical records (assessments, archives, the SPEC change log) are exempt.
LIVE_FILES = (
    "CLAUDE.md",
    "AGENTS.md",
    ".ai/README.md",
    "scripts/README.md",
    "docs/development/agents/migration.md",
    "scripts/check_workflow_inventory.py",
    "scripts/check_agent_docs_consistency.py",
    "scripts/config/root_allowlist.json",
)

_FLEET_MANAGED = re.compile(
    r"<!-- BEGIN FLEET-MANAGED.*?<!-- END FLEET-MANAGED[^>]*-->", re.DOTALL
)
# The fleet roster names the retired agent id; that list is owned elsewhere.
_ROSTER_LINE = re.compile(r"^.*\blocal\b.*\bgaai\b.*$", re.MULTILINE)


def _tracked(*pathspecs: str) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", "--", *pathspecs],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def _live_text(relative_path: str) -> str:
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    text = _FLEET_MANAGED.sub("", text)
    return _ROSTER_LINE.sub("", text)


def test_no_tracked_files_under_gaai_dir() -> None:
    assert _tracked(".gaai/") == []


def test_gaai_launcher_and_slash_commands_are_gone() -> None:
    assert _tracked("scripts/maintenance/start-gaai-daemon.sh") == []
    assert _tracked(".claude/commands/gaai-*.md") == []


@pytest.mark.parametrize("relative_path", LIVE_FILES)
def test_live_files_do_not_reference_gaai(relative_path: str) -> None:
    assert "gaai" not in _live_text(relative_path).lower()


def test_roster_filter_only_hides_roster_lines() -> None:
    """A pointer line is still caught after the roster filter runs."""
    pointer = "> GAAI framework installed in `.gaai/`.\n"
    roster = "   `user > claude > codex > jules > local > gaai`. The\n"
    assert "gaai" in _ROSTER_LINE.sub("", pointer).lower()
    assert "gaai" not in _ROSTER_LINE.sub("", roster).lower()
