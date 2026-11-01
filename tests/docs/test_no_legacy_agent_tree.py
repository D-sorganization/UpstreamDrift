"""The legacy .agent/ tree stays removed; agy and Codex read .agents/ (#14201)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_tracked_files_under_legacy_agent_dir() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".agent/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == []


def test_docs_do_not_describe_agent_dir_as_live() -> None:
    readme = (REPO_ROOT / ".ai" / "README.md").read_text(encoding="utf-8")
    assert "`.agent/skills/`" not in readme
    assert "`.agent/workflows/`" not in readme


def test_gaai_project_rules_fence_ci_watch_and_fix() -> None:
    # The core skill keeps a session alive for CI; the fleet PR lifecycle
    # ends it at PR open, so the project layer fences it (RM #2567).
    rules = REPO_ROOT / ".gaai" / "project" / "contexts" / "rules"
    text = (rules / "project.rules.md").read_text(encoding="utf-8")
    assert "invoke the `ci-watch-and-fix` skill" not in text
    assert "Do not run core skill `delivery/ci-watch-and-fix`" in text
    assert "fleet-pr-lifecycle" in text
