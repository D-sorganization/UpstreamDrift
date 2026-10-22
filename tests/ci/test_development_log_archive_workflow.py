"""Contracts for the scheduled development-log archiver workflow."""

from __future__ import annotations

import os
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from shared_scripts.development_log import ARCHIVE_AFTER_DAYS, archive_repository_devlog
from shared_scripts.development_log_schema import (
    CANONICAL_RELATIVE_PATH,
    TERMINAL_STATES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/development-log-archive.yml"

pytestmark = pytest.mark.unit


def _workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(name: str) -> dict:
    return next(
        step
        for step in _workflow()["jobs"]["archive"]["steps"]
        if step.get("name") == name
    )


def _minimal_devlog(*, last_verified: date, state: str = "shipped") -> str:
    assert state in TERMINAL_STATES
    return (
        "# Development Log — fixture\n\n"
        "## Active\n\n"
        "### DL-#99999 · Archive workflow fixture\n\n"
        f"- **State:** {state}\n"
        "- **Owner:** cursor\n"
        "- **Issue:** #99999\n"
        "- **PR:** #1\n"
        "- **Paths:** `tests/ci/test_development_log_archive_workflow.py`\n"
        "- **Started:** 2026-01-01\n"
        f"- **Last verified:** {last_verified.isoformat()} (fixture).\n"
        "- **Summary:** Synthetic entry for archiver workflow tests.\n\n"
    )


def test_archive_is_noop_while_terminal_entries_are_within_archive_window(
    tmp_path: Path,
) -> None:
    today = date(2026, 10, 22)
    recent = today - timedelta(days=ARCHIVE_AFTER_DAYS)
    log_path = tmp_path / CANONICAL_RELATIVE_PATH
    log_path.parent.mkdir(parents=True)
    original = _minimal_devlog(last_verified=recent)
    log_path.write_text(original, encoding="utf-8")

    written = archive_repository_devlog(tmp_path, today=today, days=ARCHIVE_AFTER_DAYS)

    assert written == []
    assert log_path.read_text(encoding="utf-8") == original


@pytest.mark.skipif(
    not (REPO_ROOT / "scripts/automerge_guard.py").is_file(),
    reason="automerge_guard.py required for workflow contract",
)
def test_propose_step_exits_without_pr_when_archive_made_no_changes(
    tmp_path: Path,
) -> None:
    script = _step("Propose archive as a pull request")["run"]
    env_file = tmp_path / "github_env"
    env_file.write_text("OUTCOME=archive step\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "CI test"], cwd=repo, check=True)
    (repo / "docs/development").mkdir(parents=True)
    (repo / "docs/development/DEVELOPMENT_LOG.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)

    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=repo,
        env={
            **os.environ,
            "GITHUB_ENV": str(env_file),
            "GH_TOKEN": "unused",
            "BRANCH": "bot/development-log-archive",
            "REPO_SLUG": "o/r",
            "GUARD": str(REPO_ROOT / "scripts/automerge_guard.py"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no archive changes" in env_file.read_text(encoding="utf-8")


def test_workflow_runs_archiver_and_targets_docs_development_only() -> None:
    workflow = _workflow()
    assert "schedule" in workflow["on"]
    cron = workflow["on"]["schedule"][0]["cron"]
    assert cron == "0 11 * * *"

    archive_run = _step("Archive terminal development-log entries")["run"]
    assert "shared_scripts/development_log.py" in archive_run
    assert "--archive" in archive_run

    propose = _step("Propose archive as a pull request")
    propose_run = propose["run"]
    assert propose["env"]["BRANCH"] == "bot/development-log-archive"
    assert "docs/development/DEVELOPMENT_LOG.md" in propose_run
    assert "automerge_guard" in propose["env"]["GUARD"]
