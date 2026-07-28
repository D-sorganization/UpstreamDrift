"""Regression tests for the Bot CI trigger workflow budget."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "Bot-CI-Trigger.yml"


def test_bot_ci_trigger_has_no_backlog_polling_or_empty_commit_fallback() -> None:
    """The CI trigger workflow must stay bounded to one explicit PR."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    forbidden_patterns = [
        "schedule:",
        "gh pr list",
        "gh pr checks",
        "git commit --allow-empty",
        "git push origin",
        "continue-on-error: true",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in workflow


def test_bot_ci_trigger_uses_bounded_rest_inspection() -> None:
    """Current CI status inspection should use REST for one PR head SHA."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "repos/$GITHUB_REPOSITORY/pulls/$TARGET_PR" in workflow
    assert "commits/$HEAD_SHA/check-runs?per_page=100" in workflow
    assert "actions/workflows/ci-standard.yml/dispatches" in workflow
