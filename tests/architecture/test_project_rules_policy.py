from pathlib import Path

import pytest


PROJECT_RULES = (
    Path(__file__).resolve().parents[2]
    / ".gaai/project/contexts/rules/project.rules.md"
)


@pytest.mark.unit
def test_project_rules_match_authoritative_main_branch_policy() -> None:
    rules = PROJECT_RULES.read_text(encoding="utf-8")

    assert "PRs target `main`." in rules
    assert "automated merge may be enabled" in rules
    assert "Human approval required" not in rules
    assert "All AI work on `staging`" not in rules
    # The actual stale wording superseded by #9264/#9279: rejecting only the
    # `staging` variant above lets a regression to this pre-change sentence
    # pass silently, since it was never rejected in the first place.
    assert "All AI work on `main` branch. Never commit directly" not in rules
    assert "focused topic branch or isolated worktree" in rules


@pytest.mark.unit
def test_stale_main_branch_sentence_would_fail_the_guard() -> None:
    """Prove the guard above actually catches the #9264 regression case.

    Without this, a typo in the guard's literal string (e.g. matching text
    that no longer appears anywhere) could pass forever without ever
    rejecting anything.
    """
    reverted_rules = (
        "1. All AI work on `main` branch. Never commit directly to `main`.\n"
        "2. PRs target `main`. No auto-merge. Human approval required.\n"
    )

    assert "All AI work on `main` branch. Never commit directly" in reverted_rules
