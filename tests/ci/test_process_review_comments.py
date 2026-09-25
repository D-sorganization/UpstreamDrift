"""Unit tests for PR review comment to issue conversion script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.ci.process_review_comments import (
    CommentToIssueProcessor,
    format_issue_body,
    format_issue_title,
    is_actionable_comment,
    is_bot_user,
    is_pr_open,
    main,
)

pytestmark = pytest.mark.unit


def _write_pr_and_comments(
    comments_dir: Path, pr_num: int, state: str, comments: list[dict[str, Any]]
) -> None:
    comments_dir.mkdir(parents=True, exist_ok=True)
    pr_details = {
        "title": f"Test PR #{pr_num}",
        "author": {"login": "human-author"},
        "headRefName": f"feat/pr-{pr_num}",
        "state": state,
    }
    (comments_dir / f"pr_{pr_num}_details.json").write_text(
        json.dumps(pr_details), encoding="utf-8"
    )
    (comments_dir / f"pr_{pr_num}_review_comments.json").write_text(
        json.dumps(comments), encoding="utf-8"
    )


def test_is_bot_user_detects_bot_logins_and_types() -> None:
    assert is_bot_user({"type": "User"}, "chatgpt-codex-connector[bot]") is True
    assert is_bot_user({"type": "User"}, "copilot-pull-request-reviewer[bot]") is True
    assert is_bot_user({"type": "User"}, "github-actions[bot]") is True
    assert is_bot_user({"type": "User"}, "dependabot[bot]") is True
    assert is_bot_user({"type": "Bot"}, "some-bot") is True
    assert is_bot_user({"type": "bot"}, "other-agent") is True
    assert is_bot_user({"type": "User"}, "google-labs-jules") is True
    assert is_bot_user({"type": "User"}, "jules-bot") is True
    assert is_bot_user({"type": "User"}, "dieterolson") is False
    assert is_bot_user({"type": "User"}, "octocat") is False
    assert is_bot_user({}, "contributor") is False
    assert is_bot_user(None, "alice") is False


def test_is_pr_open_checks_state() -> None:
    assert is_pr_open({"state": "OPEN"}) is True
    assert is_pr_open({"state": "open"}) is True
    assert is_pr_open({"state": "CLOSED"}) is False
    assert is_pr_open({"state": "MERGED"}) is False
    assert is_pr_open({}) is False


def test_is_actionable_comment_requires_substantive_action() -> None:
    assert (
        is_actionable_comment("```suggestion\nvalue = calculate_offset(x)\n```") is True
    )
    assert (
        is_actionable_comment("TODO: add boundary check for negative index values")
        is True
    )
    assert (
        is_actionable_comment("FIXME: memory leak when socket connection resets")
        is True
    )
    assert (
        is_actionable_comment("Action item: implement retry logic for network requests")
        is True
    )
    assert (
        is_actionable_comment(
            "/create-issue please extract this into a separate module"
        )
        is True
    )
    assert (
        is_actionable_comment("Please fix the synchronization lock here before merging")
        is True
    )
    assert (
        is_actionable_comment(
            "This needs to be fixed to avoid race conditions in production"
        )
        is True
    )
    assert (
        is_actionable_comment("Must be updated to match the new schema specification")
        is True
    )

    # False positive rejection
    assert (
        is_actionable_comment(
            "We could consider this pattern if we ever decide to change it."
        )
        is False
    )
    assert (
        is_actionable_comment("I should check the benchmark results when they finish.")
        is False
    )

    # Trivial / short comments rejection
    assert is_actionable_comment("LGTM!") is False
    assert is_actionable_comment("Thanks for fixing!") is False
    assert is_actionable_comment("done") is False
    assert is_actionable_comment("please fix") is False
    assert is_actionable_comment("") is False
    assert is_actionable_comment("   \n\t  ") is False


def test_format_issue_title_truncation() -> None:
    assert (
        format_issue_title("src/utils.py", 42) == "[review] Feedback on src/utils.py:42"
    )
    long_path = "src/very/deeply/nested/path/to/some/excessively/long/named/component/module_name_long_identifier.py"
    long_title = format_issue_title(long_path, 1234)
    assert len(long_title) <= 100
    assert long_title.endswith("...")


def test_format_issue_body_includes_metadata_and_marker() -> None:
    body = format_issue_body(
        pr_num="10931",
        pr_title="Harden review comments workflow",
        pr_author="dieterolson",
        pr_branch="fix/10931-hardened",
        author="lead-reviewer",
        file_path="src/engine.py",
        line=150,
        comment_id="987654321",
        html_url="https://github.com/D-sorganization/UpstreamDrift/pull/10931#discussion_r987654321",
        created_at="2026-09-25T05:00:00Z",
        feedback_body="TODO: please ensure thread lock is released in finally block",
    )
    assert "<!-- comment-to-issue: comment_id=987654321 pr=10931 -->" in body
    assert "**Source PR:** #10931 - Harden review comments workflow" in body
    assert "**File:** `src/engine.py` (line 150)" in body
    assert "TODO: please ensure thread lock is released in finally block" in body
    assert "- **PR Author:** @dieterolson" in body
    assert "- **Reviewed by:** @lead-reviewer" in body


def test_processor_skips_bot_comments(tmp_path: Path) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    comments = [
        {
            "id": 11,
            "user": {"login": "chatgpt-codex-connector[bot]", "type": "Bot"},
            "body": "```suggestion\nx=1\n```",
        },
        {
            "id": 22,
            "user": {"login": "google-labs-jules", "type": "User"},
            "body": "TODO: verify this calculation",
        },
    ]
    _write_pr_and_comments(comments_dir, 100, "OPEN", comments)

    runner = MagicMock()
    processor = CommentToIssueProcessor(
        comments_dir=comments_dir,
        archive_dir=archive_dir,
        repo="D-sorganization/UpstreamDrift",
        cmd_runner=runner,
    )
    result = processor.run()
    assert result.created_count == 0
    assert result.bot_count == 2
    runner.assert_not_called()


def test_processor_skips_closed_pr_comments(tmp_path: Path) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    comments = [{"id": 33, "user": {"login": "dev"}, "body": "```suggestion\nx=1\n```"}]
    _write_pr_and_comments(comments_dir, 200, "CLOSED", comments)

    runner = MagicMock()
    processor = CommentToIssueProcessor(
        comments_dir=comments_dir,
        archive_dir=archive_dir,
        repo="D-sorganization/UpstreamDrift",
        cmd_runner=runner,
    )
    result = processor.run()
    assert result.created_count == 0
    assert result.skipped_closed_count == 1
    runner.assert_not_called()


def test_processor_skips_duplicate_issues_stateless(tmp_path: Path) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    comments = [
        {
            "id": 44,
            "user": {"login": "reviewer"},
            "body": "TODO: please ensure thread lock is released properly",
            "path": "src/lock.py",
            "line": 55,
            "created_at": "2026-09-25T00:00:00Z",
            "html_url": "https://github.com/org/repo/pull/300#discussion_r44",
        }
    ]
    _write_pr_and_comments(comments_dir, 300, "OPEN", comments)

    def mock_runner(args: list[str]) -> tuple[int, str, str]:
        if "issue" in args and "list" in args:
            return 0, json.dumps([{"number": 999}]), ""
        return 0, "", ""

    processor = CommentToIssueProcessor(
        comments_dir=comments_dir,
        archive_dir=archive_dir,
        repo="D-sorganization/UpstreamDrift",
        cmd_runner=mock_runner,
    )
    result = processor.run()
    assert result.created_count == 0
    assert result.duplicate_count == 1


def test_processor_creates_issue_for_valid_comment(tmp_path: Path) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    comments = [
        {
            "id": 55,
            "user": {"login": "senior-engineer"},
            "body": "TODO: please ensure thread lock is released properly",
            "path": "src/lock.py",
            "line": 55,
            "created_at": "2026-09-25T00:00:00Z",
            "html_url": "https://github.com/org/repo/pull/400#discussion_r55",
        }
    ]
    _write_pr_and_comments(comments_dir, 400, "OPEN", comments)

    created_args: list[list[str]] = []

    def mock_runner(args: list[str]) -> tuple[int, str, str]:
        if "issue" in args and "list" in args:
            return 0, "[]", ""
        if "issue" in args and "create" in args:
            created_args.append(args)
            return 0, "https://github.com/org/repo/issues/10950\n", ""
        return 0, "", ""

    processor = CommentToIssueProcessor(
        comments_dir=comments_dir,
        archive_dir=archive_dir,
        repo="D-sorganization/UpstreamDrift",
        cmd_runner=mock_runner,
    )
    result = processor.run()
    assert result.created_count == 1
    assert len(created_args) == 1
    assert "[review] Feedback on src/lock.py:55" in created_args[0]

    tracking_file = archive_dir / "comment_tracking.json"
    assert tracking_file.exists()
    tracking_data = json.loads(tracking_file.read_text(encoding="utf-8"))
    assert "55" in tracking_data["processed_comments"]
    assert "55" in tracking_data["created_issues"]


def test_processor_respects_rate_limit(tmp_path: Path) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    comments = [
        {
            "id": i,
            "user": {"login": "senior-engineer"},
            "body": f"TODO: please ensure check {i} is handled in the codebase",
            "path": f"src/mod_{i}.py",
            "line": i,
            "created_at": "2026-09-25T00:00:00Z",
            "html_url": f"https://github.com/org/repo/pull/500#discussion_r{i}",
        }
        for i in range(1, 10)
    ]
    _write_pr_and_comments(comments_dir, 500, "OPEN", comments)

    def mock_runner(args: list[str]) -> tuple[int, str, str]:
        if "issue" in args and "list" in args:
            return 0, "[]", ""
        if "issue" in args and "create" in args:
            return 0, "https://github.com/org/repo/issues/new\n", ""
        return 0, "", ""

    processor = CommentToIssueProcessor(
        comments_dir=comments_dir,
        archive_dir=archive_dir,
        repo="D-sorganization/UpstreamDrift",
        max_issues=3,
        cmd_runner=mock_runner,
    )
    result = processor.run()
    assert result.created_count == 3
    assert result.rate_limited_count == 6


def test_processor_archive_only(tmp_path: Path) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    comments = [
        {"id": 10, "user": {"login": "dev2"}, "body": "TODO: fix this calculation"}
    ]
    _write_pr_and_comments(comments_dir, 1, "OPEN", comments)

    runner = MagicMock()
    processor = CommentToIssueProcessor(
        comments_dir=comments_dir,
        archive_dir=archive_dir,
        archive_only=True,
        cmd_runner=runner,
    )
    result = processor.run()
    assert result.archived_count == 1
    assert result.created_count == 0
    runner.assert_not_called()


def test_main_cli_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    comments_dir, archive_dir = tmp_path / "comments", tmp_path / "archive"
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    comments = [
        {"id": 10, "user": {"login": "dev2"}, "body": "TODO: fix this calculation"}
    ]
    _write_pr_and_comments(comments_dir, 1, "OPEN", comments)

    exit_code = main(
        [
            "--comments-dir",
            str(comments_dir),
            "--archive-dir",
            str(archive_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert output_file.exists()
    assert "created_count=1" in output_file.read_text(encoding="utf-8")
