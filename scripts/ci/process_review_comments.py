#!/usr/bin/env python3
"""Convert PR review comments into actionable GitHub issues with strict deduplication and bot filtering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable

logger = logging.getLogger(__name__)

KNOWN_BOT_LOGINS: frozenset[str] = frozenset(
    {
        "chatgpt-codex-connector",
        "copilot-pull-request-reviewer",
        "github-actions",
        "dependabot",
        "renovate",
        "jules-bot",
        "jules-ai",
        "google-labs-jules",
        "cursor",
        "bugbot",
        "codecov",
        "linear",
    }
)

DIRECTIVE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:^|\s)(?:TODO:|FIXME:|ACTION:|Action item:|/create-issue|/issue)(?:\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:please fix|needs? to be fixed|must be (?:fixed|updated|changed)|"
        r"please (?:update|change|remove|add)|"
        r"should be (?:refactored|extracted|handled|fixed)|"
        r"can we (?:track|fix|address) this)\b",
        re.IGNORECASE,
    ),
)

TRIVIAL_PHRASES: frozenset[str] = frozenset(
    {
        "lgtm",
        "lgtm!",
        "looks good",
        "looks good to me",
        "thanks",
        "thanks!",
        "thank you",
        "nice",
        "+1",
        "done",
        "fixed",
        "resolved",
        "approved",
    }
)

SubprocessRunner = Callable[[list[str]], tuple[int, str, str]]


def default_subprocess_runner(args: list[str]) -> tuple[int, str, str]:
    """Execute a subprocess command and return exit code, stdout, stderr."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except OSError as exc:
        logger.error("Subprocess execution failed for command %s: %s", args, exc)
        return 1, "", str(exc)


def is_bot_user(user_info: dict[str, Any] | None, author_login: str) -> bool:
    """Return True if the author is an automated bot or service account."""
    clean_login = author_login.strip().lower()
    if clean_login.endswith("[bot]") or clean_login in KNOWN_BOT_LOGINS:
        return True
    return bool(user_info and user_info.get("type", "").strip().lower() == "bot")


def is_pr_open(pr_details: dict[str, Any]) -> bool:
    """Verify if the PR is currently open."""
    return pr_details.get("state", "").upper() == "OPEN"


def is_actionable_comment(body: str) -> bool:
    """Determine if a review comment has actionable suggestions or directives."""
    text = body.strip()
    if len(text) < 20 or text.lower() in TRIVIAL_PHRASES:
        return False
    if "```suggestion" in text:
        return True
    return any(regex.search(text) for regex in DIRECTIVE_REGEXES)


def format_issue_title(file_path: str, line: int) -> str:
    """Format issue title within 100 characters limit."""
    title = f"[review] Feedback on {file_path}:{line}"
    return title[:97] + "..." if len(title) > 100 else title


@dataclass
class ReviewCommentContext:
    """Contextual metadata describing a review comment and its source PR."""

    pr_num: str
    pr_title: str
    pr_author: str
    pr_branch: str
    author: str
    file_path: str
    line: int
    comment_id: str
    html_url: str
    created_at: str
    feedback_body: str


def format_issue_body(ctx: ReviewCommentContext) -> str:
    """Format the GitHub issue body with embedded metadata marker."""
    return f"""<!-- comment-to-issue: comment_id={ctx.comment_id} pr={ctx.pr_num} -->
## Review Comment from @{ctx.author}

**Source PR:** #{ctx.pr_num} - {ctx.pr_title}
**File:** `{ctx.file_path}` (line {ctx.line})
**Original Comment:** {ctx.html_url}

---

### Feedback

{ctx.feedback_body}

---

### Context

- **PR Author:** @{ctx.pr_author}
- **Branch:** `{ctx.pr_branch}`
- **Reviewed by:** @{ctx.author}
- **Date:** {ctx.created_at}

---

*This issue was auto-generated from a PR review comment.*"""


@dataclass
class ProcessResult:
    """Summary of comment processing operations."""

    total_comments: int = 0
    archived_count: int = 0
    created_count: int = 0
    bot_count: int = 0
    skipped_closed_count: int = 0
    duplicate_count: int = 0
    rate_limited_count: int = 0
    created_issues: list[dict[str, str]] = field(default_factory=list)


class CommentToIssueProcessor:
    """Processes review comments, applies safety gates, and manages issue generation."""

    def __init__(
        self,
        comments_dir: Path,
        archive_dir: Path,
        repo: str = "",
        max_issues: int = 5,
        archive_only: bool = False,
        dry_run: bool = False,
        cmd_runner: SubprocessRunner | None = None,
    ) -> None:
        self.comments_dir = comments_dir
        self.archive_dir = archive_dir
        self.repo = repo
        self.max_issues = max_issues
        self.archive_only = archive_only
        self.dry_run = dry_run
        self.cmd_runner = cmd_runner or default_subprocess_runner

    def _load_json_file(self, path: Path, default: Any) -> Any:
        if not path.exists() or path.stat().st_size == 0:
            return default
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load JSON file %s: %s", path, exc)
            return default

    def _issue_exists_remotely(self, comment_id: str) -> bool:
        marker = f"comment-to-issue: comment_id={comment_id}"
        args = ["gh", "issue", "list"]
        if self.repo:
            args.extend(["--repo", self.repo])
        args.extend(
            [
                "--search",
                marker,
                "--state",
                "all",
                "--json",
                "number",
                "--limit",
                "1",
            ]
        )
        returncode, stdout, stderr = self.cmd_runner(args)
        if returncode != 0:
            logger.warning("Remote issue query failed (%s): %s", returncode, stderr)
            return False
        try:
            found = json.loads(stdout.strip())
            return bool(isinstance(found, list) and len(found) > 0)
        except json.JSONDecodeError:
            return False

    def _dispatch_issue_create(self, title: str, body: str) -> tuple[int, str, str]:
        args = ["gh", "issue", "create"]
        if self.repo:
            args.extend(["--repo", self.repo])
        args.extend(
            [
                "--title",
                title,
                "--body",
                body,
                "--label",
                "review-feedback,auto-generated",
            ]
        )
        return self.cmd_runner(args)

    def _evaluate_issue_candidate(
        self,
        ctx: ReviewCommentContext,
        actionable: bool,
        tracking: dict[str, Any],
        result: ProcessResult,
    ) -> None:
        if not actionable or self.archive_only:
            return
        if ctx.comment_id in tracking.get("created_issues", {}):
            result.duplicate_count += 1
            return
        if self._issue_exists_remotely(ctx.comment_id):
            tracking["created_issues"][ctx.comment_id] = "remote-existing"
            result.duplicate_count += 1
            return
        if result.created_count >= self.max_issues:
            result.rate_limited_count += 1
            return

        issue_title = format_issue_title(ctx.file_path, ctx.line)
        if self.dry_run:
            result.created_count += 1
            return

        code, stdout, stderr = self._dispatch_issue_create(
            issue_title, format_issue_body(ctx)
        )
        if code == 0:
            url = stdout.strip()
            result.created_count += 1
            result.created_issues.append(
                {"comment_id": ctx.comment_id, "issue_url": url}
            )
            tracking.setdefault("created_issues", {})[ctx.comment_id] = url
        else:
            logger.error("Failed to create issue: %s", stderr)
        if ctx.comment_id not in tracking.setdefault("processed_comments", []):
            tracking["processed_comments"].append(ctx.comment_id)

    def _process_single_comment(
        self,
        comment: dict[str, Any],
        pr_meta: tuple[str, str, str, str],
        tracking: dict[str, Any],
        result: ProcessResult,
    ) -> dict[str, Any] | None:
        result.total_comments += 1
        comment_id = str(comment.get("id", ""))
        user_info = comment.get("user")
        author = user_info.get("login", "unknown") if user_info else "unknown"

        if is_bot_user(user_info, author):
            result.bot_count += 1
            return None

        pr_num, pr_title, pr_author, pr_branch = pr_meta
        body = comment.get("body", "")
        file_path = comment.get("path", "")
        line = comment.get("line") or comment.get("original_line") or 0
        actionable = is_actionable_comment(body)

        ctx = ReviewCommentContext(
            pr_num=pr_num,
            pr_title=pr_title,
            pr_author=pr_author,
            pr_branch=pr_branch,
            author=author,
            file_path=file_path,
            line=line,
            comment_id=comment_id,
            html_url=comment.get("html_url", ""),
            created_at=comment.get("created_at", ""),
            feedback_body=body,
        )
        self._evaluate_issue_candidate(ctx, actionable, tracking, result)
        return {
            "pr_number": pr_num,
            "pr_title": pr_title,
            "comment_id": comment_id,
            "author": author,
            "file": file_path,
            "line": line,
            "body": body,
            "url": ctx.html_url,
            "created_at": ctx.created_at,
            "has_suggestion": "```suggestion" in body,
            "is_actionable": actionable,
        }

    def _process_pr_file(
        self,
        details_file: Path,
        tracking: dict[str, Any],
        result: ProcessResult,
    ) -> list[dict[str, Any]]:
        match = re.search(r"pr_(\d+)_details\.json", details_file.name)
        if not match:
            return []
        pr_num = match.group(1)
        pr_details = self._load_json_file(details_file, {})
        if not pr_details:
            return []

        rf = self.comments_dir / f"pr_{pr_num}_review_comments.json"
        if not is_pr_open(pr_details):
            result.skipped_closed_count += len(self._load_json_file(rf, []))
            return []

        pr_meta = (
            pr_num,
            pr_details.get("title", f"PR #{pr_num}"),
            pr_details.get("author", {}).get("login", "unknown"),
            pr_details.get("headRefName", "unknown"),
        )
        entries: list[dict[str, Any]] = []
        for c in self._load_json_file(rf, []):
            entry = self._process_single_comment(c, pr_meta, tracking, result)
            if entry is not None:
                entries.append(entry)
        return entries

    def run(self) -> ProcessResult:
        result = ProcessResult()
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        tracking_file = self.archive_dir / "comment_tracking.json"
        tracking = self._load_json_file(
            tracking_file, {"processed_comments": [], "created_issues": {}}
        )

        archived: list[dict[str, Any]] = []
        for df in sorted(self.comments_dir.glob("pr_*_details.json")):
            archived.extend(self._process_pr_file(df, tracking, result))

        result.archived_count = len(archived)
        if archived and not self.dry_run:
            self._write_archive_markdown(archived)
            try:
                with tracking_file.open("w", encoding="utf-8") as handle:
                    json.dump(tracking, handle, indent=2)
            except OSError as exc:
                logger.error("Failed to write tracking ledger: %s", exc)

        return result

    def _write_archive_markdown(self, comments: list[dict[str, Any]]) -> None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        archive_md = self.archive_dir / f"review_comments_{date_str}.md"
        try:
            with archive_md.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n## Review Comments Session - {datetime.now().isoformat()}\n\n"
                )
                for c in comments:
                    handle.write(
                        f"### PR #{c['pr_number']}: {c['file']}:{c['line']}\n\n"
                        f"- Author: @{c['author']}\n"
                        f"- Actionable: {'Yes' if c['is_actionable'] else 'No'}\n"
                        f"- Suggestion: {'Yes' if c['has_suggestion'] else 'No'}\n"
                        f"- URL: {c['url']}\n\n"
                        f"```\n{c['body']}\n```\n\n---\n\n"
                    )
        except OSError as exc:
            logger.error("Failed to write archive markdown %s: %s", archive_md, exc)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert PR review comments into GitHub issues."
    )
    parser.add_argument(
        "--comments-dir",
        type=Path,
        default=Path(".github/review_comments"),
        help="Directory containing review comments and PR details JSON files",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("docs/review_archive"),
        help="Directory for archiving comments and tracking ledger",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in owner/repo format",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=int(os.environ.get("MAX_ISSUES_PER_RUN", "5")),
        help="Maximum issues allowed to create per run",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Archive comments without creating GitHub issues",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate issue creation without modifying GitHub state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)

    processor = CommentToIssueProcessor(
        comments_dir=args.comments_dir,
        archive_dir=args.archive_dir,
        repo=args.repo,
        max_issues=args.max_issues,
        archive_only=args.archive_only,
        dry_run=args.dry_run,
    )

    result = processor.run()
    logger.info(
        "Processed: total=%d, archived=%d, created=%d, bots=%d, closed=%d, duplicates=%d, rate_limited=%d",
        result.total_comments,
        result.archived_count,
        result.created_count,
        result.bot_count,
        result.skipped_closed_count,
        result.duplicate_count,
        result.rate_limited_count,
    )

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        try:
            with open(output_path, "a", encoding="utf-8") as out:
                out.write(f"archived_count={result.archived_count}\n")
                out.write(f"created_count={result.created_count}\n")
                out.write(f"bot_count={result.bot_count}\n")
                out.write(f"duplicate_count={result.duplicate_count}\n")
        except OSError as exc:
            logger.warning("Failed to append to GITHUB_OUTPUT: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
