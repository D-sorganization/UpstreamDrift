#!/usr/bin/env python3
"""Anti-phantom-merge Rule 3: closes-issue path-membership check.

Extracted from inline ``run:`` logic in
``.github/workflows/anti-phantom-merge.yml`` (UD #9091) so the check is
unit-testable without a workflow change.

Changed-file resolution order:

1. Local ``git diff`` against ``git merge-base <base> <head>`` when the
   merge-base computation succeeds (the shallow ``--depth=1`` base fetch
   can defeat it, so failure is expected and handled).
2. A caller-supplied changed-file list (``PR_CHANGED_FILES`` env or
   ``--changed-files``) — the same GitHub API list the count fallback in
   the workflow already fetches.
3. ``gh pr view <n> --repo <r> --json files --jq '.files[].path'`` — the
   API pattern the workflow already uses.
4. Otherwise the check fails closed with a clear diagnostic.

Output contract: informational logs go to *stderr* (``logging``); on
failure the PR-comment-worthy report is printed to *stdout* so the
workflow can forward it to ``fail()``. Exit 0 = pass, 1 = fail closed or
rule violation.
"""

from __future__ import annotations

import argparse
import logging
import os
import posixpath
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ISSUE_REF_PATTERN = re.compile(
    r"(?:Closes|Fixes|Resolves)\s+#([0-9]+)", re.IGNORECASE
)
ISSUE_PATH_PATTERN = re.compile(r"(?:src|tests|rust_core|api)/[A-Za-z0-9_/.-]+")
ENV_CHANGED_FILES = "PR_CHANGED_FILES"
MAX_PATHS_SHOWN = 5

IssueBodyLoader = Callable[[str], "str | None"]
ApiListLoader = Callable[[], "list[str] | None"]


class ChangedFilesUnavailableError(RuntimeError):
    """Raised when neither git diff nor an API file list is available."""


def _repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[2]


def _run_git(args: list[str], repo_root: Path) -> str | None:
    """Run git, returning stdout or None when the command fails."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.debug("git %s failed: %s", args[0], result.stderr.strip())
        return None
    return result.stdout


def _merge_base(base_sha: str, head_sha: str, repo_root: Path) -> str | None:
    """Return the merge-base SHA, or None when git cannot resolve it."""
    out = _run_git(["merge-base", base_sha, head_sha], repo_root)
    if out is None:
        return None
    return out.strip() or None


def _git_changed_files(
    base_sha: str, head_sha: str, repo_root: Path
) -> list[str] | None:
    """Return changed files via local git, or None when unavailable.

    ``None`` distinguishes an unusable source (failed merge-base, e.g.
    after a shallow base fetch) from a valid empty diff, which is a real
    result and must not trigger the API fallback.
    """
    merge_base = _merge_base(base_sha, head_sha, repo_root)
    if merge_base is None:
        return None
    out = _run_git(
        ["diff", "--name-only", f"{merge_base}...{head_sha}"], repo_root
    )
    if out is None:
        return None
    return [line for line in out.splitlines() if line.strip()]


def _env_changed_files(env_value: str | None) -> list[str]:
    """Parse a newline-separated changed-file list from the environment."""
    if not env_value:
        return []
    return [line.strip() for line in env_value.splitlines() if line.strip()]


def _api_changed_files(pr_number: str, repo: str) -> list[str] | None:
    """Return the PR changed-file list via the GitHub CLI, or None."""
    result = subprocess.run(
        [
            "gh", "pr", "view", pr_number, "--repo", repo,
            "--json", "files", "--jq", ".files[].path",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.debug("gh pr view failed: %s", result.stderr.strip())
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def _resolve_changed_files(
    base_sha: str | None,
    head_sha: str | None,
    env_list: str,
    api_loader: ApiListLoader,
) -> tuple[list[str], str]:
    """Resolve the PR changed files; raise when every source is unusable.

    Args:
        base_sha: PR base SHA (empty disables the git source).
        head_sha: PR head SHA.
        env_list: Raw ``PR_CHANGED_FILES`` value (newline-separated).
        api_loader: Zero-arg loader returning the API file list or None.

    Returns:
        Tuple of the changed-file paths and the source name used.

    Raises:
        ChangedFilesUnavailableError: Neither git nor an API list worked.
    """
    if base_sha and head_sha:
        git_files = _git_changed_files(base_sha, head_sha, _repo_root())
        if git_files is not None:
            return git_files, "git"
    env_files = _env_changed_files(env_value=env_list)
    if env_files:
        return env_files, "env"
    api_files = api_loader()
    if api_files is not None:
        return api_files, "api"
    raise ChangedFilesUnavailableError(
        "unable to determine changed files: git merge-base failed (likely a "
        f"shallow base fetch) and no API changed-file list is available "
        f"(env {ENV_CHANGED_FILES} unset/empty and 'gh pr view' failed) — "
        "failing closed."
    )


def _referenced_issue_numbers(pr_body: str) -> list[str]:
    """Return unique issue numbers from Closes/Fixes/Resolves, sorted."""
    return sorted(set(ISSUE_REF_PATTERN.findall(pr_body)), key=int)


def _issue_referenced_paths(issue_body: str) -> list[str]:
    """Return unique repo paths referenced in an issue body, sorted."""
    return sorted(set(ISSUE_PATH_PATTERN.findall(issue_body)))


def _first_path_match(
    changed_files: list[str], referenced: list[str]
) -> str | None:
    """Return the first referenced path touched by the diff, if any.

    Mirrors the original shell semantics: a substring match anywhere in a
    changed path, or a directory-prefix match on the referenced path's
    parent directory.
    """
    for path in referenced:
        if any(path in changed for changed in changed_files):
            return path
        directory = posixpath.dirname(path)
        if directory in ("", ".", "/"):
            continue
        if any(
            changed.startswith(f"{directory}/") for changed in changed_files
        ):
            return directory
    return None


def _load_issue_body(pr_number: str, repo: str) -> str | None:
    """Fetch an issue body via the GitHub CLI; None when unavailable."""
    result = subprocess.run(
        [
            "gh", "issue", "view", pr_number, "--repo", repo,
            "--json", "body", "--jq", ".body",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout or None


def _evaluate_rule3(
    pr_body: str,
    changed_files: list[str],
    issue_body_loader: IssueBodyLoader,
) -> list[str]:
    """Evaluate rule 3; return one failure message per offending issue."""
    failures: list[str] = []
    for number in _referenced_issue_numbers(pr_body):
        logger.info("Checking referenced issue #%s", number)
        issue_body = issue_body_loader(number)
        if not issue_body:
            logger.info("  (issue #%s inaccessible or empty — skipping)", number)
            continue
        referenced = _issue_referenced_paths(issue_body)
        if not referenced:
            logger.info("  (issue #%s has no path references — skipping)", number)
            continue
        match = _first_path_match(changed_files, referenced)
        if match is None:
            summary = "\n".join(f"    {p}" for p in referenced[:MAX_PATHS_SHOWN])
            failures.append(
                f"PR closes issue #{number}, but the diff touches none of the "
                f"paths referenced in that issue's body:\n{summary}\n"
                "If the PR is intentionally addressing a different part of "
                f"the issue, remove the closes keyword from the PR body."
            )
        else:
            logger.info("  matched referenced path: %s", match)
    return failures


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, defaulting from the workflow's environment."""
    parser = argparse.ArgumentParser(
        description="Anti-phantom-merge Rule 3: closes-issue path check."
    )
    parser.add_argument("--base-sha", default=os.environ.get("BASE_SHA"))
    parser.add_argument("--head-sha", default=os.environ.get("HEAD_SHA"))
    parser.add_argument("--pr-number", default=os.environ.get("PR_NUMBER"))
    parser.add_argument("--repo", default=os.environ.get("REPO"))
    parser.add_argument("--pr-body", default=os.environ.get("PR_BODY", ""))
    parser.add_argument(
        "--changed-files",
        default=os.environ.get(ENV_CHANGED_FILES, ""),
        help=f"Optional newline-separated API changed-file list "
        f"(env: {ENV_CHANGED_FILES}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the rule-3 check; return 0 on pass, 1 on any failure."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = args.repo or ""

    def api_loader() -> list[str] | None:
        if not args.pr_number or not repo:
            return None
        return _api_changed_files(args.pr_number, repo)

    try:
        changed_files, source = _resolve_changed_files(
            args.base_sha, args.head_sha, args.changed_files, api_loader
        )
    except ChangedFilesUnavailableError as exc:
        logger.error("FAIL: %s", exc)
        print(f"FAIL: {exc}")
        return 1
    logger.info("changed-file source: %s (%d files)", source, len(changed_files))

    failures = _evaluate_rule3(
        args.pr_body, changed_files, lambda n: _load_issue_body(n, repo)
    )
    if failures:
        report = "\n\n".join(failures)
        logger.error("FAIL: rule 3 (closes-issue path mismatch):\n%s", report)
        print(report)
        return 1
    logger.info("Rule 3 (closes-issue path membership) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())