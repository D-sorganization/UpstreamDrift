"""Post the SPEC.md update reminder on a pull request, fail-safe by design.

Extracted from the inline `actions/github-script` heredoc in
`.github/workflows/spec-check.yml` (issue #9499). The heredoc once aborted with
`SyntaxError: Invalid or unexpected token` after an unescaped backtick landed
in the template literal, so a stale-SPEC finding never reached the contributor.

Contract:
- When posting succeeds (or a bot reminder already exists), exit 0 and let the
  workflow's "Fail if spec is stale" step produce the canonical error.
- When posting fails for ANY reason (network, permissions, token), print the
  full diagnostic to stdout — i.e. the job log — and still exit 0. The gate's
  outcome belongs to the staleness check, never to the reporting step; the
  workflow's fail step is `always()`-guarded so the job still exits non-zero.

Usage (from the workflow):

    python3 scripts/post_spec_reminder.py \
        --repository "${{ github.repository }}" \
        --issue-number "${{ github.event.pull_request.number }}" \
        --policy-url "<policy markdown url>"

Authentication uses the `GITHUB_TOKEN` (or `GH_TOKEN`) environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

COMMENT_MARKER = "SPEC.md Update Required"
FALLBACK_POLICY_URL = (
    "https://github.com/D-sorganization/Repository_Management/blob/main/"
    "docs/SPECIFICATION_DRIVEN_DEVELOPMENT_POLICY.md"
)
_API_BASE = "https://api.github.com"
_REPOSITORY_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")


def build_reminder_body(policy_url: str) -> str:
    """Build the reminder markdown posted onto the pull request.

    Args:
        policy_url: Absolute URL of the Specification-Driven Development
            Policy document linked from the reminder.

    Returns:
        The reminder body keyed on ``COMMENT_MARKER`` so repeat runs can
        detect and suppress duplicates.
    """
    return f"""## -- {COMMENT_MARKER}

Source files were modified in this PR but `SPEC.md` was not updated.

**Per the [Specification-Driven Development Policy]({policy_url}):**
- Any PR that adds, removes, or changes functionality must update SPEC.md
- Update the relevant sections (features, architecture, tests, dependencies, etc.)
- Add ONE change-log row keyed by this PR: `| YYYY-MM-DD | #<pr> | summary |`
- Do NOT bump the `Spec Version` field and do NOT put a serial
  version in the row: see Repository_Management#1520

**To resolve:**
1. Update `SPEC.md` to reflect your changes, OR
2. Add the `spec-exempt` label if this PR genuinely doesn't affect the spec (e.g., pure refactor with no behavior change)

*This check is enforced by the spec-check CI workflow.*"""


def _token_from_env() -> str:
    """Read the API token from the environment.

    Returns:
        The token string, possibly empty when no variable is set.

    Raises:
        ValueError: If no token is configured; the GitHub API requires one.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        raise ValueError(
            "No API token available: set the GITHUB_TOKEN (or GH_TOKEN) "
            "environment variable"
        )
    return token


def _validate_repository(repository: str) -> str:
    """Validate an ``owner/name`` repository identifier.

    Args:
        repository: Repository identifier in GitHub ``owner/name`` form.

    Returns:
        The validated identifier.

    Raises:
        ValueError: If the identifier does not match ``owner/name``.
    """
    if not _REPOSITORY_PATTERN.match(repository):
        raise ValueError(
            f"--repository must be in 'owner/name' form, got {repository!r}"
        )
    return repository


def _validate_issue_number(issue_number: int) -> int:
    """Validate a pull request or issue number.

    Args:
        issue_number: GitHub issue/PR number.

    Returns:
        The validated number.

    Raises:
        ValueError: If the number is not a positive integer.
    """
    if issue_number <= 0:
        raise ValueError(f"--issue-number must be positive, got {issue_number}")
    return issue_number


def _request(method: str, path: str, token: str, payload: Any = None) -> Any:
    """Call the GitHub REST API and return the parsed ``data`` envelope.

    Args:
        method: HTTP verb, ``GET`` or ``POST``.
        path: API path beginning with ``/``.
        token: Bearer token for authentication.
        payload: JSON request body for ``POST`` calls.

    Returns:
        The parsed JSON response body.

    Raises:
        RuntimeError: On a non-2xx response or transport failure.
    """
    request = urllib.request.Request(
        f"{_API_BASE}{path}",
        data=json.dumps(payload).encode("utf-8") if payload else None,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"GitHub API {method} {path} failed: {error} {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"GitHub API {method} {path} unreachable: {error}"
        ) from error


def _has_bot_reminder(comments: list[dict[str, Any]]) -> bool:
    """Check whether a bot reminder comment already exists.

    Args:
        comments: Issue comment objects as returned by the REST API.

    Returns:
        True when any bot-authored comment carries ``COMMENT_MARKER``.
    """
    return any(
        COMMENT_MARKER in (comment.get("body") or "")
        and (comment.get("user") or {}).get("type") == "Bot"
        for comment in comments
    )


def _post_reminder(repository: str, issue_number: int, policy_url: str) -> bool:
    """Post the reminder unless a bot comment already exists.

    Args:
        repository: ``owner/name`` repository identifier.
        issue_number: Pull request number.
        policy_url: Policy document URL linked from the reminder.

    Returns:
        True when a comment was posted, False when one already existed.

    Raises:
        ValueError: On malformed input boundaries.
        RuntimeError: On any API or transport failure.
    """
    token = _token_from_env()
    comments_path = f"/repos/{repository}/issues/{issue_number}/comments"
    existing = _request("GET", comments_path, token)
    if _has_bot_reminder(existing.get("data", [])):
        print(f"A {COMMENT_MARKER} reminder already exists; nothing to do.")
        return False
    body = build_reminder_body(policy_url)
    _request(
        "POST",
        comments_path,
        token,
        {"body": body, "issue_number": issue_number},
    )
    print(f"Posted the {COMMENT_MARKER} reminder to {repository}#{issue_number}.")
    return True


def _print_diagnostic(repository: str, issue_number: int, failure: Exception) -> None:
    """Echo the full diagnostic to stdout so the finding reaches the job log."""
    print(
        f"::error::Could not post the {COMMENT_MARKER} reminder to "
        f"{repository}#{issue_number}: {failure}"
    )
    print("--- Diagnostic follows (intended PR comment body) ---")
    print(build_reminder_body(FALLBACK_POLICY_URL))
    print(
        "--- End diagnostic: the SPEC.md staleness gate below still fails this run ---"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: post the reminder, never lose the finding.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Always 0: reporting failures are printed to the
        job log while the workflow's `always()`-guarded staleness step owns
        the non-zero exit.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", required=True, help="Repository as owner/name")
    parser.add_argument(
        "--issue-number", required=True, type=int, help="Pull request number"
    )
    parser.add_argument("--policy-url", default=FALLBACK_POLICY_URL, help="Policy URL")
    args = parser.parse_args(argv)

    repository = _validate_repository(args.repository)
    issue_number = _validate_issue_number(args.issue_number)
    try:
        _post_reminder(repository, issue_number, args.policy_url)
    except Exception as failure:  # noqa: BLE001 - fail-safe reporting boundary
        _print_diagnostic(repository, issue_number, failure)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
