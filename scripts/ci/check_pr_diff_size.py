#!/usr/bin/env python3
"""Report PR changed-line count vs a reference cap (#14741, RD #4118 parity).

Counts ``git diff --numstat`` added + deleted lines between ``base_sha`` and
``head_sha`` (triple-dot / merge-base semantics). By default this exits **0**
and emits GitHub notices/warnings when over the reference cap. Prefer small,
cohesive PRs; do **not** split a cohesive fix only to stay under a line count.

Optional bookkeeping (still advisory — not merge blockers):

- Deletion-only PRs (zero lines added) — reported as exempt.
- Label ``golden-regen`` on the pull request (from ``GITHUB_EVENT_PATH``).
- Allowlist file ``config/pr_size_allowlist.json`` listing PR numbers.

Use ``--strict`` for enforcement (exit 1 when over cap and not exempt), e.g.
local policy or tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ALLOWLIST_PATH = Path("config/pr_size_allowlist.json")


def parse_numstat(line: str) -> tuple[int, int, str] | None:
    """Return ``(added, deleted, path)`` for a ``git diff --numstat`` line."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        return None
    added_s, deleted_s, path = parts
    if added_s == "-" or deleted_s == "-":
        return None
    return int(added_s), int(deleted_s), path


def diff_line_total(
    repo_root: Path, base_sha: str, head_sha: str
) -> tuple[int, int, int]:
    """Return ``(total_changed, total_added, total_deleted)`` for the diff."""
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--numstat",
            f"{base_sha}...{head_sha}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total_added = total_deleted = 0
    for raw in proc.stdout.splitlines():
        parsed = parse_numstat(raw)
        if parsed is None:
            continue
        a, d, _path = parsed
        total_added += a
        total_deleted += d
    return total_added + total_deleted, total_added, total_deleted


def exempt_deletion_only(total_added: int, total_deleted: int) -> bool:
    """True when the diff adds nothing (pure deletion PR)."""
    return total_added == 0 and total_deleted > 0


def _load_event() -> dict:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def pr_number_from_event() -> int | None:
    """Return pull_request.number from the GitHub Actions event payload, if any."""
    payload = _load_event()
    pr = payload.get("pull_request")
    if isinstance(pr, dict) and isinstance(pr.get("number"), int):
        return int(pr["number"])
    return None


def _pr_labels_from_event() -> frozenset[str]:
    pr = _load_event().get("pull_request")
    if not isinstance(pr, dict):
        return frozenset()
    labels = pr.get("labels")
    if not isinstance(labels, list):
        return frozenset()
    names: list[str] = []
    for item in labels:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return frozenset(names)


def _load_allowlist(path: Path) -> frozenset[int]:
    if not path.exists():
        return frozenset()
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(int(x) for x in data.get("allowed_prs", []))


def _github_output_write(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


@dataclass(frozen=True)
class PrSizeVerdict:
    """Outcome of comparing a PR diff to the reference cap."""

    within_limit: bool
    total: int
    total_added: int
    total_deleted: int
    exempt_reason: str | None


def assess_pr_diff_size(
    repo_root: Path,
    base_sha: str,
    head_sha: str,
    *,
    max_lines: int,
    pr_number: int | None,
    allowlist_path: Path,
) -> PrSizeVerdict:
    """Classify the diff vs ``max_lines``; exemptions are advisory bookkeeping only."""
    total, total_added, total_deleted = diff_line_total(repo_root, base_sha, head_sha)
    if total <= max_lines:
        return PrSizeVerdict(True, total, total_added, total_deleted, None)
    if exempt_deletion_only(total_added, total_deleted):
        return PrSizeVerdict(True, total, total_added, total_deleted, "deletion-only")
    if "golden-regen" in _pr_labels_from_event():
        return PrSizeVerdict(True, total, total_added, total_deleted, "golden-regen")
    if pr_number is not None and pr_number in _load_allowlist(allowlist_path):
        return PrSizeVerdict(
            True, total, total_added, total_deleted, f"allowlisted PR #{pr_number}"
        )
    return PrSizeVerdict(False, total, total_added, total_deleted, None)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR number (defaults to pull_request.number from GITHUB_EVENT_PATH)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help=f"path to allowlist JSON (default: <repo-root>/{DEFAULT_ALLOWLIST_PATH})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when over the cap and not exempt (optional enforcement)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_root = args.repo_root.resolve()
    allowlist = args.allowlist or (repo_root / DEFAULT_ALLOWLIST_PATH)
    pr_number = args.pr if args.pr is not None else pr_number_from_event()

    verdict = assess_pr_diff_size(
        repo_root,
        args.base_sha,
        args.head_sha,
        max_lines=args.max_lines,
        pr_number=pr_number,
        allowlist_path=allowlist,
    )

    advisory = not verdict.within_limit
    _github_output_write("within_limit", "true" if verdict.within_limit else "false")
    _github_output_write("advisory", "true" if advisory else "false")
    _github_output_write("total_lines", str(verdict.total))

    if verdict.within_limit:
        detail = (
            f"{verdict.total} changed lines (<= {args.max_lines}); "
            f"{verdict.total_added} added, {verdict.total_deleted} deleted"
        )
        if verdict.exempt_reason:
            detail += f" — exempt ({verdict.exempt_reason})"
        print(detail)
        return 0

    msg = (
        f"PR size advisory: {verdict.total} changed lines (reference cap {args.max_lines}); "
        f"{verdict.total_added} added, {verdict.total_deleted} deleted. "
        "Prefer small cohesive PRs; do not split a cohesive fix only to hit a line count."
    )
    print(msg)
    print(f"::warning::{msg}")
    if args.strict:
        print(msg, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry for tests (``sys.argv`` override via ``argv``)."""
    if argv is not None:
        old = sys.argv
        sys.argv = [old[0], *argv]
        try:
            return _main()
        finally:
            sys.argv = old
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
