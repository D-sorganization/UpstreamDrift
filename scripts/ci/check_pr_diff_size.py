#!/usr/bin/env python3
"""Enforce fleet PR size cap (lane rules: 500 changed lines unless exempt).

Counts ``git diff --numstat`` added + deleted lines between base and head.
Exemptions:
- Deletion-only PRs (zero lines added across the diff).
- PRs labeled ``golden-regen`` (GitHub event file only).

See issue #14741.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


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


def _pr_labels_from_event() -> frozenset[str]:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path:
        return frozenset()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    pr = payload.get("pull_request")
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


def check_pr_diff_size(
    repo_root: Path,
    base_sha: str,
    head_sha: str,
    *,
    max_lines: int,
) -> str | None:
    """Return an error message string when the cap is exceeded, else ``None``."""
    total, total_added, total_deleted = diff_line_total(repo_root, base_sha, head_sha)
    if total <= max_lines:
        return None
    if exempt_deletion_only(total_added, total_deleted):
        return None
    if "golden-regen" in _pr_labels_from_event():
        return None
    return (
        f"PR diff exceeds {max_lines} changed lines (added+deleted): "
        f"{total} ({total_added} added, {total_deleted} deleted); "
        "split the change, apply for a deletion-only PR, or label golden-regen."
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--max-lines", type=int, default=500)
    args = parser.parse_args()
    err = check_pr_diff_size(
        args.repo_root.resolve(),
        args.base_sha,
        args.head_sha,
        max_lines=args.max_lines,
    )
    if err:
        print(err, file=sys.stderr)
        return 1
    print(
        "pr-size gate: within limit",
        f"(base {args.base_sha[:8]}..head {args.head_sha[:8]})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
