#!/usr/bin/env python3
"""Run pre-commit pre-push hooks against the actual push range.

Git sends pre-push hooks one line per ref in this form:

    <local-ref> <local-sha> <remote-ref> <remote-sha>

For new branches the remote SHA is all zeroes. The stock pre-commit hook cannot
derive a useful range in that case, so it may fall back to broad file selection.
This wrapper resolves a base explicitly and invokes pre-commit with
``--from-ref`` / ``--to-ref``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ZERO_SHAS = {"0" * 40, "0" * 64}
MAINLINE_REFS = ("refs/heads/main", "refs/heads/master", "main", "master")
BASE_CANDIDATES = ("origin/main", "origin/HEAD", "main", "origin/master", "master")


@dataclass(frozen=True)
class PushRef:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _git_output(args: Sequence[str]) -> str | None:
    result = _run(["git", *args])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def is_zero_sha(sha: str) -> bool:
    return sha in ZERO_SHAS


def parse_push_refs(stdin: Iterable[str]) -> list[PushRef]:
    refs: list[PushRef] = []
    for line in stdin:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        refs.append(PushRef(*parts[:4]))
    return refs


def current_head_ref() -> PushRef | None:
    head = _git_output(["rev-parse", "--verify", "HEAD"])
    if head is None:
        return None
    return PushRef("HEAD", head, "", "")


def merge_base(local_sha: str, base_candidate: str) -> str | None:
    return _git_output(["merge-base", local_sha, base_candidate])


def upstream_ref() -> str | None:
    return _git_output(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]
    )


def resolve_from_ref(push_ref: PushRef) -> str | None:
    """Resolve the left side of the pre-push diff.

    For pushes directly to mainline (main/master), use the remote SHA provided by
    Git so only new commits pushed to mainline are checked.
    For feature branches, resolve the merge-base with mainline so changes are scoped
    to the branch's net diff against mainline, ignoring commits and unformatted files
    introduced via a merge of main (Repository_Management#1686).
    Falls back to remote_sha, branch upstream, or merge-base with default refs.
    """
    is_mainline_push = (
        push_ref.remote_ref in MAINLINE_REFS or push_ref.local_ref in MAINLINE_REFS
    )
    if (
        is_mainline_push
        and push_ref.remote_sha
        and not is_zero_sha(push_ref.remote_sha)
    ):
        return push_ref.remote_sha

    # For feature branches, prefer merge-base with mainline to exclude
    # merge-of-main commits.
    for candidate in BASE_CANDIDATES:
        base = merge_base(push_ref.local_sha, candidate)
        if base:
            return base

    # Fall back to remote SHA if available
    if push_ref.remote_sha and not is_zero_sha(push_ref.remote_sha):
        return push_ref.remote_sha

    upstream = upstream_ref()
    if upstream:
        base = merge_base(push_ref.local_sha, upstream)
        if base:
            return base

    return None


def run_pre_commit(from_ref: str, to_ref: str) -> int:
    env = os.environ.copy()
    env["FLEET_HOOK_FROM_REF"] = from_ref
    env["FLEET_HOOK_TO_REF"] = to_ref
    cmd = [
        "pre-commit",
        "run",
        "--hook-stage",
        "pre-push",
        "--from-ref",
        from_ref,
        "--to-ref",
        to_ref,
    ]
    return subprocess.run(cmd, env=env, check=False).returncode


def main(argv: Sequence[str] | None = None, stdin: Iterable[str] | None = None) -> int:
    _ = argv if argv is not None else sys.argv[1:]
    refs = parse_push_refs(stdin if stdin is not None else sys.stdin)
    if not refs:
        head_ref = current_head_ref()
        refs = [head_ref] if head_ref else []

    status = 0
    for push_ref in refs:
        if is_zero_sha(push_ref.local_sha):
            continue
        from_ref = resolve_from_ref(push_ref)
        if from_ref is None:
            print(
                "pre-push: unable to resolve a diff base; skipping local hooks "
                "instead of scanning the whole repository.",
                file=sys.stderr,
            )
            continue
        status |= run_pre_commit(from_ref, push_ref.local_sha)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
