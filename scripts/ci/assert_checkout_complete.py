#!/usr/bin/env python3
"""Fail fast when a CI checkout is missing tracked files (#9507).

``actions/checkout`` can exit 0 on a self-hosted runner whose ``_work`` state
is damaged (for example an empty sparse-checkout pattern set that silently
prunes the tree), and the job then dies later with a confusing secondary error
such as ``npm ci can only install with an existing package-lock.json``. This
helper runs right after checkout, compares the index with the working tree and
reports an explicit "incomplete checkout" error naming the runner. It always
prints the effective sparse-checkout state so the next occurrence can be
diagnosed from the log alone.

Contract
--------
Preconditions:
    ``--root`` is a git working tree; ``--max-report`` is a positive integer.
Postconditions:
    Exit 0: every tracked, non-gitlink path that is not intentionally excluded
    by sparse-checkout (skip-worktree) exists in the working tree, and sparse
    checkout is not enabled with an empty pattern set that pruned files.
    Exit 1: the checkout is incomplete (reason and runner name printed).
    Exit 2: ``--root`` is not a git working tree or git could not be run.

Intentional sparse checkouts are respected: index entries carrying the
skip-worktree bit are excluded from verification. Gitlinks (submodules) are
skipped because CI checkouts do not populate them.

Runbook: no runner-corruption runbook exists in this repository; see
issue #9507 for the diagnosis and the drain-the-runner follow-up.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

DEFAULT_MAX_REPORT = 20
_GITLINK_MODE = "160000"


class SparseState(NamedTuple):
    """Effective sparse-checkout configuration of a working tree."""

    enabled: bool
    patterns: tuple[str, ...]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def sparse_state(root: Path) -> SparseState:
    """Return whether sparse checkout is enabled and its pattern list."""
    cfg = _git(root, "config", "--type=bool", "core.sparseCheckout")
    enabled = cfg.returncode == 0 and cfg.stdout.strip() == "true"
    if not enabled:
        return SparseState(False, ())
    listing = _git(root, "sparse-checkout", "list")
    patterns = (
        tuple(line for line in listing.stdout.splitlines() if line.strip())
        if listing.returncode == 0
        else ()
    )
    return SparseState(True, patterns)


def _tracked_entries(root: Path) -> list[tuple[str, bool]]:
    """Return ``(path, skip_worktree)`` for tracked non-gitlink entries."""
    proc = _git(root, "ls-files", "-z", "-t", "-s")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    entries: list[tuple[str, bool]] = []
    for record in proc.stdout.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        tag, mode = meta.split()[0], meta.split()[1]
        if mode == _GITLINK_MODE:
            continue
        entries.append((path, tag == "S"))
    return entries


def find_missing(root: Path) -> list[str]:
    """Return tracked paths absent from the working tree.

    Paths excluded by sparse-checkout (skip-worktree) are not reported.
    """
    assert root.is_dir(), f"root must be a directory: {root}"
    return [
        path
        for path, skipped in _tracked_entries(root)
        if not skipped and not os.path.lexists(root / path)
    ]


def _skipped_count(root: Path) -> int:
    return sum(1 for _, skipped in _tracked_entries(root) if skipped)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _emit(message: str) -> None:
    print(message)  # noqa: T201 - intentional CI log output


def main(argv: list[str] | None = None) -> int:
    """Run the checkout-completeness assertion; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--max-report", type=_positive_int, default=DEFAULT_MAX_REPORT)
    args = parser.parse_args(argv)

    runner = os.environ.get("RUNNER_NAME") or "unknown"
    root: Path = args.root
    probe = _git(root, "rev-parse", "--is-inside-work-tree")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        _emit(f"::error::{root} is not a git working tree (runner {runner})")
        return 2

    state = sparse_state(root)
    if state.enabled:
        shown = ", ".join(state.patterns) if state.patterns else "<empty>"
        _emit(f"sparse-checkout: enabled; patterns: {shown}")
    else:
        _emit("sparse-checkout: disabled")

    missing = find_missing(root)
    pruned_by_empty_set = (
        state.enabled and not state.patterns and _skipped_count(root) > 0
    )
    if not missing and not pruned_by_empty_set:
        _emit(f"checkout complete on runner {runner}")
        return 0

    _emit(
        f"::error::incomplete checkout on runner {runner}: the workspace does "
        "not match the checked-out commit (see issue #9507). Drain or clean "
        "this runner's _work directory and re-run."
    )
    if pruned_by_empty_set:
        _emit(
            "sparse-checkout is enabled with an empty pattern set, which "
            "prunes the working tree."
        )
    if missing:
        _emit(f"{len(missing)} tracked file(s) missing:")
        for path in missing[: args.max_report]:
            _emit(f"  {path}")
        extra = len(missing) - args.max_report
        if extra > 0:
            _emit(f"  ... and {extra} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
