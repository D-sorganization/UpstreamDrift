#!/usr/bin/env python3
"""Motion-matching leaderboard freshness gate (issues #4237, #14767).

``run_leaderboard.m`` records the commit it ran on as ``git_head`` in
``motion_matching/results/.leaderboard_meta.json``. The gate asks whether the
committed leaderboard still describes the optimizer code at ``HEAD``.

A squash merge orphans the ``git_head`` written on a PR branch, so every later
PR failed the ancestry check, even docs-only ones (#14767). With ``--base``
(the PR base commit) the gate first looks at what the PR itself changes under
``motion_matching/``: if it changes no solver input, the leaderboard cannot be
made stale by it and the gate passes without consulting ``git_head``. Any
solver change still needs ``git_head`` to be a known ancestor of ``HEAD`` with
no solver change after it, i.e. a rerun on the branch.

Solver inputs are every file under ``motion_matching/`` except Markdown,
anything under a ``tests/`` directory, and the leaderboard's own outputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import re
import subprocess
import sys

DEFAULT_MM_PATH = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching"
)
META_RELATIVE = "results/.leaderboard_meta.json"
_NOT_SOLVER_INPUT = re.compile(
    r"^(results/(LEADERBOARD\.md|\.leaderboard_meta\.json|.*\.mat)"
    r"|.*\.md|(.*/)?tests/.*)$"
)
RERUN = (
    "Re-run the leaderboard locally and commit the refreshed outputs:\n"
    "  cd src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab\n"
    "  matlab -batch \"addpath(genpath('motion_matching')); run_leaderboard\""
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def _git_lines(repo: Path, *args: str) -> list[str]:
    result = _git(repo, *args)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return sorted({line for line in result.stdout.splitlines() if line})


def solver_inputs(paths: list[str], mm_path: str) -> list[str]:
    """Return the paths that can change leaderboard results."""
    prefix = mm_path.rstrip("/") + "/"
    return [
        path
        for path in paths
        if path.startswith(prefix) and not _NOT_SOLVER_INPUT.match(path[len(prefix) :])
    ]


def _read_git_head(meta: Path) -> str:
    sha = str(json.loads(meta.read_text(encoding="utf-8")).get("git_head") or "")
    if not sha.strip():
        raise ValueError(f"git_head missing or empty in {meta}")
    return sha.strip()


def check(
    repo: Path, mm_path: str, base: str | None, emit: Callable[[str], None]
) -> int:
    """Return 0 when the committed leaderboard is fresh at ``HEAD``, else 1.

    ``base`` is the PR base commit, or ``None`` (e.g. ``workflow_dispatch``)
    to apply the full ancestry contract. Every finding is passed to ``emit``.
    """
    if not mm_path.strip():
        raise ValueError("mm_path must name the motion_matching directory")
    meta = repo / mm_path / META_RELATIVE
    if not meta.is_file():
        emit(f"::error::Missing {mm_path}/{META_RELATIVE}; run run_leaderboard.m.")
        return 1
    if base is not None:
        changed = _git_lines(repo, "diff", "--name-only", f"{base}...HEAD", "--")
        if not solver_inputs(changed, mm_path):
            emit(
                "This PR changes no solver inputs under motion_matching/ (only "
                "docs, tests or leaderboard outputs); the leaderboard stays valid."
            )
            return 0
    git_head = _read_git_head(meta)
    emit(f"Leaderboard pinned to git_head={git_head}")
    if _git(repo, "cat-file", "-e", f"{git_head}^{{commit}}").returncode != 0:
        emit(f"::error::git_head {git_head} is not a known commit in this repo.")
        emit(RERUN)
        return 1
    if _git(repo, "merge-base", "--is-ancestor", git_head, "HEAD").returncode != 0:
        emit(f"::error::Leaderboard git_head {git_head} is not an ancestor of HEAD.")
        emit(RERUN)
        return 1
    since = _git_lines(
        repo, "log", "--name-only", "--pretty=format:", f"{git_head}..HEAD", "--"
    )
    stale = solver_inputs(since, mm_path)
    if stale:
        emit(
            "::error::motion_matching/ solver inputs changed since the leaderboard "
            f"was last regenerated (git_head={git_head}):"
        )
        for path in stale:
            emit(f"  - {path}")
        emit(RERUN)
        return 1
    emit(f"Leaderboard is fresh: no solver input changed since {git_head}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by ``motion-matching-leaderboard.yml``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mm-path", default=DEFAULT_MM_PATH)
    parser.add_argument(
        "--base", default=None, help="PR base commit; omit for the full contract"
    )
    args = parser.parse_args(argv)
    return check(args.repo, args.mm_path, args.base or None, print)


if __name__ == "__main__":
    sys.exit(main())
