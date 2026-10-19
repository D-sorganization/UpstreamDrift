"""Resolve the superproject's pinned ``vendor/ud-tools`` gitlink SHA.

WSL Git cannot read a Windows-created worktree ``.git`` pointer, so Docker
explorer builds need a helper that selects ``git.exe`` when appropriate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_VENDOR_GITLINK = Path("vendor/ud-tools")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def git_executable_for_repo(repo_root: Path) -> str:
    """Return ``git`` or ``git.exe`` for the checkout's host layout."""
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be pathlib.Path")
    pointer = repo_root / ".git"
    if not pointer.is_file():
        return "git"
    gitdir = pointer.read_text(encoding="utf-8").strip()
    windows_pointer = gitdir.startswith("gitdir: ") and gitdir[9:10] == ":"
    return "git.exe" if windows_pointer else "git"


def _git_rev_parse_gitlink(repo_root: Path, executable: str) -> str | None:
    try:
        completed = subprocess.run(
            (executable, "rev-parse", "HEAD:vendor/ud-tools"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip().lower()
    return sha if _SHA_RE.fullmatch(sha) else None


def _git_ls_tree_gitlink(repo_root: Path, executable: str) -> str | None:
    try:
        completed = subprocess.run(
            (executable, "ls-tree", "HEAD", "--", _VENDOR_GITLINK.as_posix()),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    fields = completed.stdout.strip().split()
    if len(fields) != 4 or fields[0] != "160000" or fields[1] != "commit":
        return None
    sha = fields[2].lower()
    return sha if _SHA_RE.fullmatch(sha) else None


def resolve_tools_gitlink_sha(repo_root: Path) -> str:
    """Return the 40-hex commit recorded for ``vendor/ud-tools`` at ``HEAD``.

    Raises:
        RuntimeError: when the gitlink cannot be read from the repository.
        TypeError: when ``repo_root`` is not a :class:`~pathlib.Path`.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be pathlib.Path")
    root = repo_root.resolve(strict=False)
    executable = git_executable_for_repo(root)
    for candidate in (
        _git_rev_parse_gitlink(root, executable),
        _git_ls_tree_gitlink(root, executable),
    ):
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "cannot read the vendor/ud-tools gitlink; export TOOLS_GITLINK_SHA=<40-hex sha> explicitly"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the pinned vendor/ud-tools gitlink SHA for Docker builds",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="UpstreamDrift repository root (default: cwd)",
    )
    args = parser.parse_args()
    try:
        print(resolve_tools_gitlink_sha(args.root))
    except (RuntimeError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(64) from exc


if __name__ == "__main__":
    main()
