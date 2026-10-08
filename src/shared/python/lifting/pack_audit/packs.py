"""Locate the four engine model-pack checkouts and make them importable.

The packs live in separate fleet repositories
(``D-sorganization/{OpenSim,MuJoCo,Drake,Pinocchio}_Models``).  This audit
reads them as they are checked out next to UpstreamDrift and never modifies
them.  Unavailable packs are reported, never faked.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .names import ENGINES

_REPO_DIRS = {
    "mujoco": "MuJoCo_Models",
    "opensim": "OpenSim_Models",
    "drake": "Drake_Models",
    "pinocchio": "Pinocchio_Models",
}
PACK_ROOT_ENV = "LIFT_PACK_ROOT"


@dataclass(frozen=True)
class PackLocation:
    """A pack checkout: where it is, which commit, and under what licence."""

    engine: str
    repo: str
    root: Path
    src: Path
    package: str
    commit: str | None
    licence: str | None

    def activate(self) -> None:
        """Put the pack's ``src`` directory on ``sys.path`` (idempotent)."""
        path = str(self.src)
        if path not in sys.path:
            sys.path.insert(0, path)


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _licence_name(root: Path) -> str | None:
    licence = root / "LICENSE"
    if not licence.is_file():
        return None
    first = licence.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return first[0].strip() if first else None


def search_roots(extra: list[Path] | None = None) -> list[Path]:
    """Directories searched for pack checkouts, in priority order."""
    roots = [Path(p) for p in extra or []]
    env = os.environ.get(PACK_ROOT_ENV)
    if env:
        roots.append(Path(env))
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            roots.append(parent.parent)
            # A git worktree sits one level deeper (``<repo>-worktrees/<name>``).
            roots.append(parent.parent.parent)
            break
    roots.append(Path.home() / "Repositories")
    return roots


def locate_pack(
    engine: str, extra_roots: list[Path] | None = None
) -> PackLocation | None:
    """Return the checkout for *engine*, or ``None`` when it is not on disk.

    Raises:
        ValueError: If *engine* is not one of the four supported engines.
    """
    if engine not in _REPO_DIRS:
        raise ValueError(f"unknown engine {engine!r}; use {list(ENGINES)}")
    repo = _REPO_DIRS[engine]
    package = f"{engine}_models"
    for base in search_roots(extra_roots):
        root = base / repo
        src = root / "src"
        if (src / package).is_dir():
            return PackLocation(
                engine=engine,
                repo=repo,
                root=root,
                src=src,
                package=package,
                commit=_git_commit(root),
                licence=_licence_name(root),
            )
    return None
