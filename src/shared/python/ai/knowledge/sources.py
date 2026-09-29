"""Resolve a manifest's globs against repository checkouts."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .manifest import PackManifest, SourceSpec


@dataclass(frozen=True)
class SourceFile:
    """One file selected for the pack; ``path`` is POSIX and repo-relative."""

    repo: str
    path: str
    authority: str
    absolute: Path

    def sha256(self) -> str:
        return hashlib.sha256(self.absolute.read_bytes()).hexdigest()


def select_files(manifest: PackManifest, roots: Mapping[str, Path]) -> list[SourceFile]:
    """Files the manifest selects (first matching source wins), sorted."""
    require_roots(manifest, roots)
    chosen: dict[tuple[str, str], SourceFile] = {}
    listings: dict[str, list[str]] = {}
    for spec in manifest.sources:
        root = Path(roots[spec.repo])
        if spec.repo not in listings:
            listings[spec.repo] = list_repo_files(root)
        for rel in listings[spec.repo]:
            key = (spec.repo, rel)
            if key not in chosen and matches(spec, rel):
                chosen[key] = SourceFile(spec.repo, rel, spec.authority, root / rel)
    return [chosen[k] for k in sorted(chosen)]


def require_roots(manifest: PackManifest, roots: Mapping[str, Path]) -> None:
    """Precondition: a checkout directory exists for every manifest repository."""
    missing = [
        r for r in manifest.repos if r not in roots or not Path(roots[r]).is_dir()
    ]
    if missing:
        raise ValueError(
            f"no checkout directory for repositories: {', '.join(missing)}"
        )


def matches(spec: SourceSpec, rel: str) -> bool:
    return any(_glob_re(g).fullmatch(rel) for g in spec.include) and not any(
        _glob_re(g).fullmatch(rel) for g in spec.exclude
    )


def list_repo_files(root: Path) -> list[str]:
    """Tracked plus untracked-unignored files of a checkout, else a walk."""
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return _walk(root)
    names = {n for n in out.decode("utf-8", "replace").split("\0") if n}
    return sorted(n for n in names if (root / n).is_file())


def head_commit(root: Path) -> str:
    """HEAD sha of ``root``, or ``""`` when it is not a git checkout."""
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _walk(root: Path) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            found.append(Path(dirpath, name).relative_to(root).as_posix())
    return sorted(found)


@lru_cache(maxsize=256)
def _glob_re(glob: str) -> re.Pattern[str]:
    """``**/`` spans zero or more directories; ``*`` and ``?`` stay within one."""
    out: list[str] = []
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    return re.compile("".join(out))
