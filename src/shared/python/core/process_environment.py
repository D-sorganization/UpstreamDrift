"""Explicit import roots for repository Python workers, independent of Qt."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def repo_python_environment(
    repo_root: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Copy settings and put this checkout's ``src`` first on PYTHONPATH.

    Launch ``python -m src...`` from ``repo_root``. The working directory
    resolves ``src``; this explicit path resolves bare packages such as
    ``bunkershot3d``. Parent ``sys.path`` edits do not propagate to workers.
    Preserve all other settings, including SDK plugin and rendering choices.
    Reapplying this function is idempotent and never mutates ``base``.
    """
    env = dict(os.environ if base is None else base)
    src = str(repo_root / "src")
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != src]
    env["PYTHONPATH"] = os.pathsep.join([src, *parts])
    return env
