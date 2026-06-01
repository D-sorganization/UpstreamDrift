"""Shared engine-version and git-commit provenance probes (issue #6939).

Five engine providers re-implemented the same ``importlib.metadata`` version
cascade, and three modules reimplemented a ``git rev-parse`` subprocess. These
two helpers are the single source of truth so the probes cannot drift.

Public API:
    engine_package_version(module, *distributions) -> str
        Resolve an engine wheel's version, trying ``module.__version__`` first
        and then each distribution name, returning ``"unknown"`` otherwise.
    git_commit_short() -> str
        Best-effort short SHA of ``HEAD``; ``"unknown"`` outside a repo.
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from types import ModuleType
from typing import Any

__all__ = [
    "engine_package_version",
    "git_commit_short",
]

# Bound the git subprocess so a hung/foreign git never stalls a fit.
_GIT_TIMEOUT_S = 2.0


def engine_package_version(
    module: ModuleType | Any | None,
    *distributions: str,
) -> str:
    """Return an engine wheel's version string, or ``"unknown"``.

    Resolution order (issue #6939, unifying the five copied cascades):

    1. ``module.__version__`` when ``module`` is an imported module exposing a
       non-empty string attribute (covers ``pydrake``, ``mujoco``, etc.);
    2. :func:`importlib.metadata.version` for each name in ``distributions``,
       in order (covers wheels whose import name differs from the distribution
       name, e.g. ``pin`` for ``pinocchio`` or ``drake`` for ``pydrake``);
    3. ``"unknown"`` if nothing resolves.

    Args:
        module: The already-imported engine module (or ``None`` when the
            engine is not installed). Pass ``None`` to skip the
            ``__version__`` probe and go straight to distribution lookup.
        *distributions: Distribution names to try via
            :func:`importlib.metadata.version`, in priority order.

    Returns:
        The first non-empty version string found, else ``"unknown"``.
    """
    if isinstance(module, ModuleType):
        version = getattr(module, "__version__", None)
        if isinstance(version, str) and version:
            return version
    for dist in distributions:
        try:
            resolved = _metadata_version(dist)
        except PackageNotFoundError:
            continue
        if resolved:
            return resolved
    return "unknown"


def git_commit_short() -> str:
    """Return the short SHA of ``HEAD``, or ``"unknown"`` if unavailable.

    Best-effort: returns ``"unknown"`` when git is absent, the working tree is
    not a repository, or the call times out. Never raises (issue #6939).
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_S,
        )
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        OSError,
    ):
        return "unknown"
    return out.decode("ascii", errors="replace").strip() or "unknown"
