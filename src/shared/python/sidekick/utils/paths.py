"""Canonical repository path utilities.

Single source of truth for repository root discovery and path resolution.
All other modules should import get_repo_root() from here.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT_MARKERS = (".git", "pyproject.toml", "tools.json")
_MAX_SEARCH_DEPTH = 10


def get_repo_root(start_path: Path | str | None = None) -> Path:
    """Find the repository root by searching upward for marker files.

    Args:
        start_path: Directory to start searching from.
            Defaults to the directory of the calling file.

    Returns:
        Absolute path to the repository root.

    Raises:
        FileNotFoundError: If no repository root can be found.
    """
    if start_path is None:
        # Use calling file's directory as default
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            calling_file = frame.f_back.f_globals.get("__file__")
            if calling_file:
                start_path = Path(calling_file).resolve().parent
        if start_path is None:
            start_path = Path.cwd()

    current = Path(start_path).resolve()

    for _ in range(_MAX_SEARCH_DEPTH):
        if any((current / marker).exists() for marker in _REPO_ROOT_MARKERS):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        f"Repository root not found searching upward from {start_path}. "
        f"Looked for markers: {_REPO_ROOT_MARKERS}"
    )
