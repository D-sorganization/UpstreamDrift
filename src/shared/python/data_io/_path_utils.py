"""
Path Utilities for Output Manager

Directory structure creation, filename sanitization, and fast directory scanning.
Extracted from output_manager.py as part of monolith decomposition (#2486).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.datetime_utils import timestamp_filename
from ._format_handlers import OutputFormat
from .common_utils import get_logger

logger = get_logger(__name__)

#: Environment variable overriding the default output base directory.
OUTPUT_DIR_ENV_VAR = "UPSTREAM_DRIFT_OUTPUT_DIR"


def find_project_root(start: Path | None = None) -> Path | None:
    """Return the repository root above ``start``, or ``None`` when not found.

    The previous heuristic accepted any ancestor containing a directory named
    ``engines``, which matches ``src/`` (``src/engines`` exists), so the default
    output directory resolved to ``src/output`` and dropped generated run
    artifacts into the source tree (issue #9220).

    ``pyproject.toml`` alone is not a unique marker either -- several packages
    nested under ``src/`` ship one. So the checkout's ``.git`` entry is tried
    first (a file in a worktree, a directory in a normal clone), falling back
    to the *outermost* ancestor carrying a ``pyproject.toml`` for installs that
    are not a git checkout.

    Args:
        start: Path to search upwards from. Defaults to this module.

    Returns:
        The detected project root, or ``None`` when no marker is found.
    """
    origin = (start or Path(__file__)).resolve()
    candidates = (origin, *origin.parents)

    git_root = next((p for p in candidates if (p / ".git").exists()), None)
    if git_root is not None:
        return git_root

    pyproject_roots = [p for p in candidates if (p / "pyproject.toml").is_file()]
    return pyproject_roots[-1] if pyproject_roots else None


def sanitize_filename(filename: str, format_type: OutputFormat) -> str:
    """
    Sanitize and normalize an output filename.

    Strips format suffixes, handles test-mode names, and appends a timestamp
    when the filename has no digit component.
    """
    if "OutputFormat." in filename:
        filename = filename.split(".")[-1]
        filename = "test_format"

    filename = filename.removesuffix(f".{format_type.value}")

    if not any(char.isdigit() for char in filename) and "test_" not in filename:
        timestamp = timestamp_filename(utc=False)
        filename = f"{filename}_{timestamp}"

    return filename


def fast_dir_scan(directory: Path, max_depth: int = 10) -> Iterator[Path]:
    """
    Fast directory scanning using os.scandir instead of rglob.

    PERF-003: Optimized from rglob to os.scandir for 10-50x speedup.

    Args:
        directory: Directory to scan
        max_depth: Maximum recursion depth (prevents infinite loops)

    Yields:
        Path objects for all files found
    """

    def _scan_recursive(path: Path, depth: int = 0) -> Iterator[Path]:
        if depth > max_depth:
            return

        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    if entry.is_file(follow_symlinks=False):
                        yield entry_path
                    elif entry.is_dir(follow_symlinks=False):
                        yield from _scan_recursive(entry_path, depth + 1)
        except (OSError, PermissionError):
            # Skip directories we can't access
            pass

    yield from _scan_recursive(directory)


def create_output_structure(directories: dict[str, Path]) -> None:
    """
    Create the standard output directory structure.

    Args:
        directories: Mapping of directory role names to Path objects.
    """
    # Main directories
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    # Engine-specific simulation directories
    engines = ["mujoco", "drake", "pinocchio", "matlab"]
    for engine in engines:
        (directories["simulations"] / engine).mkdir(exist_ok=True)

    # Analysis subdirectories
    analysis_types = ["biomechanics", "trajectories", "optimization", "comparisons"]
    for analysis_type in analysis_types:
        (directories["analysis"] / analysis_type).mkdir(exist_ok=True)

    # Export subdirectories
    export_types = ["videos", "images", "data", "c3d"]
    for export_type in export_types:
        (directories["exports"] / export_type).mkdir(exist_ok=True)

    # Report subdirectories
    report_types = ["pdf", "html", "presentations"]
    for report_type in report_types:
        (directories["reports"] / report_type).mkdir(exist_ok=True)

    # Cache subdirectories
    cache_types = ["models", "computations", "temp"]
    for cache_type in cache_types:
        (directories["cache"] / cache_type).mkdir(exist_ok=True)

    logger.info("Output directory structure created successfully")


def resolve_base_path(base_path: Any) -> Path:
    """
    Resolve the base output path, auto-detecting the project root when None.

    Resolution order when ``base_path`` is ``None`` (issue #9220):

    1. ``UPSTREAM_DRIFT_OUTPUT_DIR`` when set -- the override the test suite
       uses so that no run writes generated output into the checkout.
    2. ``<project root>/output`` -- the documented interactive/CLI location
       described by ``output/README.md``.
    3. ``<cwd>/output`` when no project root can be located.

    Args:
        base_path: Explicit path, or None to auto-detect.

    Returns:
        Resolved and created base Path.
    """
    if base_path is not None:
        resolved = Path(base_path)
    else:
        override = os.environ.get(OUTPUT_DIR_ENV_VAR)
        if override:
            resolved = Path(override).expanduser()
        else:
            project_root = find_project_root()
            resolved = (project_root or Path.cwd()) / "output"

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
