# ruff: noqa: E501
"""Bootstrap module for launch scripts that run before package installation.

**Legitimate sys.path exception** (issue #682): This module is the *only*
sanctioned place for ``sys.path.insert`` in the repository.  It exists
solely for ``launch_pyqt6.py`` entry-point scripts that may execute
before the package has been installed via ``pip install -e .``.  Library
and test code must NOT call :func:`ensure_paths` -- they should rely on
the installed package instead.

Usage in launch_pyqt6.py scripts::

    from pathlib import Path
    # One-time bootstrap: find repo root and add standard paths
    _root = Path(__file__).resolve().parents[2]  # adjust depth to reach repo root
    exec((_root / "src" / "shared" / "python" / "upstream_drift_tools" / "bootstrap.py").read_text())

Or more explicitly::

    from sidekick.bootstrap import ensure_paths
    ensure_paths()

This module exists as a transition mechanism. Once all imports use the installed
package (via ``pip install -e .``), this module becomes unnecessary.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "ensure_paths",
]


def ensure_paths(repo_root: Path | str | None = None) -> Path:
    """Add standard source directories to sys.path if not already present.

    This is intended for launch scripts and development use only.
    Library code should never call this function.

    Args:
        repo_root: Repository root path. Auto-detected if None.

    Returns:
        The resolved repository root path.
    """
    resolved_root: Path
    if repo_root is None:
        from shared.python.sidekick.utils.paths import get_repo_root

        resolved_root = Path(get_repo_root()).resolve()
    else:
        resolved_root = Path(repo_root).resolve()

    standard_paths = [
        resolved_root / "src",
        resolved_root / "src" / "python" / "src",
    ]

    for path in standard_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return resolved_root
