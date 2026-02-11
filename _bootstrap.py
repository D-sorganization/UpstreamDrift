"""Repository-level bootstrap module.

When ``pip install -e .`` has NOT been run **or** a script is executed
directly (e.g. ``python src/engines/physics_engines/drake/python/__main__.py``),
the shared library directories need to be on ``sys.path``.

This module performs a **conditional** bootstrap: it adds the repo root,
``src/``, and ``src/shared/python/`` to ``sys.path`` only if they are not
already present.

Usage in standalone scripts / entry points::

    # 1. Find repo root (inline, before any src.* imports)
    import sys
    from pathlib import Path
    _root = next(
        (p for p in Path(__file__).resolve().parents
         if (p / "pyproject.toml").exists()),
        Path(__file__).resolve().parent,
    )
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    # 2. Use bootstrap to set up remaining paths
    from _bootstrap import bootstrap
    bootstrap(__file__)
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap(caller_file: str) -> Path:
    """Bootstrap import paths for a script or entry point.

    Adds the repository root, ``src/``, and ``src/shared/python/``
    to ``sys.path`` if not already present.

    Args:
        caller_file: ``__file__`` from the calling module.

    Returns:
        The resolved repository root directory.
    """
    caller = Path(caller_file).resolve()
    # Walk up until we find pyproject.toml (repo root marker)
    repo_root = caller.parent
    for _ in range(10):
        if (repo_root / "pyproject.toml").exists():
            break
        repo_root = repo_root.parent
    else:
        repo_root = caller.parent

    paths_to_add = [
        str(repo_root),
        str(repo_root / "src"),
        str(repo_root / "src" / "shared" / "python"),
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    return repo_root
