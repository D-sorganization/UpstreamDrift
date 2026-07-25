"""Resolve ``__main__.py`` tile targets to importable dotted module paths.

Several launcher tiles point at a package entry point, e.g.::

    "path": "src/shared/python/pendulum_simulator/__main__.py"

Running that file as a *script* (``python <abs>/__main__.py``) gives it no
parent package, so its very first ``from .gui import MainWindow`` dies with
``ImportError: attempted relative import with no known parent package`` and the
child exits instantly.  The launcher only checked that ``Popen`` returned a
handle, so it reported "Launched ... (PID: n)" and showed nothing — the silent
exits in #8065 and #8069.

The fix is to preserve the **full** dotted path including the ``src.`` prefix
(#8086) and hand the tile to ``python -m src.shared.python.pendulum_simulator``.

This module is pure path arithmetic so it can be unit tested without Qt or a
subprocess.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PACKAGE_MAIN_FILENAME", "resolve_package_main_module"]

PACKAGE_MAIN_FILENAME = "__main__.py"


def resolve_package_main_module(script_path: Path, repo_path: Path) -> str | None:
    """Return the dotted module name for a package ``__main__.py`` target.

    Args:
        script_path: Absolute path of the resolved tile artifact.
        repo_path: Absolute repository root; the dotted path is relative to it.

    Returns:
        The dotted package name (e.g. ``"src.shared.python.pendulum_simulator"``)
        when ``script_path`` is the ``__main__.py`` of an importable package
        rooted inside ``repo_path``; otherwise ``None``.

    Raises:
        TypeError: If either argument is not a :class:`~pathlib.Path`.

    Postcondition:
        The returned name never starts or ends with ``"."`` and every segment
        is a valid Python identifier.
    """
    if not isinstance(script_path, Path):
        raise TypeError(f"script_path must be a Path, got {type(script_path).__name__}")
    if not isinstance(repo_path, Path):
        raise TypeError(f"repo_path must be a Path, got {type(repo_path).__name__}")

    if script_path.name != PACKAGE_MAIN_FILENAME:
        return None

    package_dir = script_path.parent
    if not (package_dir / "__init__.py").exists():
        return None

    try:
        relative = package_dir.resolve().relative_to(repo_path.resolve())
    except ValueError:
        # Outside the repository (e.g. a sibling provider repo) — the caller
        # cannot guarantee it is on PYTHONPATH, so stay with the script path.
        return None

    parts = relative.parts
    if not parts or not all(part.isidentifier() for part in parts):
        return None

    # Every intermediate directory must also be a package for ``-m`` to work.
    current = repo_path
    for part in parts:
        current = current / part
        if not (current / "__init__.py").exists():
            return None

    return ".".join(parts)
