"""Fail-fast diagnosis for a foreign ``shared`` package (UpstreamDrift#14748).

A host interpreter with an editable install of another repository (for example
Tools_Private) can bind the top-level ``shared`` package to that checkout, so
the vendored Tools tree is never consulted and the CLI dies with an opaque
``No module named 'shared.python.sidekick'``. This module only explains that
state; it does not change import order or ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path


def shared_shadow_message(vendor_src: Path, shared_file: str | None) -> str | None:
    """Return a remedy message when ``shared`` resolves outside *vendor_src*.

    Args:
        vendor_src: the pinned Tools ``src`` directory (``vendor/ud-tools/src``).
        shared_file: ``shared.__file__``; ``None`` for a namespace package.

    Returns:
        ``None`` when ``shared`` is a namespace package or lives under
        *vendor_src*; otherwise a human-readable explanation with the remedy.

    Raises:
        TypeError: if *vendor_src* is not a ``Path``.
    """
    if not isinstance(vendor_src, Path):
        raise TypeError(f"vendor_src must be a Path, got {type(vendor_src).__name__}")
    if shared_file is None:
        return None
    if Path(shared_file).resolve().is_relative_to(vendor_src.resolve()):
        return None
    return (
        f"The top-level 'shared' package resolves to {shared_file}, outside "
        f"{vendor_src}. An editable install or a PYTHONPATH entry from another "
        "repository is shadowing the pinned Tools tree. Uninstall that editable "
        "package or run from the repository venv."
    )
