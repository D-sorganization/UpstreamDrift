"""Bootstrap and diagnostics for ``python -m src.tools.native_viewer_export`` (#14748).

Pytest and the launcher inject ``src/shared/python`` (and vendored Tools paths)
before importing the CLI. A bare ``python -m`` from the repository root does
not, so ``src.shared.python.*`` imports can fail before ``shared`` spellings
would resolve. :func:`ensure_native_viewer_export_imports` prepends the repo root
and installs shared import aliases.

Separately, a host interpreter with an editable install of another repository
can bind the top-level ``shared`` package outside the pinned Tools tree; the CLI
then dies with an opaque import error. :func:`shared_shadow_message` explains
that case when :func:`ensure_native_viewer_export_imports` was not enough.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_native_viewer_export_imports() -> None:
    """Prepend the repository root and install shared import aliases once."""
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    importlib.import_module("src")
    from src.shared.python.import_aliases import install_shared_import_aliases

    install_shared_import_aliases()


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
