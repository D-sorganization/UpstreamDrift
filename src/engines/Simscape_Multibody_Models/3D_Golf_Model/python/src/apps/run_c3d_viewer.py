#!/usr/bin/env python3
"""Standalone entry-point wrapper for the C3D Motion Analysis viewer."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[6]
_REPO_SRC = _REPO_ROOT / "src"
_REPO_SHARED_PY = _REPO_SRC / "shared" / "python"
_VIEWER_MODULE = (
    "src.engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
)

# Import roots in descending precedence. ``src/shared/python`` carries the flat
# ``sidekick`` package that shared helpers import by bare name; it was
# previously omitted, so a source-checkout run depended on the caller having
# exported PYTHONPATH (issue #8088). It is listed last so the pre-existing
# resolution order for everything else is unchanged.
_IMPORT_ROOTS = (_REPO_SRC, _REPO_ROOT, _REPO_SHARED_PY)


def _ensure_import_paths() -> None:
    """Expose repo packages for script-mode execution without package pivoting.

    Postcondition:
        Every entry of :data:`_IMPORT_ROOTS` is on ``sys.path``, ordered as
        declared in :data:`_IMPORT_ROOTS`.
    """
    # Inserting at index 0 reverses the iteration order, so walk backwards to
    # land on the declared precedence (cf. issue #7997, where an unreversed
    # loop silently inverted launcher import precedence).
    for path in (str(root) for root in reversed(_IMPORT_ROOTS)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_main() -> Callable[[], None]:
    _ensure_import_paths()
    module = importlib.import_module(_VIEWER_MODULE)
    return module.main


if __name__ == "__main__":
    _load_main()()
