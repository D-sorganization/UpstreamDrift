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
_VIEWER_MODULE = (
    "src.engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
)


def _ensure_import_paths() -> None:
    """Expose repo packages for script-mode execution without package pivoting."""
    for path in (str(_REPO_ROOT), str(_REPO_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_main() -> Callable[[], None]:
    _ensure_import_paths()
    module = importlib.import_module(_VIEWER_MODULE)
    return module.main


if __name__ == "__main__":
    _load_main()()
