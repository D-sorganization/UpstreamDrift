#!/usr/bin/env python3
"""Standalone entry-point wrapper for the C3D Motion Analysis viewer.

The viewer module (``c3d_viewer.py``) uses relative imports
(``from .core.models import C3DDataModel`` etc.), so it cannot be run
directly via ``python c3d_viewer.py``. The launcher's
``ProcessManager.launch_script`` invokes ``python <path>`` and would
trip those imports.

This wrapper inserts the parent ``src/`` directory onto ``sys.path``
so the ``apps.c3d_viewer`` package resolves with its relative imports
intact, then dispatches to the viewer's :func:`main`.

Used by ``src/launchers/launcher_simulation.py:_launch_c3d_viewer``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo and engine each ship a ``src/`` package. Resolve the canonical
# C3D reader FIRST against the repo root (so it imports its own siblings
# via the canonical chain), then swap ``sys.modules['src']`` and
# ``sys.path`` to point at the engine's local package so the viewer's
# relative imports (``.services.*``, ``...c3d_reader``) resolve.
_HERE = Path(__file__).resolve().parent
_PYTHON_DIR = _HERE.parent.parent  # apps -> src -> python
_REPO_ROOT = _HERE.parents[6]  # apps -> src -> python -> 3D_Golf_Model -> Simscape_..
# -> engines -> src -> repo

# Step 1: import canonical reader from the repo root.
sys.path.insert(0, str(_REPO_ROOT))
from src.shared.python.upstream_drift_tools.lab.bio import (  # noqa: E402
    c3d_reader as _canonical,  # noqa: E402
)

# Step 2: pivot ``src`` to the engine's local package so the viewer's
# relative imports resolve. Keep the canonical module reachable via the
# fully-qualified name so ``sys.modules`` cache hits don't re-execute it.
_canonical_qualname = "src.shared.python.upstream_drift_tools.lab.bio.c3d_reader"
sys.modules[_canonical_qualname] = _canonical

# Drop the repo's ``src`` so importing ``src`` afterwards picks up the
# engine package (we keep _REPO_ROOT on sys.path for shared.python.*
# absolute imports done by the engine shim).
for _modname in list(sys.modules):
    if _modname == "src" or _modname.startswith("src."):
        if _modname == _canonical_qualname:
            continue
        if _modname.startswith("src.shared."):
            continue  # keep canonical chain hot
        del sys.modules[_modname]

sys.path.insert(0, str(_PYTHON_DIR))

# Step 3: also add ``<repo>/src/`` so bare top-level imports done by the
# viewer code (``from shared.python.security...``) resolve. The engine's
# local ``src/`` does not contain a ``shared/`` subpackage, so adding the
# repo's ``src/`` directory does not conflict with the engine package.
_REPO_SRC = _REPO_ROOT / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from src.apps.c3d_viewer import main  # noqa: E402  (post-sys.path pivot)

if __name__ == "__main__":
    main()
