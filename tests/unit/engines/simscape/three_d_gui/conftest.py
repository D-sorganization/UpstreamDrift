"""Per-directory conftest that pivots sys.path for the C3D viewer tests."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


def _pivot_sys_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    engine_python = (
        repo_root
        / "src"
        / "engines"
        / "Simscape_Multibody_Models"
        / "3D_Golf_Model"
        / "python"
    )
    engine_src = engine_python / "src"
    if not engine_src.is_dir():
        return

    # Pre-cache shared deps the viewer uses so they survive the pivot.
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    import importlib

    # Ensure ``src.shared`` and ``src.shared.python`` are concretely loaded
    # so they remain in ``sys.modules`` after we drop ``src`` itself.
    # Without these, ``from src import shared`` after the pivot fails
    # because the engine's ``src/__init__.py`` has no ``shared`` attribute.
    for qual in (
        "src",
        "src.shared",
        "src.shared.python",
        "sidekick.lab.bio.c3d_reader",
        "src.shared.python.qt_utils.wheel_event_filter",
        "src.shared.python.motion_matching.body_skeleton",
    ):
        with contextlib.suppress(ImportError):
            sys.modules[qual] = importlib.import_module(qual)

    # Drop the repo's ``src`` package so we can rebind it to the engine's.
    # Keep ``src.shared`` and everything beneath it so test conftests
    # that import ``src.shared.python.*`` after the pivot still resolve.
    keep_prefix = "src.shared"
    preserved_shared = sys.modules.get("src.shared")
    for modname in list(sys.modules):
        if modname == "src" or modname.startswith("src."):
            if modname == keep_prefix or modname.startswith(keep_prefix + "."):
                continue
            del sys.modules[modname]

    # Bind ``src`` directly to the engine's package via importlib.util so we
    # don't have to fight pytest's sys.path mutations.
    import importlib.util as _util

    spec = _util.spec_from_file_location(
        "src",
        str(engine_src / "__init__.py"),
        submodule_search_locations=[str(engine_src)],
    )
    if spec is None or spec.loader is None:
        return
    src_mod = _util.module_from_spec(spec)
    sys.modules["src"] = src_mod
    spec.loader.exec_module(src_mod)
    # Make ``src`` a multi-rooted package so submodules from EITHER the
    # engine src/ or the repo src/ resolve cleanly. Engine path comes
    # first so ``src.apps`` (engine-only) wins; repo path supplies
    # ``src.tools``, ``src.shared``, etc.
    repo_src = repo_root / "src"
    src_mod.__path__ = [str(engine_src), str(repo_src)]
    if preserved_shared is not None:
        src_mod.shared = preserved_shared

    # Re-attach any preserved ``src.<sub>`` modules as attributes of the
    # freshly rebound ``src`` package. CPython's import machinery uses
    # ``getattr(parent, child)`` for ``from parent import child``, so
    # leaving them only in ``sys.modules`` is not enough — the lookup
    # walks ``src.__dict__`` first and raises ``ImportError`` if missing.
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("src."):
            continue
        parts = modname.split(".")
        if len(parts) != 2:
            continue  # only attach direct children; nested attrs cascade
        setattr(src_mod, parts[1], mod)

    # Add ``<repo>/src`` for bare ``shared.python.*`` imports the viewer does.
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.append(repo_src_str)


_pivot_sys_path()
