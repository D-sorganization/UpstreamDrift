"""Per-directory conftest that pivots sys.path for the C3D viewer tests.

Mirrors the pivot used by ``tests/unit/engines/simscape/three_d_gui``
so the engine's ``src.apps.*`` namespace shadows the repo's top-level
``src.`` package when running the C3D viewer plot-style tests.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTEST_QT_API", "pyqt6")


def _pivot_sys_path() -> None:
    here = Path(__file__).resolve()
    # tests/unit/c3d_viewer/ui/conftest.py -> repo root is parents[4].
    repo_root = here.parents[4]
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

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    import importlib

    for qual in (
        "src.shared.python.upstream_drift_tools.lab.bio.c3d_reader",
        "src.shared.python.qt_utils.wheel_event_filter",
        "src.shared.python.motion_matching.body_skeleton",
        "src.shared.python.plot_style",
    ):
        with contextlib.suppress(ImportError):
            sys.modules[qual] = importlib.import_module(qual)

    keep_prefix = "src.shared."
    for modname in list(sys.modules):
        if modname == "src" or modname.startswith("src."):
            if modname.startswith(keep_prefix):
                continue
            del sys.modules[modname]

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
    repo_src = repo_root / "src"
    src_mod.__path__ = [str(engine_src), str(repo_src)]

    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.append(repo_src_str)


_pivot_sys_path()
