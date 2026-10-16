"""Importable shim for the Simscape 3D Golf Model embed adapter.

The real adapter package lives at
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps``.
The ``3D_Golf_Model`` path segment starts with a digit, so that package
can never be reached through a dotted import path (issue #8856) — the
old bootstrap entry
``engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps._embed_adapter``
was unimportable on two counts (missing ``src.`` prefix, invalid
identifier) and failed silently.

This module loads the ``apps`` package from its file path under a
synthetic, importable package name. Executing the package ``__init__``
registers the C3D viewer's :class:`EmbeddableTool` adapter with the
process-wide registry, exactly as a normal dotted import would.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

__all__ = [
    "APPS_PACKAGE_NAME",
    "ROOT_PACKAGE_NAME",
    "TOOL_ID",
    "load_simscape_apps_package",
]

ROOT_PACKAGE_NAME = "upstream_drift_simscape_3d_golf"
"""Synthetic name for the Simscape ``python/src`` directory.

The apps package uses ``from ...c3d_reader import ...`` (three levels up,
``apps.services`` -> ``apps`` -> ``src``). Registering ``src`` as a parent
package makes those relative imports resolvable (issue #12995).
"""

APPS_PACKAGE_NAME = f"{ROOT_PACKAGE_NAME}.apps"
"""Synthetic import name for the Simscape ``apps`` package."""

TOOL_ID = "c3d_viewer"
"""Tool id the adapter registers under."""

_SRC_ROOT = Path(__file__).resolve().parents[2]
_PKG_DIR = (
    _SRC_ROOT
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "python"
    / "src"
)
_APPS_DIR = _PKG_DIR / "apps"


def _ensure_root_package() -> ModuleType:
    """Register ``python/src`` as a synthetic parent package (idempotent).

    The package ``__init__`` is deliberately not executed: only the search
    path is needed so sibling modules (``c3d_reader``, ``logger_utils``)
    resolve through relative imports from ``apps``.
    """
    root = sys.modules.get(ROOT_PACKAGE_NAME)
    if root is None:
        if not _PKG_DIR.is_dir():
            raise ImportError(f"Simscape python/src package not found at {_PKG_DIR}")
        root = ModuleType(ROOT_PACKAGE_NAME)
        root.__path__ = [str(_PKG_DIR)]  # type: ignore[attr-defined]
        root.__package__ = ROOT_PACKAGE_NAME
        sys.modules[ROOT_PACKAGE_NAME] = root
    return root


def load_simscape_apps_package() -> ModuleType:
    """Load (or return the cached) Simscape ``apps`` package.

    Returns:
        The loaded package module.

    Raises:
        ImportError: If the package files are missing, the import spec
            cannot be built, or executing the package failed to register
            the ``c3d_viewer`` tool (a silent-registration regression).
    """
    module = sys.modules.get(APPS_PACKAGE_NAME)
    if module is None:
        root = _ensure_root_package()
        init_path = _APPS_DIR / "__init__.py"
        if not init_path.is_file():
            raise ImportError(f"Simscape apps package not found at {init_path}")
        spec = importlib.util.spec_from_file_location(
            APPS_PACKAGE_NAME,
            init_path,
            submodule_search_locations=[str(_APPS_DIR)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build an import spec for {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[APPS_PACKAGE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(APPS_PACKAGE_NAME, None)
            raise
        root.apps = module  # type: ignore[attr-defined]

    # The apps package registers its adapter best-effort (it suppresses
    # ImportError so the engine keeps working without the launcher on
    # sys.path). From the launcher's side that silence is a bug, not a
    # feature — surface it as a loud ImportError so bootstrap records it.
    from src.shared.python.launcher_embed import get_embeddable_tool

    if get_embeddable_tool(TOOL_ID) is None:
        raise ImportError(
            f"Simscape apps package loaded but did not register {TOOL_ID!r}; "
            "check that src/shared/python/launcher_embed is importable from "
            "the apps package __init__"
        )
    return module


load_simscape_apps_package()
