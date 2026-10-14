"""Shared OpenSim access for tests (#12065).

The single test-side route to the OpenSim SDK, the counterpart of
:mod:`tests.helpers.ezc3d_support`.  Before importing it holds the
native-runtime process boundary: in a process that already hosts the Python
``ezc3d`` extension (or a foreign ``ezc3d.dll``) the import is never attempted
and the caller -- a test, a fixture, or a whole module when called at import
time -- is skipped with an explicit :data:`BOUNDARY_SKIP_PREFIX` reason instead
of loading an ABI-incompatible DLL.

It skips for absence *only* when OpenSim is genuinely not installed
(``ModuleNotFoundError``).  Any other ``ImportError`` (a broken SDK install, a
DLL conflict) is a real failure and must surface, not hide as a skip.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from tests.helpers.native_runtime_boundary import (
    BOUNDARY_SKIP_PREFIX,
    require_runtime_boundary,
)

__all__ = ["BOUNDARY_SKIP_PREFIX", "SKIP_REASON", "import_opensim", "require_opensim"]

SKIP_REASON = "OpenSim Python bindings are not installed"


def import_opensim() -> ModuleType:
    """Import and return the OpenSim SDK module (no boundary check, no skip)."""
    return importlib.import_module("opensim")


def require_opensim() -> ModuleType:
    """Return ``opensim``; skip if it is not installed or would cross the boundary.

    Postcondition: on return the module is loaded in a process that hosts no
    conflicting native C3D runtime.  Skips are module-level capable, so a call
    at import time skips the whole test module.

    Raises:
        ImportError: OpenSim is installed but fails to import for any reason
            other than being absent.
    """
    require_runtime_boundary("opensim")
    try:
        return import_opensim()
    except ModuleNotFoundError as exc:
        if exc.name != "opensim":
            raise
        pytest.skip(SKIP_REASON, allow_module_level=True)
