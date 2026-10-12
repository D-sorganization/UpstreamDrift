"""Shared ezc3d access for tests (#12065).

The single test-side route to the Python ``ezc3d`` extension.  It goes through
the production guard so a native OpenSim DLL conflict is diagnosed the same
way, and it skips *only* when ``ezc3d`` is genuinely not installed
(``ModuleNotFoundError``).  Any other ``ImportError`` (a broken wheel, an ABI
or DLL conflict) is a real failure and must surface, not hide as a skip.
"""

from __future__ import annotations

from types import ModuleType

import pytest

from src.shared.python.data_io.native_dll_guard import import_ezc3d

__all__ = ["ezc3d_installed", "require_ezc3d"]

SKIP_REASON = "ezc3d is not installed"


def ezc3d_installed() -> bool:
    """Return True when ``ezc3d`` imports; False only if it is not installed.

    Other ``ImportError`` subclasses propagate.
    """
    try:
        import_ezc3d()
    except ModuleNotFoundError:
        return False
    return True


def require_ezc3d() -> ModuleType:
    """Return the ``ezc3d`` module, or skip the test if it is not installed."""
    try:
        return import_ezc3d()
    except ModuleNotFoundError:
        pytest.skip(SKIP_REASON)
