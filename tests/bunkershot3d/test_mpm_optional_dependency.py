"""Regression coverage for optional MuJoCo loading in the MPM backend (#8084).

A MuJoCo install that is *present but unloadable* — the usual Windows case
where the wheel is built against a newer MSVC runtime than the host — raises
``OSError(1114, "DLL initialization routine failed")`` from ``import mujoco``,
not ``ImportError``. The driver's guard must degrade to ``mujoco = None``
instead of letting that escape and take the launcher down at import time.

The previous version of this test asserted the contract indirectly, by
importing ``src.shared.python.gui_pkg.draggable_tabs`` and hoping the driver
came along transitively. That coupled a bunker-physics test to the whole Qt +
OpenCV GUI stack (it failed here with ``No module named 'cv2'``) and used the
non-canonical ``src.bunkershot3d`` module name. It now exercises the guard
directly.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.unit

_DRIVER_MODULE = "bunkershot3d.backends.mpm.driver"


@pytest.fixture
def _purge_driver_module() -> Iterator[None]:
    """Drop the driver module so the next import re-executes its guard."""
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == _DRIVER_MODULE or name.startswith(_DRIVER_MODULE + ".")
    }
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        sys.modules.update(saved)


def _import_driver_with_mujoco_raising(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> object:
    """Import the driver with ``import mujoco`` raising *error*."""
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "mujoco":
            raise error
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    return importlib.import_module(_DRIVER_MODULE)


@pytest.mark.usefixtures("_purge_driver_module")
def test_import_survives_mujoco_dll_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OSError(1114)`` from a broken native library must not escape."""
    module = _import_driver_with_mujoco_raising(
        monkeypatch, OSError(1114, "DLL initialization routine failed")
    )

    assert module.mujoco is None
    assert hasattr(module, "MPMDriver"), (
        "the driver class must still be importable when MuJoCo is unusable"
    )


@pytest.mark.usefixtures("_purge_driver_module")
def test_import_survives_mujoco_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original ImportError path must keep working."""
    module = _import_driver_with_mujoco_raising(
        monkeypatch, ImportError("No module named 'mujoco'")
    )

    assert module.mujoco is None


@pytest.mark.usefixtures("_purge_driver_module")
def test_unrelated_import_errors_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not become a blanket ``except``.

    Only a failure to load *mujoco* is tolerated. If some other dependency of
    the driver is broken, that must surface rather than be silently degraded
    into a confusing "MuJoCo unavailable" state.
    """
    real_import = builtins.__import__

    def _fake_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "numpy":
            raise ImportError("numpy is broken")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ImportError, match="numpy is broken"):
        importlib.import_module(_DRIVER_MODULE)
