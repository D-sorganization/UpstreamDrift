"""Regression coverage for optional MuJoCo loading in the MPM backend."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def test_mpm_driver_import_survives_mujoco_dll_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken optional MuJoCo DLL must not prevent launcher startup (#8084)."""
    module_name = "src.bunkershot3d.backends.mpm.driver"
    original_import = builtins.__import__

    def _raise_dll_error_for_mujoco(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "mujoco":
            raise OSError(1114, "DLL initialization routine failed")
        return original_import(name, globals, locals, fromlist, level)

    for cached_name in tuple(sys.modules):
        if cached_name.startswith(
            (
                "src.bunkershot3d",
                "src.shared.python.gui_pkg",
                "src.shared.python.pose_estimation",
                "src.shared.python.simulation_backends",
            )
        ):
            monkeypatch.delitem(sys.modules, cached_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_dll_error_for_mujoco)

    importlib.import_module("src.shared.python.gui_pkg.draggable_tabs")
    driver_module = sys.modules[module_name]

    assert driver_module.mujoco is None
