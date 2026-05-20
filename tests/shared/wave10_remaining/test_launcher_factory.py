"""Tests for src.shared.python.launcher_factory."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from src.shared.python import launcher_factory as lf


@pytest.mark.unit
def test_get_engine_module_known_engines():
    # Every known engine maps to a non-empty module string.
    for name in ("mujoco", "drake", "pinocchio", "opensim", "myosuite", "pendulum"):
        path = lf.get_engine_module(name)
        assert isinstance(path, str)
        assert path  # non-empty


@pytest.mark.unit
def test_get_engine_module_unknown():
    assert lf.get_engine_module("not_a_real_engine") is None


@pytest.mark.unit
def test_engine_modules_constant_shape():
    # Map keys are lowercase identifiers, values are dotted Python paths.
    for k, v in lf.ENGINE_MODULES.items():
        assert k.islower()
        assert "." in v


@pytest.mark.unit
def test_launch_engine_directly_unknown_engine_exits(caplog):
    with pytest.raises(SystemExit) as exc_info:
        lf.launch_engine_directly("does_not_exist")
    assert exc_info.value.code == 1


@pytest.mark.unit
def test_launch_engine_directly_invokes_main(monkeypatch):
    fake_module = types.ModuleType("fake_engine_module")
    main_calls: list[int] = []

    def fake_main() -> None:
        main_calls.append(1)

    fake_module.main = fake_main  # type: ignore[attr-defined]

    fake_path = "test.fake.engine.module"
    monkeypatch.setitem(lf.ENGINE_MODULES, "fake_engine", fake_path)
    monkeypatch.setitem(sys.modules, fake_path, fake_module)

    lf.launch_engine_directly("fake_engine")
    assert main_calls == [1]


@pytest.mark.unit
def test_launch_engine_directly_module_without_main_exits(monkeypatch):
    fake_module = types.ModuleType("fake_no_main_module")
    fake_path = "test.fake.no_main_module"
    monkeypatch.setitem(lf.ENGINE_MODULES, "fake_no_main", fake_path)
    monkeypatch.setitem(sys.modules, fake_path, fake_module)

    with pytest.raises(SystemExit) as exc_info:
        lf.launch_engine_directly("fake_no_main")
    assert exc_info.value.code == 1


@pytest.mark.unit
def test_launch_engine_directly_import_error_exits(monkeypatch):
    monkeypatch.setitem(
        lf.ENGINE_MODULES, "fake_missing", "definitely.not.an.importable.module.xyz"
    )

    def fake_import(name):
        raise ImportError(f"cannot import {name}")

    monkeypatch.setattr(lf.importlib, "import_module", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        lf.launch_engine_directly("fake_missing")
    assert exc_info.value.code == 1


@pytest.mark.unit
def test_launch_engine_directly_uses_get_engine_module(monkeypatch):
    """Verify the function consults ENGINE_MODULES via get_engine_module."""
    called = MagicMock(return_value=None)
    monkeypatch.setattr(lf, "get_engine_module", called)
    with pytest.raises(SystemExit):
        lf.launch_engine_directly("xx")
    called.assert_called_once_with("xx")
