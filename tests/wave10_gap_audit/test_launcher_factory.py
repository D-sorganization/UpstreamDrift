"""Coverage for src/shared/python/launcher_factory.py."""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.shared.python import launcher_factory


def test_get_engine_module_known() -> None:
    assert launcher_factory.get_engine_module("mujoco") is not None
    assert launcher_factory.get_engine_module("drake") is not None
    assert launcher_factory.get_engine_module("pinocchio") is not None
    assert launcher_factory.get_engine_module("opensim") is not None
    assert launcher_factory.get_engine_module("myosuite") is not None
    assert launcher_factory.get_engine_module("pendulum") is not None


def test_get_engine_module_unknown() -> None:
    assert launcher_factory.get_engine_module("nonexistent_engine") is None


def test_launch_engine_unknown_exits(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        launcher_factory.launch_engine_directly("bogus_engine")
    assert exc.value.code == 1


def test_launch_engine_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("fake_module")
    fake_main = MagicMock()
    fake_module.main = fake_main  # type: ignore[attr-defined]

    def fake_import(_name: str) -> types.ModuleType:
        return fake_module

    monkeypatch.setattr(launcher_factory.importlib, "import_module", fake_import)
    launcher_factory.launch_engine_directly("mujoco")
    fake_main.assert_called_once()


def test_launch_engine_missing_main_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("fake_module_no_main")

    monkeypatch.setattr(
        launcher_factory.importlib,
        "import_module",
        lambda _name: fake_module,
    )
    with pytest.raises(SystemExit) as exc:
        launcher_factory.launch_engine_directly("mujoco")
    assert exc.value.code == 1


def test_launch_engine_import_error_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(_name: str) -> Any:
        raise ImportError("missing dep")

    monkeypatch.setattr(launcher_factory.importlib, "import_module", fake_import)
    with pytest.raises(SystemExit) as exc:
        launcher_factory.launch_engine_directly("mujoco")
    assert exc.value.code == 1
